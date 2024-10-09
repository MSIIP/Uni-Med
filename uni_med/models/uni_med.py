import re 
import math

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from uni_med.common.registry import registry
from uni_med.models.uni_med_base import UniMedBase

import torch.nn.functional as F

@registry.register_model("uni_med")
class UniMed(UniMedBase):
    """
    UniMed model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/uni_med.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            resample_rate = 4,
            resample_method = "projection",
            projector_type="linear",
            llm_model_name="",
            llm_model_path="",
            sft_type="lora",
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            max_txt_len=300,
            end_sym='\n',
            use_grad_checkpoint_llm=False,
            prompt_template='[INST] {} [/INST]',
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            has_moe=False,
            router_method = 'router_task_token',
            num_experts=3,
            task_token_c =768,
            router_type="soft",
            sparse_topk=2,
            num_task_tokens=10,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llm_model_name=llm_model_name,
            llm_model_path=llm_model_path,
            sft_type=sft_type,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_txt_len=max_txt_len,
            end_sym=end_sym,      
            prompt_template=prompt_template,
            max_context_len=max_context_len,
            low_resource=low_resource,
            device_8bit=device_8bit, 
        )

        self.resample_rate = resample_rate
        self.resample_method = resample_method
        self.projector_type = projector_type
        
        self.task_tokens = nn.ParameterDict()
        self.tasks = ['vqa', 'refer', 'identify', 'caption', 'cls']
        self.num_task_tokens = num_task_tokens
        self.task_token_c = task_token_c
        for task in self.tasks:
            if self.num_task_tokens != 0:
                self.task_tokens[task] = nn.Parameter(torch.empty([1, self.num_task_tokens, self.task_token_c]))
                nn.init.normal_(self.task_tokens[task], std=0.02)
            else:
                self.task_tokens[task] = None
        
        self.has_moe = has_moe
        self.num_experts = num_experts
        self.router_type = router_type
        self.router_method = router_method
        self.sparse_topk = sparse_topk
        self.moe_layers = nn.ModuleDict()

        if self.resample_method == 'projection':
            img_f_dim = self.visual_encoder.num_features * self.resample_rate
        else:
            img_f_dim = self.visual_encoder.num_features
                
        if self.projector_type=="linear":
            projector = nn.Linear(img_f_dim, self.llm_model.config.hidden_size)
            
        elif self.projector_type=="moe_linear":
            for expert in range(self.num_experts):
                expert = str(expert)
                self.moe_layers[expert] = nn.ModuleList()
                self.moe_layers[expert].append(nn.Linear(img_f_dim, self.llm_model.config.hidden_size))
            
            projector = self.moe_layers

            if self.router_type == 'soft' or 'sparse':
                if self.router_method == 'router_task_token':
                    self.router = Mlp(img_f_dim + self.task_token_c, (img_f_dim + self.task_token_c) * 4, self.num_experts)
                elif self.router_method == 'router_token':
                    self.router = Mlp(img_f_dim, img_f_dim * 4, self.num_experts)
                elif self.router_method == 'router_task':
                    self.router = Mlp(self.task_token_c, self.task_token_c * 4, self.num_experts)
            
        elif self.projector_type=="moe_mlp":
            for expert in range(self.num_experts):
                expert = str(expert)
                self.moe_layers[expert] = nn.ModuleList()
                self.moe_layers[expert].append(nn.Linear(img_f_dim, self.llm_model.config.hidden_size))
                self.moe_layers[expert].append(nn.GELU()) 
                self.moe_layers[expert].append(nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)) 
                
            projector = self.moe_layers

            if self.router_type == 'soft' or 'sparse':
                if self.router_method == 'router_task_token':
                    self.router = Mlp(img_f_dim + self.task_token_c, (img_f_dim + self.task_token_c) * 4, self.num_experts)
                elif self.router_method == 'router_token':
                    self.router = Mlp(img_f_dim, img_f_dim * 4, self.num_experts)
                elif self.router_method == 'router_task':
                    self.router = Mlp(self.task_token_c, self.task_token_c * 4, self.num_experts)
        
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(img_f_dim, self.llm_model.config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU()) 
                    modules.append(nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)) 
                projector = nn.Sequential(*modules)
            else:
                raise ValueError(f"projector_type {self.projector_type} not supported")
        
        self.llm_proj = projector
            
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llm_model.gradient_checkpointing_enable()

    def encode_img(self, image, task):
        device = image.device
        routing_weights = None

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            if self.resample_method == 'projection':
                image_embeds = image_embeds.view(bs, int(pn/self.resample_rate), int(hs * self.resample_rate))  # print(image_embeds.shape)[bs, 64, 5632]
            
            else :
                image_embeds = image_embeds.view(bs,int(math.sqrt(pn)),int(math.sqrt(pn)),hs)
                image_embeds=image_embeds.permute(0,3,1,2)
                if self.resample_method == 'avgpool':
                    image_embeds= F.avg_pool2d(image_embeds, kernel_size = 2, stride = 2)  
                elif self.resample_method == 'maxpool':
                    image_embeds= F.max_pool2d(image_embeds, kernel_size = 2, stride = 2)
                image_embeds=image_embeds.permute(0,2,3,1)
                image_embeds=image_embeds.view(bs, -1, hs)  
            
            if self.task_tokens[task] != None:
                task_tokens_embeds = self.task_tokens[task].repeat(bs,int(pn/self.resample_rate), 1)
                mix_embeds = torch.cat([self.task_tokens[task].repeat(bs,int(pn/self.resample_rate), 1), image_embeds], dim=2)  

            if self.has_moe:
                if self.projector_type=="moe_linear" or "moe_mlp":

                    if self.router_type == "soft":
                        if self.router_method == "router_task_token": 
                            routing_weights = self.router(mix_embeds).sigmoid()
                        elif self.router_method == "router_token":      
                            routing_weights = self.router(image_embeds).sigmoid()
                        elif self.router_method == "router_task":      
                            routing_weights = self.router(task_tokens_embeds).sigmoid()
                        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                    
                    elif self.router_type=="hard":
                        expert_idx = self.tasks.index(task)  # Assuming task_name is the name of the current task
                        routing_weights = torch.zeros(bs, image_embeds.size(1), self.num_experts).to(device)
                        routing_weights[:, :, expert_idx] = 1.0
                        
                    elif self.router_type=="constant":
                        routing_weights = (torch.ones(bs, image_embeds.size(1), self.num_experts) / self.num_experts).to(device)
                        
                    elif self.router_type=="sparse":
                        if self.router_method == "router_task_token": 
                            logits = self.router(mix_embeds)
                        elif self.router_method == "router_token": 
                            logits = self.router(image_embeds)
                        elif self.router_method == "router_task":
                            logits = self.router(task_tokens_embeds)
                                
                        top_k_logits, indices = logits.topk(self.sparse_topk, dim=-1)
                        zeros = torch.full_like(logits, float('-inf'))
                        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
                        routing_weights = F.softmax(sparse_logits, dim=-1).to(device)

                    image_embeds_experts = []
                    for expert_id in range(self.num_experts):
                        image_embeds_expert = image_embeds
                        for layer in self.llm_proj[str(expert_id)]:
                            image_embeds_expert = layer(image_embeds_expert)
                        
                        routing_weight = routing_weights[:, :, expert_id]
                        image_embeds_expert = image_embeds_expert * routing_weight[:, :, None]
                        image_embeds_experts.append(image_embeds_expert)
                        
                    inputs_llm = sum(image_embeds_experts)
                    
            else:
                inputs_llm = self.llm_proj(image_embeds)
        
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                
        return inputs_llm, atts_llm, routing_weights

    @classmethod
    def from_config(cls, cfg):

        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        
        resample_rate = cfg.get("resample_rate",4)
        resample_method = cfg.get("resample_method","projection")
        projector_type = cfg.get("projector_type")
        
        llm_model_name = cfg.get("llm_model_name")
        llm_model_path = cfg.get("llm_model_path")
        sft_type = cfg.get("sft_type", "lora")
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')
        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_context_len = cfg.get("max_context_len", 3800)

        low_resource = cfg.get("low_resource", False)
        
        has_moe = cfg.get("has_moe", False)
        num_experts = cfg.get("num_experts", 3)
        router_type = cfg.get("router_type", "soft")
        router_method = cfg.get("router_method", "router_task_token")
        sparse_topk = cfg.get("sparse_topk", 2)
        num_task_tokens = cfg.get("num_task_tokens", 10)
        task_token_c = cfg.get("task_token_c", 768)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            resample_rate=resample_rate,
            resample_method=resample_method,
            projector_type=projector_type,
            llm_model_name=llm_model_name,
            llm_model_path=llm_model_path,
            sft_type=sft_type,
            lora_target_modules=lora_target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,    
            prompt_template=prompt_template,   
            max_context_len=max_context_len,
            low_resource=low_resource,
            
            has_moe=has_moe,
            num_experts=num_experts,
            router_type=router_type,
            router_method=router_method,
            sparse_topk=sparse_topk,
            num_task_tokens=num_task_tokens,
            task_token_c= task_token_c,
        )

        ckpt_path = cfg.get("ckpt", "") 
        if ckpt_path:
            print("Load Uni-Med Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
 