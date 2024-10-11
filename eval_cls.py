import os
import json

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluate_metrics import normalize_word
from uni_med.datasets.datasets.medmnist_dataset import MedMNISTDataset_2D_Derma_Eval, MedMNISTDataset_2D_OrganS_Eval

from uni_med.common.eval_utils import init_model, eval_parser
from uni_med.conversation.conversation import Conversation
from uni_med.common.config import Config

def calculate_acc(data):
    acc_scores = []
    for ann in data:
        answer_pred = normalize_word(ann["answer_pred"])
        answer_gt = normalize_word(ann["answer_gt"])
        acc = 1 if answer_gt == answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()
    
    return acc_scores

def evaluate_all(uni_med_predict):    
    
    acc_scores = calculate_acc(uni_med_predict)

    return {
        "accuracy": acc_scores,
    }


def prepare_texts(texts, conv_temp, img_temp='<Img><ImageHere></Img>'):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], f'{img_temp} {text}') for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor, text_processor = init_model(args)
bos_token = model.llm_tokenizer.bos_token
conv_temp = Conversation(
    system="",
    roles=(f"{bos_token}[INST] ", " [/INST]"),
    messages=[],
    sep="",
    offset=2,
).copy()
model.eval()
cfg_save_path = cfg.run_cfg.save_path
save_path = os.path.join(cfg_save_path, str(args.dataset[0]))


def eval_2d(dataset_name, dataset_cls):
    data_dir = cfg.evaluation_datasets_cfg["medmnist_2d"]["data_dir"]
    batch_size = cfg.evaluation_datasets_cfg["medmnist_2d"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["medmnist_2d"]["max_new_tokens"]

    data = dataset_cls(vis_processor, text_processor, data_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    for indexes, images, questions, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='cls')

        for index, question, answer_gt , answer_pred \
            in zip(indexes, questions, answers_gt, answers_pred):

            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip() # todo: change <unk> to unk_token

            result['index'] = index.item()
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)

    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path,f"{dataset_name}.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))

    # evaluate the result
    metrics = evaluate_all(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))

if 'medmnist_2d_derma' in args.dataset:
    eval_2d("medmnist_2d_derma", MedMNISTDataset_2D_Derma_Eval)

if 'medmnist_2d_organs' in args.dataset:
    eval_2d("medmnist_2d_organs", MedMNISTDataset_2D_OrganS_Eval)