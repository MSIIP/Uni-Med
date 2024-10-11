import os
import re
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from uni_med.common.config import Config
from uni_med.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from uni_med.conversation.conversation import Conversation

from uni_med.datasets.datasets.slake_dataset import ReferSlakeDataset_Eval
from uni_med.datasets.datasets.sa_med_dataset import ReferSAMedDataset_Eval

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='ref_slake', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
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
res=args.res



if 'ref_slake' in args.dataset:
    data_dir = cfg.evaluation_datasets_cfg["ref_slake"]["data_dir"]
    batch_size = cfg.evaluation_datasets_cfg["ref_slake"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ref_slake"]["max_new_tokens"]

    data = ReferSlakeDataset_Eval(vis_processor, text_processor, data_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, bboxes, image_sizes in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='refer')
        for answer, question, img_id, bbox, image_size in zip(answers, questions, img_ids, bboxes, image_sizes):
            result = dict()

            gt_bbox = [0,0,0,0]
            bbox = bbox.tolist()
            gt_bbox[0] = bbox[0]
            gt_bbox[1] = bbox[1]
            gt_bbox[2] = bbox[0] + bbox[2]
            gt_bbox[3] = bbox[1] + bbox[3]

            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                integers = re.findall(r'\d+', answer)
                pred_bbox = [int(num) for num in integers][:4]
                width = image_size.tolist()[0]
                height = image_size.tolist()[1]
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                iou_score = computeIoU(pred_bbox, gt_bbox)
                iou_scores.append(iou_score)
                
                result['pred_bbox'] = pred_bbox
                result['iou_score'] = iou_score
            else:
                iou_scores.append(0)
                result['pred_bbox'] = f'Unable to match: {answer}'
                result['iou_score'] = 0
            
            result['image_id'] = img_id
            result['item'] = question.replace('[refer] give me the location of','').strip()
            result['gt_bbox'] = gt_bbox
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "ref_slake.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    iou_scores = np.array(iou_scores)
    metrics = {'miou': iou_scores.mean(), 'acc': (iou_scores>0.5).mean()}
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))


if 'ref_sa_med' in args.dataset:
    image_dir = cfg.evaluation_datasets_cfg["ref_sa_med"]["image_dir"]
    region_dir = cfg.evaluation_datasets_cfg["ref_sa_med"]["region_dir"]
    batch_size = cfg.evaluation_datasets_cfg["ref_sa_med"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ref_sa_med"]["max_new_tokens"]

    data = ReferSAMedDataset_Eval(vis_processor, text_processor, image_dir, region_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, bboxes, image_sizes in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='refer')
        for answer, question, img_id, bbox, image_size in zip(answers, questions, img_ids, bboxes, image_sizes):
            result = dict()

            gt_bbox = [0,0,0,0]
            bbox = bbox.tolist()
            gt_bbox[0] = bbox[0]
            gt_bbox[1] = bbox[1]
            gt_bbox[2] = bbox[0] + bbox[2]
            gt_bbox[3] = bbox[1] + bbox[3]

            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                integers = re.findall(r'\d+', answer)
                pred_bbox = [int(num) for num in integers][:4]
                width = image_size.tolist()[0]
                height = image_size.tolist()[1]
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                iou_score = computeIoU(pred_bbox, gt_bbox)
                iou_scores.append(iou_score)
                
                result['pred_bbox'] = pred_bbox
                result['iou_score'] = iou_score
            else:
                iou_scores.append(0)
                result['pred_bbox'] = f'Unable to match: {answer}'
                result['iou_score'] = 0
            
            result['image_id'] = img_id
            result['item'] = question.replace('[refer] give me the location of','').strip()
            result['gt_bbox'] = gt_bbox
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "ref_sa_med.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    iou_scores = np.array(iou_scores)
    metrics = {'miou': iou_scores.mean(), 'acc': (iou_scores>0.5).mean()}
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))


if 'ref_sa_med' in args.dataset:
    image_dir = cfg.evaluation_datasets_cfg["ref_sa_med"]["image_dir"]
    region_dir = cfg.evaluation_datasets_cfg["ref_sa_med"]["region_dir"]
    batch_size = cfg.evaluation_datasets_cfg["ref_sa_med"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ref_sa_med"]["max_new_tokens"]

    data = ReferSAMedDataset_Eval(vis_processor, text_processor, image_dir, region_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, bboxes, image_sizes in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='refer')
        for answer, question, img_id, bbox, image_size in zip(answers, questions, img_ids, bboxes, image_sizes):
            result = dict()

            gt_bbox = [0,0,0,0]
            bbox = bbox.tolist()
            gt_bbox[0] = bbox[0]
            gt_bbox[1] = bbox[1]
            gt_bbox[2] = bbox[0] + bbox[2]
            gt_bbox[3] = bbox[1] + bbox[3]

            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                integers = re.findall(r'\d+', answer)
                pred_bbox = [int(num) for num in integers][:4]
                width = image_size.tolist()[0]
                height = image_size.tolist()[1]
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                iou_score = computeIoU(pred_bbox, gt_bbox)
                iou_scores.append(iou_score)
                
                result['pred_bbox'] = pred_bbox
                result['iou_score'] = iou_score
            else:
                iou_scores.append(0)
                result['pred_bbox'] = f'Unable to match: {answer}'
                result['iou_score'] = 0
            
            result['image_id'] = img_id
            result['item'] = question.replace('[refer] give me the location of','').strip()
            result['gt_bbox'] = gt_bbox
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "ref_sa_med.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    iou_scores = np.array(iou_scores)
    metrics = {'miou': iou_scores.mean(), 'acc': (iou_scores>0.5).mean()}
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))
