import os
import json

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from evaluate_metrics import split_sentence, normalize_word
from uni_med.datasets.datasets.mimic_caption_dataset import MimicCaptionEvalData
from uni_med.datasets.datasets.medpix_dataset import MedPixSingleEvalData

from uni_med.common.eval_utils import init_model, eval_parser
from uni_med.conversation.conversation import Conversation
from uni_med.common.config import Config

def json_trans(data):
    gt = {}
    pred ={}
    for i in range(len(data)):
        answer_pred = data[i]["answer_pred"]
        answer_gt = data[i]["answer_gt"]
        
        pred[i] = answer_pred
        gt[i] = answer_gt
        
    return pred, gt

def to_coco_format(caption):
    coco = {}
    for k, v in caption.items():
        coco[k] = [{'caption': v}]
    return coco

def pycocoeval(data, scorers):
    tokenizer = PTBTokenizer()
    pred, gt=json_trans(data)
    pred = to_coco_format(pred)
    gt = to_coco_format(gt)
    pred = tokenizer.tokenize(pred)
    gt = tokenizer.tokenize(gt)

    eval = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gt, pred)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval[m] = sc
        else:
            eval[method] = score
    
    return eval

def calculate_bleu(data):
    scorers = [
        (Bleu(4), ["bleu1", "bleu2", "bleu3", "bleu4"]),
    ]

    result = pycocoeval(data, scorers)
    return result

def calculate_meteor(data):
    for i in range(len(data)):
        data[i]["answer_pred"] = normalize_word(data[i]["answer_pred"])
        data[i]["answer_gt"] = normalize_word(data[i]["answer_gt"])
    
    scorers = [
        (Meteor(),"meteor")
    ]

    result = pycocoeval(data, scorers)
    return result['meteor']

def calculate_rouge(data):
    rouge = Rouge()

    rouge1_list = []
    rouge2_list = []
    rougel_list = []

    for ann in data:
        answer_gt = ann["answer_gt"]
        answer_pred = ann["answer_pred"]
        
        answer_pred = normalize_word(answer_pred)
        answer_gt = normalize_word(answer_gt)
        
        if not answer_gt or not answer_pred:
            continue

        rouge_score = rouge.get_scores(answer_pred, answer_gt)

        rouge1_list.append(rouge_score[0]["rouge-1"]["f"])
        rouge2_list.append(rouge_score[0]["rouge-2"]["f"])
        rougel_list.append(rouge_score[0]["rouge-l"]["f"])

    rouge1 = np.array(rouge1_list).mean()
    rouge2 = np.array(rouge2_list).mean()
    rougel = np.array(rougel_list).mean()

    return {
        'rouge_1': rouge1,
        'rouge_2': rouge2,
        'rouge_l': rougel,
    }

def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    
    if len(candidate_words) == 0:
        return 0
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)

def calculate_f1(data):
    f1_list = []
    
    for ann in data:
        f1_list.append(calculate_f1score(ann["answer_pred"], ann["answer_gt"]))
        
    f1 = np.array(f1_list).mean()
    
    return f1

def evaluate_all(uni_med_predict):    
    
    uni_med_predict_filtered = [ann for ann in uni_med_predict if ann['answer_gt'] and ann['answer_pred']]

    bleu_scores = calculate_bleu(uni_med_predict_filtered)
    meteor_scores = calculate_meteor(uni_med_predict_filtered)
    rouge_scores = calculate_rouge(uni_med_predict_filtered)
    f1_scores = calculate_f1(uni_med_predict_filtered)

    return {
        "bleu_scores": bleu_scores,
        "meteor_scores": meteor_scores,
        "rouge_scores": rouge_scores,
        "f1_score": f1_scores,
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


if 'mimic_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["mimic_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["mimic_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["mimic_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["mimic_caption"]["max_new_tokens"]
    
    ann_paths = [os.path.join(eval_file_path, 'annotation.json')]
    data = MimicCaptionEvalData(vis_processor, text_processor, img_path, ann_paths)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    for image_paths, images, questions, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='caption')

        for image_path, question, answer_gt , answer_pred \
            in zip(image_paths, questions, answers_gt, answers_pred):

            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip()

            result['image_path'] = image_path
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)

    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path,"mimic_caption.json")
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


if 'medpix_single' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["medpix_single"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["medpix_single"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["medpix_single"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["medpix_single"]["max_new_tokens"]
    
    ann_paths = [os.path.join(eval_file_path, 'MedPix_single_test.csv')]
    data = MedPixSingleEvalData(vis_processor, text_processor, img_path, ann_paths)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    for image_paths, images, questions, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='caption')

        for image_path, question, answer_gt , answer_pred \
            in zip(image_paths, questions, answers_gt, answers_pred):

            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip()

            result['image_path'] = image_path
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred

            uni_med_predict.append(result)

    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path,"medpix_single.json")
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
    