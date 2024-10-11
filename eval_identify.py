import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluate_metrics import split_sentence, normalize_word
from nltk.translate.bleu_score import sentence_bleu
from uni_med.common.config import Config
from uni_med.common.eval_utils import prepare_texts, init_model, eval_parser
from uni_med.conversation.conversation import Conversation

from uni_med.datasets.datasets.slake_dataset import InvReferSlakeDataset_Eval
from uni_med.datasets.datasets.sa_med_dataset import InvReferSAMedDataset_Eval

def calculate_bleu(data):
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []

    for ann in data:
        answer_gt = normalize_word(ann["answer_gt"]).split()
        answer_pred = normalize_word(ann["answer_pred"]).split()

        bleu1_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1, 0, 0, 0)))
        bleu2_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./2., 1./2.)))
        bleu3_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./3., 1./3., 1./3.)))
        bleu4_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./4., 1./4., 1./4., 1./4.)))

    bleu1 = np.array(bleu1_list).mean()
    bleu2 = np.array(bleu2_list).mean()
    bleu3 = np.array(bleu3_list).mean()
    bleu4 = np.array(bleu4_list).mean()

    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4
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

def calculate_acc(data):
    acc_scores = []
    for ann in data:
        answer_pred = normalize_word(ann["answer_pred"])
        answer_gt = normalize_word(ann["answer_gt"])
        acc = 1 if answer_gt == answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()
    
    return acc_scores

def evaluate(uni_med_predict):
    
    bleu_scores = calculate_bleu(uni_med_predict)
    f1_scores = calculate_f1(uni_med_predict)
    acc_scores = calculate_acc(uni_med_predict)

    return {
        "bleu_scores": bleu_scores,
        "f1_score": f1_scores,
        "acc_score": acc_scores,
    }

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='invref_slake', help="dataset to evaluate")
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



if 'invref_slake' in args.dataset:
    data_dir = cfg.evaluation_datasets_cfg["invref_slake"]["data_dir"]
    batch_size = cfg.evaluation_datasets_cfg["invref_slake"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["invref_slake"]["max_new_tokens"]

    data = InvReferSlakeDataset_Eval(vis_processor, text_processor, data_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='identify')
        for answer_pred, question, img_id, answer_gt in zip(answers_pred, questions, img_ids, answers_gt):
            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip()
            
            result['image_id'] = img_id
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "invref_slake.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    # evaluate the result
    metrics = evaluate(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))


if 'invref_sa_med' in args.dataset:
    image_dir = cfg.evaluation_datasets_cfg["invref_sa_med"]["image_dir"]
    region_dir = cfg.evaluation_datasets_cfg["invref_sa_med"]["region_dir"]
    batch_size = cfg.evaluation_datasets_cfg["invref_sa_med"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["invref_sa_med"]["max_new_tokens"]

    data = InvReferSAMedDataset_Eval(vis_processor, text_processor, image_dir, region_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='identify')
        for answer_pred, question, img_id, answer_gt in zip(answers_pred, questions, img_ids, answers_gt):
            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip()
            
            result['image_id'] = img_id
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "invref_sa_med.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    # evaluate the result
    metrics = evaluate(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))


if 'invref_sa_med' in args.dataset:
    image_dir = cfg.evaluation_datasets_cfg["invref_sa_med"]["image_dir"]
    region_dir = cfg.evaluation_datasets_cfg["invref_sa_med"]["region_dir"]
    batch_size = cfg.evaluation_datasets_cfg["invref_sa_med"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["invref_sa_med"]["max_new_tokens"]

    data = InvReferSAMedDataset_Eval(vis_processor, text_processor, image_dir, region_dir)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    iou_scores = []
    for images, questions, img_ids, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='identify')
        for answer_pred, question, img_id, answer_gt in zip(answers_pred, questions, img_ids, answers_gt):
            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip()
            
            result['image_id'] = img_id
            result['question'] = question
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)
    
    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path, "invref_sa_med.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))
    
    # evaluate the result
    metrics = evaluate(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))