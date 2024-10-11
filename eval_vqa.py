import os
import json
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

import numpy as np
from evaluate_metrics import split_sentence, normalize_word
from nltk.translate.bleu_score import sentence_bleu
from uni_med.datasets.datasets.slake_vqa_dataset import SlakeVQAEvalData
from uni_med.datasets.datasets.path_vqa_dataset import PathVQAEvalData

from uni_med.common.eval_utils import init_model, eval_parser
from uni_med.conversation.conversation import Conversation
from uni_med.common.config import Config

def calculate_acc(data):
    acc_scores = []
    for ann in data:
        answer_pred = normalize_word(ann["answer_pred"])
        answer_gt = normalize_word(ann["answer_gt"])
        acc = 1 if answer_gt in answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()
    
    return acc_scores

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

def evaluate_all(uni_med_predict):    
    
    bleu_scores = calculate_bleu(uni_med_predict)
    f1_scores = calculate_f1(uni_med_predict)

    return {
        "bleu_scores": bleu_scores,
        "f1_score": f1_scores,
    }

def evaluate_open_and_closed(uni_med_predict):    
    open_subset = [ann for ann in uni_med_predict if ann["answer_type"] == "OPEN"]
    open_bleu_scores = calculate_bleu(open_subset)
    open_f1_scores = calculate_f1(open_subset)
    
    closed_subset = [ann for ann in uni_med_predict if ann["answer_type"] == "CLOSED"]
    closed_bleu_scores = calculate_bleu(closed_subset)
    closed_f1_scores = calculate_f1(closed_subset)
    closed_acc_scores = calculate_acc(closed_subset)
    
    all_bleu_scores = calculate_bleu(uni_med_predict)
    all_f1_scores = calculate_f1(uni_med_predict)
   
    return {
        "open": {
            "bleu_scores": open_bleu_scores,
            "f1_score": open_f1_scores,
        },
        "closed": {
            "bleu_scores": closed_bleu_scores,
            "f1_score": closed_f1_scores,
            "acc_score": closed_acc_scores,
        },
        "all": {
            "bleu_scores": all_bleu_scores,
            "f1_score": all_f1_scores,
        }
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

if 'slakevqa_en' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["slakevqa_en"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["slakevqa_en"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["slakevqa_en"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["slakevqa_en"]["max_new_tokens"]
    
    img_list = open(os.path.join(eval_file_path, 'test.txt')).read().split()
    
    slakevqa_test_split_en = []
    for i in img_list:
        questions_path = os.path.join(img_path, i, 'question.json')
        questions = json.load(open(questions_path))
        for question in questions:
            if question['q_lang'] == 'en' and len(question['answer']) != 0:
                slakevqa_test_split_en.append(question)
    
    data = SlakeVQAEvalData(slakevqa_test_split_en, vis_processor, text_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    for images, questions, answer_types, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='vqa')

        for question, answer_type, answer_gt , answer_pred \
            in zip(questions, answer_types, answers_gt, answers_pred):

            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip() # todo: change <unk> to unk_token

            result['question'] = question
            result['answer_type'] = answer_type
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)

    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path,"slakevqa_en.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))

    # evaluate the result
    metrics = evaluate_open_and_closed(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))

if 'path_vqa' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["path_vqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["path_vqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["path_vqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["path_vqa"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "test_vqa.pkl")
    with open(evaluation_annntation_path, 'rb') as f:
        pvqa_test_split = pickle.load(f)
    
    data = PathVQAEvalData(pvqa_test_split, vis_processor, text_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    uni_med_predict = []

    for question_ids, images, questions, answer_types, answers_gt in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers_pred = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, task='vqa')

        for question_id, question, answer_type, answer_gt , answer_pred \
            in zip(question_ids, questions, answer_types, answers_gt, answers_pred):

            result = dict()
            answer_pred = answer_pred.lower().replace('<unk>','').strip() # todo: change <unk> to unk_token

            result['question_id'] = int(question_id)
            result['question'] = question
            result['answer_type'] = "CLOSED" if answer_type == "yes/no" else "OPEN"
            result['answer_gt'] = answer_gt
            result['answer_pred'] = answer_pred
            # print(result)

            uni_med_predict.append(result)

    # save the result
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_save_path= os.path.join(save_path,"path_vqa.json")
    with open(file_save_path,'w') as f:
        json.dump(uni_med_predict, f)
    print("save the result to {}".format(file_save_path))

    # evaluate the result
    metrics = evaluate_open_and_closed(uni_med_predict)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f, sort_keys=True)
    print("save the metrics to {}".format(metric_save_path))
    print("metrics: {}".format(metrics))
