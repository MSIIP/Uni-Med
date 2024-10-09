import os
import json
import random

from PIL import Image
from torch.utils.data import Dataset


class SlakeVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, data_dir, lang):

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data_dir = data_dir

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        img_list = open(os.path.join(data_dir, 'train.txt')).read().split()

        exist_annotation = []
        for i in img_list:
            questions_path = os.path.join(data_dir, 'imgs', i, 'question.json')
            questions = json.load(open(questions_path))
            for question in questions:
                if question['q_lang'] == lang and len(question['answer']) != 0:
                    exist_annotation.append(question)

        self.annotation = exist_annotation
    
    def __len__(self):
        return len(self.annotation)


    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.data_dir, 'imgs', ann["img_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer = ann["answer"]

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }


class SlakeVQADataset_en(SlakeVQADataset):
    def __init__(self, vis_processor, text_processor, data_dir):
        super().__init__(vis_processor, text_processor, data_dir, lang="en")

class SlakeVQAEvalData(Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.loaded_data)


    def __getitem__(self, index):
        ann = self.loaded_data[index]

        image_path = os.path.join(self.root_path, ann["img_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        answer_type = ann["answer_type"]
        answer = self.text_processor(ann["answer"])

        return image, question, answer_type, answer
