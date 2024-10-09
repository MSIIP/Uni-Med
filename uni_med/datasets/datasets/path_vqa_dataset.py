import os
import random
import pickle
from PIL import Image

from torch.utils.data import Dataset


class PathVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=[]):

        self.vis_root = vis_root

        annotation = []
        for ann_path in ann_paths:
            ann = pickle.load(open(ann_path, "rb"))
            annotation.extend(ann)
        
        # self.annotation = []
        # for ann in annotation:
        #     if len(ann['sent']) < 50 and len(list(ann['label'].keys())[0]) < 10:
        #         self.annotation.append(ann)
            
        self.annotation = annotation

        self.split = self.annotation[0]['img_id'].split('_')[0]
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def __len__(self):
        return len(self.annotation)

    def get_data(self, index):
        ann = self.annotation[index]

        image_id = ann["img_id"]
        image_path = os.path.join(self.vis_root, self.split, image_id + '.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["sent"])
        question_id = ann["question_id"]

        answer = list(ann['label'].keys())[0]

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

class PathVQAEvalData(Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.loaded_data)


    def __getitem__(self, index):
        ann = self.loaded_data[index]

        image_id = ann["img_id"]
        image_path = os.path.join(self.root_path, 'test', image_id+'.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["sent"])
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        question_id = ann["question_id"]
        answer_type = ann["answer_type"]
        answer = self.text_processor(list(ann['label'].keys())[0])

        return question_id, image, question, answer_type, answer