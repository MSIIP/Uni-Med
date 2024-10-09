import os
import random
import json

from torch.utils.data import Dataset


class PubMedQADataset(Dataset):
    def __init__(self, text_processor, ann_path):
        self.text_processor = text_processor

        self.ann = []
        with open(os.path.join(ann_path, 'train_set.json'), 'r') as f:
            for k, v in json.load(f).items():
                self.ann.append(v)
        
        self.instruction_pool =[
            "[qa] {}",
            "[qa] If you are a doctor, please answer the following question using 'yes', 'no' or 'maybe': {}"
        ]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        instruction = random.choice(self.instruction_pool).format(info["QUESTION"])
        answer = info["final_decision"]

        return {
            "instruction_input": self.text_processor(instruction),
            "answer": self.text_processor(answer),
        }
        