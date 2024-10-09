import os
import random
import jsonlines

from torch.utils.data import Dataset


class MedQADataset(Dataset):
    def __init__(self, text_processor, ann_path, source):
        self.text_processor = text_processor

        self.ann = []
        with jsonlines.open(os.path.join(ann_path, 'questions', source, 'train.jsonl'), 'r') as f:
            self.ann.extend(f)
        
        self.instruction_pool =[
            "[qa] {}",
            "[qa] If you are a doctor, please answer the following question briefly: {}"
        ]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        instruction = random.choice(self.instruction_pool).format(info["question"])
        answer = info["answer"]

        return {
            "instruction_input": self.text_processor(instruction),
            "answer": self.text_processor(answer),
        }

class MedQADataset_en(MedQADataset):
    def __init__(self, text_processor, ann_path):
        super().__init__(text_processor, ann_path, source='US')
