import os
import pandas as pd
import random
from PIL import Image
import json
from torch.utils.data import Dataset


class MimicCaptionDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=[]):
    
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            mimic_caption = json.load(open(ann_path))["train"]

            for row in mimic_caption:
                image_path = os.path.join(self.vis_root, "files", row["image_path"][0])
                if os.path.exists(image_path):
                    self.annotation.append(row)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            'Describe the given chest x-ray image in detail.',
            'Take a look at this chest x-ray and describe the findings and impression.',
            'Could you provide a detailed description of the given x-ray image?',
            'Describe the given chest x-ray image as detailed as possible.',
            'What are the key findings in this chest x-ray image?',
        ]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, "files", ann["image_path"][0])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["report"])

        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
            "image": image,
            "answer": caption,
            "instruction_input": instruction,
        }


class MimicCaptionEvalData(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=[]):
        
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            mimic_caption = json.load(open(ann_path))["test"]

            for row in mimic_caption:
                image_path = os.path.join(self.vis_root, "files", row["image_path"][0])
                if os.path.exists(image_path):
                    self.annotation.append(row)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
    
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, "files", ann["image_path"][0])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["report"])

        instruction = "[caption] Describe the given chest x-ray image in detail. "

        return ann["image_path"][0], image, instruction, caption