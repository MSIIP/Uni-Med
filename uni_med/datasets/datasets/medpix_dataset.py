import os
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset


class MedPixSingleDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=[]):
    
        self.vis_root = vis_root

        self.case_list = pd.read_csv(ann_paths[0])
        self.case_list = self.case_list.dropna()
        self.case_list = self.case_list.loc[self.case_list['type'] == 'caption']
        self.case_list = self.case_list[self.case_list.apply(lambda x: os.path.exists(os.path.join(vis_root, x['name'])), axis=1)]
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Describe this input image.",
            "Help captioning the image.",
            "What can be inflected from the scan?",
            "Can you give a caption for this image?",
            "Can you provide a brief summary of the radiology image?",
            "Please write a report about the image?",
            "Can you provide an analysis of this image?",
            "Can you explain what is shown in this image?",
            "What can be indicated from the radiologic scans?",
            "What can you infer from this photograph?",
        ]
    
    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        sample = self.case_list.iloc[index]

        image_path = os.path.join(self.vis_root, sample['name'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = self.text_processor(sample['context'])
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
            "image": image,
            "answer": caption,
            "instruction_input": instruction,
        }


class MedPixSingleEvalData(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=[]):
    
        self.vis_root = vis_root

        self.case_list = pd.read_csv(ann_paths[0])
        self.case_list = self.case_list.dropna()
        self.case_list = self.case_list.loc[self.case_list['type'] == 'caption']
        self.case_list = self.case_list[self.case_list.apply(lambda x: os.path.exists(os.path.join(vis_root, x['name'])), axis=1)]
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
    
    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        sample = self.case_list.iloc[index]

        image_path = os.path.join(self.vis_root, sample['name'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = self.text_processor(sample['context'])
        instruction = "[caption] Describe this input image. "

        return image_path, image, instruction, caption