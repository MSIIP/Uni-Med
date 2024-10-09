from torch.utils.data import Dataset, ConcatDataset
from medmnist import DermaMNIST, OrganSMNIST

class FormatedDataset(Dataset):
    def __init__(self, dataset, question):
        self.dataset = dataset
        self.question = question
        self.labels = dataset.info['label']

        for k in self.labels:
            v = self.labels[k]
            if v.endswith("-left"):
                v = v.replace("-left", "")
                v = "left " + v
            if v.endswith("-right"):
                v = v.replace("-right", "")
                v = "right " + v
            self.labels[k] = v
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image = self.dataset[index][0]
        answer = self.labels[str(self.dataset[index][1].item())]

        return image, self.question, answer

class MedMNISTDataset_2D_Train_Base(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.config = {
            'split': "train",
            'root': vis_root,
            'size': 224,
            'mmap_mode': 'r',
        }

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        ann = self.dataset[index]

        return {
            "image": self.vis_processor(ann[0].convert("RGB")),
            "question": ann[1],
            "answer": ann[2],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = "<Img><ImageHere></Img> [cls] {} ".format(data['question'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }


class MedMNISTDataset_2D_small(MedMNISTDataset_2D_Train_Base):
    def __init__(self, vis_processor, text_processor, vis_root):
        super().__init__(vis_processor, text_processor, vis_root)
        self.dataset = ConcatDataset([
            FormatedDataset(DermaMNIST(**self.config), question="Which category does this multi-source dermatoscopic image of common pigmented skin lesions belong to: actinic keratoses and intraepithelial carcinoma, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, or vascular lesions?"),
            FormatedDataset(OrganSMNIST(**self.config), question="Which category does this CT image belong to: bladder, left femur, right femur, heart, left kidney, right kidney, liver, left lung, right lung, pancreas, or spleen?"),
        ])


class MedMNISTDataset_2D_Eval_Base(Dataset):
    def __init__(self, vis_processor, text_processor, data_dir):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data_dir = data_dir

        self.config = {
            'split': "test",
            'root': data_dir,
            'size': 224,
            'mmap_mode': 'r',
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ann = self.dataset[index]

        image = self.vis_processor(ann[0].convert("RGB"))
        question = "[cls] " + ann[1]
        answer = self.text_processor(ann[2])

        return index, image, question, answer

class MedMNISTDataset_2D_Derma_Eval(MedMNISTDataset_2D_Eval_Base):
    def __init__(self, vis_processor, text_processor, data_dir):
        super().__init__(vis_processor, text_processor, data_dir)
        self.dataset = FormatedDataset(DermaMNIST(**self.config), question="Which category does this multi-source dermatoscopic image of common pigmented skin lesions belong to: actinic keratoses and intraepithelial carcinoma, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, or vascular lesions?")

class MedMNISTDataset_2D_OrganS_Eval(MedMNISTDataset_2D_Eval_Base):
    def __init__(self, vis_processor, text_processor, data_dir):
        super().__init__(vis_processor, text_processor, data_dir)
        self.dataset = FormatedDataset(OrganSMNIST(**self.config), question="Which category does this CT image belong to: bladder, left femur, right femur, heart, left kidney, right kidney, liver, left lung, right lung, pancreas, or spleen?")