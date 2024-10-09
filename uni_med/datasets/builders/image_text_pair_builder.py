import logging

from uni_med.common.registry import registry
from uni_med.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from uni_med.datasets.datasets.slake_dataset import ReferSlakeDataset, InvReferSlakeDataset
from uni_med.datasets.datasets.slake_vqa_dataset import SlakeVQADataset_en
from uni_med.datasets.datasets.path_vqa_dataset import PathVQADataset
from uni_med.datasets.datasets.medqa_dataset import MedQADataset_en
from uni_med.datasets.datasets.pubmedqa_dataset import PubMedQADataset
from uni_med.datasets.datasets.mimic_caption_dataset import MimicCaptionDataset
from uni_med.datasets.datasets.medmnist_dataset import MedMNISTDataset_2D_small
from uni_med.datasets.datasets.sa_med_dataset import ReferSAMedDataset, InvReferSAMedDataset
from uni_med.datasets.datasets.medpix_dataset import MedPixSingleDataset
    

@registry.register_builder("slakevqa_en")
class SlakeVQABuilder_en(BaseDatasetBuilder):
    train_dataset_cls = SlakeVQADataset_en
    DATASET_CONFIG_DICT = {"default": "configs/datasets/slake/vqa_en.yaml"}

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        data_dir = build_info.data_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_dir=data_dir,
        )

        return datasets
    

@registry.register_builder("ref_slake")
class RefSlakeBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReferSlakeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/slake/ref.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        data_dir = build_info.data_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_dir=data_dir,
        )

        return datasets


@registry.register_builder("invref_slake")
class InvRefSlakeBuilder(BaseDatasetBuilder):
    train_dataset_cls = InvReferSlakeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/slake/invref.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        data_dir = build_info.data_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_dir=data_dir,
        )

        return datasets


@registry.register_builder("path_vqa")
class PathVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PathVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/path_vqa/defaults.yaml"}

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_path = build_info.images.storage

        datasets = dict()

        # annotation path
        ann_paths = ann_info.get("train").storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        # create datasets
        datasets["train"] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=vis_path,
        )

        return datasets


@registry.register_builder("medqa_en")
class MedQABuilder_en(BaseDatasetBuilder):
    train_dataset_cls = MedQADataset_en
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlp/medqa_en.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets


@registry.register_builder("pubmedqa")
class PubMedQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PubMedQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlp/pubmedqa.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets


@registry.register_builder("mimic_caption")
class MimicCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = MimicCaptionDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_caption/defaults.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_path = build_info.images.storage

        datasets = dict()

        # annotation path
        ann_paths = ann_info.get("train").storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        # create datasets
        datasets["train"] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=vis_path,
        )

        return datasets


@registry.register_builder("medmnist_2d_small")
class MedMNISTBuilder_2D_small(BaseDatasetBuilder):
    train_dataset_cls = MedMNISTDataset_2D_small
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medmnist/2d_small.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        vis_path = build_info.images.storage

        datasets = dict()

        # create datasets
        datasets["train"] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_path,
        )

        return datasets


@registry.register_builder("ref_sa_med")
class RefSAMedBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReferSAMedDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sa_med/ref.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_dir = build_info.image_dir
        region_dir = build_info.region_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            image_dir=image_dir,
            region_dir=region_dir
        )

        return datasets


@registry.register_builder("invref_sa_med")
class InvRefSAMedBuilder(BaseDatasetBuilder):
    train_dataset_cls = InvReferSAMedDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sa_med/invref.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_dir = build_info.image_dir
        region_dir = build_info.region_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            image_dir=image_dir,
            region_dir=region_dir
        )

        return datasets


@registry.register_builder("medpix_single")
class MedPixSingleBuilder(BaseDatasetBuilder):
    train_dataset_cls = MedPixSingleDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medpix/defaults.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_path = build_info.images.storage

        datasets = dict()

        # annotation path
        ann_paths = ann_info.get("train").storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        # create datasets
        datasets["train"] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=vis_path,
        )

        return datasets