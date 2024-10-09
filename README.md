# Uni-Med

<font size='5'>**Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE**</font>

Xun Zhu, Ying Hu, Fanbin Mo, Miao Li, Ji Wu <a href='https://arxiv.org/abs/2409.17508'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 

【Accepted】by The Thirty-eighth Annual Conference on Neural Information Processing Systems **(Neurips 2024) [Poster]**

## Getting Started
### Preparing

**1. Environment**

**2. Dataset**

To download the raw data，you can follow：
Dataset | Download path| Dataset | Download path
:---: | :---:|:---: | :---:
MedQA |  [Download](https://github.com/jind11/MedQA) | PubMedQA | [Download](https://github.com/pubmedqa/pubmedqa)
Slake |  [Download](https://www.med-vqa.com/slake)| Path-VQA  |  [Download](https://github.com/UCSD-AI4H/PathVQA)
MIMIC-CXR |  <a href="https://physionet.org/content/mimic-cxr-jpg/2.1.0">images</a> &nbsp;&nbsp;  <a href="https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv"> captions</a>| MPx |  <a href="https://huggingface.co/datasets/chaoyi-wu/MedPix-Images">images</a> &nbsp;&nbsp;  <a href="https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv"> captions</a>
SA-Med2D-20M |  [Download](https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M) | MNIST |  [Download](https://medmnist.com )

You can download the processed data (such as Slake-VQA/Slake-REC/Slake-REG; SA-Med2D-REC/SA-Med2D-REG) on [Google Drive](https://github.com/jind11/MedQA), which can be directly used for training.

**3. Pretrained Model Weights**

EVA-CLIP ViT-G [Download](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)

Llama 2 Chat 7B [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)


### Training

### Evaluation

## Acknowledgement
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) The standard model architecture of Uni-Med follows MiniGPT-v2. Don't forget to check this great open-source work if you don't know it before!

If you're using Uni-Med in your research or applications, please cite using this BibTeX:
```bibtex


@article{zhu2024uni,
  title={Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE},
  author={Zhu, Xun and Hu, Ying and Mo, Fanbin and Li, Miao and Wu, Ji},
  journal={arXiv preprint arXiv:2409.17508},
  year={2024}
}
