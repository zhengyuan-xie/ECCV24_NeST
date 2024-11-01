# Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation (ECCV 2024)

An official code for "[Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](https://arxiv.org/abs/2407.14142)"

This repository contains the official implementation of the following paper:

> Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation<br>Zhengyuan Xie, Haiquan Lu, Jia-wen Xiao, Enguang Wang, Le Zhang, Xialei Liu\*<br>European Conference on Computer Vision (**ECCV**), 2024<br>
## Update
- [√] Release the preliminary code and scripts for NeST. More scripts and the cleaned code will be released after careful testing.

## Introduction
Class incremental semantic segmentation aims to preserve old knowledge while learning new tasks, however, it is impeded by catastrophic forgetting and background shift issues. Prior works indicate the pivotal importance of initializing new classifiers and mainly focus on transferring knowledge from the background classifier or preparing classifiers for future classes, neglecting the flexibility and variance of new classifiers. In this paper, we propose a new classifier pre-tuning (NeST) method applied before the formal training process, learning a transformation from old classifiers to generate new classifiers for initialization rather than directly tuning the parameters of new classifiers. Our method can make new classifiers align with the backbone and adapt to the new data, preventing drastic changes in the feature extractor when learning new classes. Besides, we design a strategy considering the cross-task class similarity to initialize matrices used in the transformation, helping achieve the stability-plasticity trade-off. Experiments on Pascal VOC 2012 and ADE20K datasets show that the proposed strategy can significantly improve the performance of previous methods. 
<p align="center">
<img src="assert/github_NEST_system_design.png" alt="Image description" width="776.16" height="435.6">
</p>

## Dataset Preparation

- PASCAL VOC 2012

```
sh data/download_voc.sh
```

- ADE20K

```
sh data/download_ade.sh
```

## Environment

1. PyTorch environment:

```
conda create -n NeST python=3.8
conda activate NeST
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install packaging
```

2. APEX:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

3. inplace-abn and other packages:

```
pip install inplace-abn
pip install matplotlib
pip install captum
```

## Training

1. Dowload pretrained model from [ResNet-101_iabn](https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar) and [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth) to ```pretrained/```
2. We have prepared some training scripts in ```scripts/```. First please modify the dataset path in scripts, then you can train the model by

```
sh scripts/voc/plop+ours_resnet101_15-1.sh
```
All experiemnts are conducted on 4 NVIDIA RTX 3090 GPUs.

## Reference

If our code and paper help you, please kindly cite:

```
@inproceedings{xie2024early,
  title={Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation},
  author={Xie, Zhengyuan and Lu, Haiquan and Xiao, Jia-wen and Wang, Enguang and Zhang, Le and Liu, Xialei},
  booktitle={European Conference on Computer Vision},
  pages={183--201},
  year={2024}
}
```

## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

If you have any question, please contact <a href="zhengyuanxie2000@gmail.com">zhengyuanxie2000@gmail.com</a> 

## Acknowledgement

This code is based on [[MiB]](https://github.com/fcdl94/MiB) and [[PLOP]](https://github.com/arthurdouillard/CVPR2021_PLOP). We highly appreciate their contributions to this community.
