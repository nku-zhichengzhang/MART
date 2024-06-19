<div align="center">


# <img src="./assets/logo.png" style="vertical-align: sub;" width="400">
**MART: Masked Affective RepresenTation Learning via Masked Temporal Distribution Distillation**


<i>Zhicheng Zhang, Pancheng Zhao, Eunil Park, Jufeng Yang</i>

<a href=" "><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/CVPR-2024-orange)](https://openaccess.thecvf.com/CVPR2024)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)
</div>




**TL:DR** *We present MART, an MAE-style method for learning robust affective representation of videos that exploits the sentiment complementary and emotion intrinsic among temporal segments.*


This repository contains the official implementation of our work in CVPR 2024. The pytorch code for **MART** are released. More details can be viewed in our paper.<be>

## Publication

>**MART: Masked Affective RepresenTation Learning via Masked Temporal Distribution Distillation**<br>
Zhicheng Zhang, Pancheng Zhao, Eunil Park, Jufeng Yang<br>
<i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024</i>.</br>
[<a href="[../assets/2023_ICCV_MPOT.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_MART_Masked_Affective_RepresenTation_Learning_via_Masked_Temporal_Distribution_Distillation_CVPR_2024_paper.pdf)" target="_blank">PDF</a>]
[<a href="https://zzcheng.top/assets/pdf/2024_CVPR_MART_poster.pdf" target="_blank">Poster</a>]
[<a href="https://zzcheng.top/MART" target="_blank">Project Page</a>]
[<a href="https://github.com/nku-zhichengzhang/MART" target="_blank">Github</a>]


</br>




# ABSTRACT

Limited training data is a long-standing problem for video emotion analysis (VEA). Existing works leverage the power of large-scale image datasets for transferring while failing to extract the temporal correlation of affective cues in the video. Inspired by psychology research and empirical theory, we verify that the degree of emotion may vary in different segments of the video, thus introducing the sentiment complementary and emotion intrinsic among temporal segments. We propose an MAE-style method for learning robust affective representation of videos via masking, termed MART. First, we extract the affective cues of the lexicon and verify the extracted one by computing its matching score with video content. The hierarchical verification strategy is proposed, in terms of sentiment and emotion, to identify the matched cues alongside the temporal dimension. Then, with the verified cues, we propose masked affective modeling to recover temporal emotion distribution. We present temporal affective complementary learning that pulls the complementary part and pushes the intrinsic part of masked multimodal features, for learning robust affective representation. Under the constraint of affective complementary, we leverage cross-modal attention among features to mask the video and recover the degree of emotion among segments. Extensive experiments on five benchmark datasets demonstrate the superiority of our method in video sentiment analysis, video emotion recognition, multimodal sentiment analysis, and multimodal emotion recognition.


# DEPENDENCY


### Recommended Environment
* CUDA 11.1
* Python 3.6
* Pytorch 1.8.0

You can prepare your environment by running the following lines.

We prepare a frozen conda environment [`env`](./env.yaml) that can be directly copied.
```
conda env create -f ./env.yaml
```   





# SCRIPTS
## Preparation

**Dataset:** We preprocess the datasets under the following process via the scripts provided in [`tools`](./tools).

**Dataset Structure:** The processed datasets are constructed under the following structure.

```
VAA_VideoEmotion8
├── imgs
│   ├── Anger
│   ├── Anticipation
│   └── Disgust
│   └── Fear
│   └── Joy
│   └── Sadness
│   └── Surprise
│   └── Trust
├── mp3 
│   ├── Anger
│   ├── Anticipation
│   └── Disgust
│   └── Fear
│   └── Joy
│   └── Sadness
│   └── Surprise
│   └── Trust
├── srt 
│   ├── Anger
│   ├── Anticipation
│   └── Disgust
│   └── Fear
│   └── Joy
│   └── Sadness
│   └── Surprise
│   └── Trust
└── ve8.json
```

**Pre-trained Model:** Download the pretrain models from [[`google drive`](https://drive.google.com/drive/folders/1VwuBEJ7RPkpfi4SqUb84jSZhy1zfTYTJ?usp=drive_link)/[`baidu netdisk`](https://pan.baidu.com/s/1DMKH0Tfc_SHJFpyxEq9HbA?pwd=riu6)].

Place the audioset_10_10_0.4593.pth at './models/ast/pretrained_models'

Place the vit_base_patch16_224.pth at './models/mbt/pretrained_models/vit_base_patch16_224'

## Run
You can easily train and evaluate the model by running the script below.

You can include more details such as epoch, milestone, learning_rate, etc. Please refer to [`opts`](opts.py).

~~~~
sh run.sh
~~~~






# REFERENCE
We referenced the repos below for the code.

* [VAANet](https://github.com/maysonma/VAANet)
* [MBT](https://github.com/google-research/scenic/tree/main/scenic/projects/mbt)
* [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
* [MAEst](https://github.com/facebookresearch/mae_st)
* [MaskFeat](https://github.com/facebookresearch/pytorchvideo)
* [MILAN](https://github.com/zejiangh/MILAN)


# CITATION

If you find this repo useful in your project or research, please consider citing the relevant publication.

````
@inproceedings{zhanZhang_2024_CVPRg2024multiple,
  title={Mart: Masked affective representation learning via masked temporal distribution distillation},
  author={Zhang, Zhicheng and Zhao, Pancheng and Park, Eunil and Yang, Jufeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
````