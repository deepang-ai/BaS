# BaS: Efficient Breast Lesion Segmentation from Ultrasound Videos Across Multiple Source-limited Platforms

🎉 This work is published in [IEEE Journal of Biomedical and Health Informatics](hhttps://ieeexplore.ieee.org/document/10892059)

# Abstract
**Medical video segmentation** is fundamentally important in clinical diagnosis and treatment procedures, offering dynamic tracking of breast lesions across frames in ultrasound videos for improved segmentation performance. However, existing approaches face challenges in striking a **balance between segmentation performance and inference speed**, hindering real-time application in resource-constrained medical environments. In order to address these limitations, we present **BaS, a blazing-fast on-device breast lesion segmentation model**. BaS integrates the Stem module and BaSBlock to refine representations through inter- and intra-frame analysis on ultrasound videos. In addition, we release two versions of BaS: the BaS-S for superior segmentation performance and the BaS-L for accelerated inference times. Experimental Results indicate that BaS surpasses the top-performing models in terms of segmenting efficiency and accuracy of predictions on devices with limited resources. This work advances the development of efficient medical video segmentation frameworks applicable to multiple medical platforms. [CodeLink](https://github.com/deepang-ai/BaS)

# Network Architecture

![Overview](./figures/architecture.png)

# Data Description

## BUV 2022
Dataset Name: BUV 2022

Modality: RGB

Size: **63** unique video sequences, totaling 4619 frames, exhibit a range of spatial resolutions from 580x600 to 600x800 pixels.

The dataset is organized based on lesion malignancy, featuring *10 videos of malignant tumors and 53 of benign tumors*, and further categorizes breast nodules into 7 instances of breast cancer, 48 of breast fibroma, and 8 of breast lipoma. 

[Download Link]()

## US-VOS
Dataset Name: US-VOS

Modality: RGB

Size: 141 video sequences with spatial resolutions ranging between 580 × 600 and 600 × 800 pixels.

Dataset Paper: [Cascaded Inner-Outer Clip Retformer for Ultrasound Video Object Segmentation](https://ieeexplore.ieee.org/document/10706869)

# Training

## Install Environment
```bash
pip install -r requirements.txt
```

## Training Script
The ```config.yml``` is the global parameters control file. Dataset loading and related parameter selection are controlled through the ```config.yml```.

Train Bas
```bash
accelecate launch train.py
```

# Visualization
![outcome](./figures/outcome.png)

# Bixtex
```bib
@ARTICLE{10892059,
  author={Pang, Yan and Li, Yunhao and Huang, Teng and Liang, Jiaming and Ding, Ziyu and Chen, Hao and Zhao, Baoliang and Hu, Ying and Zhang, Zheng and Wang, Qiong},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Efficient Breast Lesion Segmentation from Ultrasound Videos Across Multiple Source-limited Platforms}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Videos;Barium;Feature extraction;Ultrasonic imaging;Semantics;Medical diagnostic imaging;Lesions;Breast;Computational modeling;Accuracy;Breast Lesion Segmentation;Ultrasound Video;Resource-limited Application;On-device Models},
  doi={10.1109/JBHI.2025.3543435}}
```