# CMKD: Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection (Coming Soon !)

## Paper
[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://arxiv.org/abs/2211.07171) (arXiv, Supplimentary Included)

[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_6) (ECCV Open Access)

[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://storage.googleapis.com/waymo-uploads/files/research/3DCam/3DCam_CMKD.pdf) (Waymo Challenge Report)

If you find our papers helpful for your research, you may cite our paper as
```
@inproceedings{YuH-CMKD-ECCV2022,
author = {Yu Hong and
Hang Dai and
Yong Ding},
title = {Cross-Modality Knowledge
Distillation Network for Monocular 3D Object
Detection},
booktitle = {{ECCV}},
series = {Lecture Notes in Computer Science},
publisher = {Springer},
year = {2022}
}
```



## Introduction
We are preparing the code, please stay tuned.

This is the official implementation of CMKD with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for KITTI and Waymo datasets.

We have also implemented another version with [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for Nuscenes dataset.

## News
**[2022.7.18] We release the first version covering the KITTI experiments.**

This implementation has some differences from our paper, but the core idea is the same.
Overall, the current version is faster to train, uses less memory, and has similar performance to the older version. We'll release instructions and more implementations later.

**[2022.7.9] Our paper has been accepted by ECCV 2022 as Oral presentation.** :fire::fire::fire:

**[2022.7.4] Our paper has been accepted by ECCV 2022.** :fire::fire:

**[2022.5.24] CMKD gets the 3rd place in the 2022 Waymo 3D camera-only detection challenge.** :fire:

In the challenge, we simply extend our baseline model from single-camera version to multi-camera version without any challenge-specific skills and achieve good results. 
Specifically, we use a lightweight res-50 backbone with 20% of the total training samples, no previous frames, no data augmentation, and no training and testing tricks to rank 3rd in the challenge.

## Framework Overview
![image](https://user-images.githubusercontent.com/82150240/177261849-be867420-d9e2-49f2-9b1f-0209e383b754.png)

## BEV Features Generation
![image](https://user-images.githubusercontent.com/82150240/177973195-8e04f2d1-f945-4332-bfff-49c8568a9c4d.png)

