# CMKD: Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection (ECCV 2022 Oral)

## Paper
[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://arxiv.org/abs/2211.07171) (arXiv, Supplimentary Included)

[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_6) (ECCV Open Access)

[Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://storage.googleapis.com/waymo-uploads/files/research/3DCam/3DCam_CMKD.pdf) (Waymo Challenge Report)

If you find our papers helpful for your research, you may cite our paper as
```
@inproceedings{yuhong-CMKD-ECCV2022,
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
This is the official implementation of CMKD with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for KITTI and Waymo datasets.

We have also implemented [another version](https://github.com/Cc-Hy/CMKD-MV) with [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for Nuscenes dataset.

## News
**[2023.3.25] We have several updates.**
* Release the multi-camera version CMKD for Nuscenes dataset, refer to [this link](https://github.com/Cc-Hy/CMKD-MV) for detail
* Release more pre-trained models for KITTI dataset


**[2023.2.14] We have several updates.**

**Notice: Due to the short schedule, instructions and pre-trained models will be released gradually in the near future, and there may be many issues and bugs, please feel free to let us know if you have any questions.**

* Support center head in response distillation
* Support more teacher models in the framework (We now have SECOND, CenterPoint and PointPillar), more pre-trained models will be released later
* Support to set different feature level as the distillation guidance
* Add visualization utils to visualize the BEV feature maps and the detection results
* Support for Nuscenes dataset will be released very soon




**[2022.11.20] We release some instructions and pre-trained models covering the KITTI experiments.**

This implementation has some differences from our paper, but the core idea is the same.
Overall, the current version is faster to train, uses less memory, and has similar performance to the older version. 

Waymo experiments and Nuscenes experiments are on the way.

**[2022.7.9] Our paper has been accepted by ECCV 2022 as Oral presentation.** :fire::fire::fire:

**[2022.7.4] Our paper has been accepted by ECCV 2022.** :fire::fire:

**[2022.5.24] CMKD gets the 3rd place in the 2022 Waymo 3D camera-only detection challenge.** :fire:

In the challenge, we simply extend our baseline model from single-camera version to multi-camera version without any challenge-specific skills and achieve good results. 
Specifically, we use a lightweight res-50 backbone with 20% of the total training samples, no previous frames, no data augmentation, and no training and testing tricks to rank 3rd in the challenge.

## Framework Overview
![image](/docs/framework.png)

## BEV Features Generation
![image](/docs/BEV%20generation.png)

## Use CMKD

### Installation

Please follow [INSTALL](docs/INSTALL.md) to install CMKD.

### Getting Started

Please follow [GETTING_START](docs/GETTING_STARTED.md) to train or evaluate the models.

## Models

### KITTI

|   | Teacher Model|  Car Easy@R40|	Car Moderate@R40	|Car Hard@R40	 | Model | Teacher Model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_eigen_R50_scd_V2.yaml)| [SECOND](tools/cfgs/kitti_models/CMKD/CMKD-scd/second_teacher.yaml) |  33.36  | 21.61  | 17.97  |  [model](https://drive.google.com/file/d/1A9rdGUdLkqOWVt8IbZfF1s0FHotQUIK_/view?usp=share_link)   | [model](https://drive.google.com/file/d/1SYbReQHNjOsWQ-zxM6mn3dq9Iyi98wAg/view?usp=share_link) |
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_R50_scd_V2.yaml)|[SECOND](tools/cfgs/kitti_models/CMKD/CMKD-scd/second_teacher.yaml)|  24.02  | 15.80  | 13.22  |  [model](https://drive.google.com/file/d/1weEb8DkAHKNa4HPgzM_Pbc7FLr-Yiuii/view?usp=share_link)  | [model](https://drive.google.com/file/d/1SYbReQHNjOsWQ-zxM6mn3dq9Iyi98wAg/view?usp=share_link) |
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-ctp/cmkd_kitti_eigen_R50_ctp_V2.yaml)|[CenterPoint](tools/cfgs/kitti_models/CMKD/CMKD-ctp/centerpoint_teacher.yaml)|  29.78  | 21.17  | 18.41  |  [model](https://drive.google.com/file/d/1fhXf5UZ0fat9ihdApCTuAVUE8ozttNej/view?usp=share_link)  |[model](https://drive.google.com/file/d/1Oqmnl6Kctg5BRKHEgtAw7ef9Lw3eyPky/view?usp=share_link)|
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-ctp/cmkd_kitti_R50_ctp_V2.yaml)|[CenterPoint](tools/cfgs/kitti_models/CMKD/CMKD-ctp/centerpoint_teacher.yaml)|  22.56  | 16.02  | 13.52  |  [model](https://drive.google.com/file/d/1tuZdy_S4EYeGaH8Mu5nDMOTu8b1PpfG6/view?usp=share_link)  |[model](https://drive.google.com/file/d/1Oqmnl6Kctg5BRKHEgtAw7ef9Lw3eyPky/view?usp=share_link)|
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-pp/cmkd_kitti_eigen_R50_pp_V2.yaml)    |[PointPillar](tools/cfgs/kitti_models/CMKD/CMKD-pp/pointpillar_teacher.yaml)|  32.25  | 21.47  | 18.21  |  [model](https://drive.google.com/file/d/1yX70t4pyTTaJr0X9lzivp0JEwB4uwOTz/view?usp=share_link)  | [model](https://drive.google.com/file/d/1JvpBqNCcJjfASs86q7Qp3772eaJU3wnL/view?usp=share_link)|
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-pp/cmkd_kitti_R50_pp_V2.yaml)|[PointPillar](tools/cfgs/kitti_models/CMKD/CMKD-pp/pointpillar_teacher.yaml)|  23.84  | 16.44 | 13.58  | [model](https://drive.google.com/file/d/1tHTLoBi2m5OqpTVM9biY4ExOZ40PfTIB/view?usp=share_link)  |[model](https://drive.google.com/file/d/1JvpBqNCcJjfASs86q7Qp3772eaJU3wnL/view?usp=share_link)|



### Waymo
Coming Soon
                  

### Nuscenes
|   |  mAP |	NDS |Model | 
|---|:---:|:---:|:---:|
| BEVDet-R50|  30.7  | 38.2  | - |
| BEVDet-R50 + CMKD|  34.7  | 42.6  | - |





