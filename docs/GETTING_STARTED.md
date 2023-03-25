# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation

Currently we provide the dataloader of KITTI dataset and NuScenes dataset, and the supporting of more datasets are on the way.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to use the depth maps for trainval set, download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI trainval set
* Download the [KITTI Raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and put in into data/kitti/raw/KITTI_Raw
* (optional) If you want to use the [sparse depth maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) for KITTI Raw, download it and put it into data/kitti/raw/depth_sparse

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
│   │   │── raw
|   |   |   |——calib & KITTI_Raw & (optional: depth_sparse)
├── pcdet
├── tools
```

* Generate the data infos by running the following command (kitti train, kitti val, kitti test): 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

* Generate the data infos by running the following command (kitti train + eigen clean, unlabeled):
```python 
python -m pcdet.datasets.kitti.kitti_dataset_cmkd create_kitti_infos_unlabel tools/cfgs/dataset_configs/kitti_dataset.yaml
```
### Waymo Open Dataset
<!-- * Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics.  -->
On the way

### Nuscenes Dataset
Please refer to [this link](https://github.com/Cc-Hy/CMKD-MV).

## Pretrained Models

If you would like to use some pretrained models, download them and put them into ../checkpoints
```
OpenPCDet
├── checkpoints
|   ├── second_teacher.pth
|   ├── ···
├── data
├── pcdet
├── tools
```
## Training & Testing for LiDAR-based Teacher Models

Please see [the official OpenPCDet instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to train or test a LiDAR-based teacher model.


## Training & Testing for Image-based Student Models
### Evaluate the pretrained models
* Test with a pretrained model: 
```python
python test_cmkd.py --cfg ${CONFIG_FILE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```python
python test_cmkd.py --cfg ${CONFIG_FILE} --ckpt_dir ${CKPT_DIR}  --eval_all
```

* To test with multiple GPUs:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 test_cmkd.py --launcher pytorch --cfg ${CONFIG_FILE} --tcp_port 16677 --ckpt ${CKPT}
```

### Train a model

* Train with a single GPU:
```python
python train_cmkd.py --cfg_file ${CONFIG_FILE} --pretrained_lidar_model ${TEACHER_MODEL_PATH}
```

* Train with multiple GPUs or multiple machines
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ${CONFIG_FILE} --tcp_port 16677 --pretrained_lidar_model ${TEACHER_MODEL_PATH}
```

* We recommand to use the following training skill (BEV optimization first, then detection)
```
python train_cmkd.py --cfg xxx_bev.yaml ···
python train_cmkd.py --cfg xxx.yaml --pretrained_img_model ${BEV_pretrained_model_path} ···
```

* To reproduce our results with SECOND teacher, use
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ../tools/cfgs/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_bev.yaml --tcp_port 16677 --pretrained_lidar_model ../checkpoints/second_teacher.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ../tools/cfgs/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_V2.yaml  --tcp_port 16677 --pretrained_lidar_model ../checkpoints/second_teacher.pth
--pretrained_img_model ../output/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_bev/default/ckpt/checkpoint_epoch_30.pth
```
