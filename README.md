<h2 align="center">A Simple yet Universal Framework for Depth Completion</h2>

<p align="center">
  <a href="https://jinhwipark.com/"><strong>Jin-Hwi Park</strong></a> · 
  <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon</strong></a>
  <br>
  <strong>NeurIPS 2024</strong><br>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=Y4tHp5Jilp">
    <strong><code>📄 Paper</code></strong>
  </a>
  <a href="https://www.jinhwipark.com/Depth-with-Sensors">
    <strong><code>🌐 Project Page</code></strong>
  </a>
  <a href="https://github.com/JinhwiPark/UniDC">
    <strong><code>💻 Source Code</code></strong>
  </a>
</p>


## The source code contains
 - Our implementation of UniDC
 - Train code for NYU, KITTI dataset

## Requirements
 - python==3.8.18
 - torch==1.9.0+cu111
 - torchvision==0.10.0+cu111
 - h5py
 - tqdm
 - scipy
 - matplotlib
 - nuscenes-devkit
 - imageio
 - pillow==9.5.0
```
 pip install opencv-python
 apt-get update
 apt-get -y install libgl1-mesa-glx -y
 apt-get -y install libglib2.0-0 -y
 pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

## Data Preparation

### NYU Depth V2 data Preparation
Please download the preprocessed NYU Depth V2 dataset in HDF5 formats provided by Fangchang Ma.
```bash
mkdir data; cd data
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
mv nyudepthv2 nyudepth_hdf5
```
After that, you will get a data structure as follows:
```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

### KITTI data Preparation
Please download the KITTI DC dataset at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.

After downloading datasets, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI DC dataset.
```
cd src/utils
python prepare_KITTI_DC.py --path_root_dc PATH_TO_KITTI_DC --path_root_raw PATH_TO_KITTI_RAW
```
After that, you will get a data structure as follows:
```
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```



### [NYU Depth V2] Training & Testing

```
# Train
python main_DP.py --data_name NYU --dir_data {Dataset Directory} --gpus 0 --num_sample random --batch_size 1 --model_name depth_prompt_main --save OURS-NYU --patch_height 240 --patch_width 320 --prop_kernel 9 --prop_time 18 --init_scaling --loss L1L2_SILogloss_init
```

### [KITTI Depth Completion] Training & Testing
```
# Train
python main_DP.py --data_name KITTIDC --dir_data {Dataset Directory} --gpus 0 --top_crop 100 --lidar_lines random_lidar --batch_size 1 --model_name depth_prompt_main --save OURS-KITTI --patch_height 240 --patch_width 1216 --prop_kernel 9 --prop_time 18 --conf_prop --init_scaling --loss L1L2_SILogloss_init 
```

### Acknowledgement
This code is based on the original implementations: 
[CSPN](https://github.com/XinJCheng/CSPN)([paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.html)), 
[NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20)([paper](https://arxiv.org/abs/2007.10042)),

```
@inproceedings{
  park2024a,
  title     = {A Simple yet Universal Framework for Depth Completion},
  author    = {Jin-Hwi Park and Hae-Gon Jeon},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024},
  url       = {https://openreview.net/forum?id=Y4tHp5Jilp}
}
```




