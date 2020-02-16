# CasMVSNet_pl
Unofficial implementation of [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/pdf/1912.06378.pdf) using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Official implementation: [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)

Reference MVSNet implementation: [MVSNet_pl](https://github.com/kwea123/MVSNet_pl)

# Installation

## Hardware

* OS: Ubuntu 16.04 or 18.04
* NVIDIA GPU with **CUDA>=10.0** (tested with 1 RTX 2080Ti)

## Software

* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    * Install [Inplace-ABN](https://github.com/mapillary/inplace_abn) by `pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11`

# Data download

Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet) and unzip. For the description of how the data is created, please refer to the original paper.

# Training

Run (example)
```
python train.py \
   --root_dir $DTU_DIR \
   --num_epochs 16 --batch_size 2 \
   --depth_interval 2.65 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
   --optimizer adam --lr 1e-3 --lr_scheduler cosine \
   --exp_name exp
```

Note that the model consumes huge GPU memory, so the batch size is generally small. For reference, the above command requires 8483MB of GPU memory.

See [opt.py](opt.py) for all configurations.

## Example training log
![log1](assets/log1.png)
![log2](assets/log2.png)

## Metrics
The metrics are collected on the DTU val set.

|           | resolution* | abs_err | 1mm acc | 2mm acc    | 4mm acc    |
| :---:     |  :---:     | :---:   |  :---:  | :---:      | :---:      |
| Paper     |  1152x864  | N/A     | N/A     | 82.6%      | 88.8%      |
| This repo |  640x512   | 4.524mm | 72.33%  | **84.35%** | **90.52%** |

*Generally,larger resolution leads to better accuracy and lower error.

## Pretrained model and log
Download the pretrained model and training log in [release](https://github.com/kwea123/CasMVSNet_pl/releases/tag/v1.0).
The above metrics correspond to this training but the model is saved on the 10th epoch (least `val_loss` but not the best in other metrics).
