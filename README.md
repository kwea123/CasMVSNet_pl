# CasMVSNet_pl
Unofficial implementation of [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/pdf/1912.06378.pdf) using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Official implementation: [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)

Reference MVSNet implementation: [MVSNet_pl](https://github.com/kwea123/MVSNet_pl)

### Update

1.  Implement groupwise correlation in [Learning Inverse Depth Regression for Multi-View Stereowith Correlation Cost Volume](https://arxiv.org/abs/1912.11746). It achieves almost the same result as original variance-based cost volume, but with fewer parameters and consumes lower memory, so it is highly recommended to use (in contrast, the inverse depth sampling in that paper turns out to have no effect in my experiments). To activate, set `--num_groups 8` in training.

2.  Since MVS models consumes a lot of GPU memory, it is indispensable to do some code tricks to reduce GPU memory consumption. I tried the followings:
    *  Replace `BatchNorm+Relu` with [Inplace-ABN](https://github.com/mapillary/inplace_abn): Reduce the memory by ~15%!
    *  `del` the tensor when it is never accessed later: Only helps a little.
    *  Use `a = a+b` in training and `a += b` in testing: Reduce about 300MB (don't know the reason..)

# Installation

## Hardware

* OS: Ubuntu 16.04 or 18.04
* NVIDIA GPU with **CUDA>=10.0** (tested with 1 RTX 2080Ti)

## Software

* Python>=3.6.1 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n casmvsnet_pl python=3.6` to create a conda environment and activate it by `conda activate casmvsnet_pl`)
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

Note that the model consumes huge GPU memory, so the batch size is generally small.

See [opt.py](opt.py) for all configurations.

## Example training log
![log1](assets/log1.png)
![log2](assets/log2.png)

## Metrics
The metrics are collected on the DTU val set.

|           | resolution | n_views | abs_err | 1mm acc | 2mm acc    | 4mm acc    | GPU mem in GB <br> (train*/val) |
| :---:     |  :---:     | :---:   | :---:   |  :---:  | :---:      | :---:      | :---:   |
| Paper     |  1152x864  | 5       | N/A     | N/A     | 82.6%      | 88.8%      | N/A / 5.3 |
| This repo <br> (same as paper) |  640x512   | 3       | 4.524mm | 72.33%  | 84.35%     | 90.52%     | 8.5 / 2.1 |
| This repo <br> (gwc**) |  1152x864  | 5       | N/A     | N/A     | N/A        | N/A        | **6.5 / 4.5** |

*Training memory is measured on `batch size=2` and `resolution=640x512`.

**Gwc with `num_groups=8`, see [update](#update) 2. This implementation aims at maintaining the concept of cascade cost volume, and build new operations to further increase the accuracy or to decrease inference time/GPU memory.

## Pretrained model and log
Download the pretrained model and training log in [release](https://github.com/kwea123/CasMVSNet_pl/releases/tag/v1.0).
The above metrics of `This repo (original)` correspond to this training but the model is saved on the 10th epoch (least `val_loss` but not the best in other metrics).

# Testing
For testing depth prediction with val/test set, please see `test.ipynb`.

For depth fusion, run `python eval.py --split test --ckpt_path ckpts/exp2/_ckpt_epoch_10.ckpt (--save_visual)`. It will generate depth prediction files under folder `results/depth`; after the depth prediction for all images finished, it will perform depth fusion for all scans and generate `.ply` files under folder `results/points`. Feel free to comment out the `# Step 1.` to experiment different depth fusion parameters, or replace `dataset.scans` in `L181` with a specific scan (e.g. `scan9`).

The fusion code is heavily borrowed from [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py) with refactoring.

Finally, to visualize the point cloud, run `python visualize_ply.py --scan {scan_number}`.

## Demo for scan9
I provide the fusion result for scan9 with the default parameters in `eval.py` in [release](https://github.com/kwea123/CasMVSNet_pl/releases/tag/v1.1).

Also A video showing how the point cloud look like (click to link to YouTube):
[![teaser](assets/demo.gif)](https://youtu.be/wCjMoBR9Nh0)
