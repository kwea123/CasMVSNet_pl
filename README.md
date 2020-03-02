# CasMVSNet_pl
Unofficial implementation of [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/pdf/1912.06378.pdf) using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Official implementation: [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)

Reference MVSNet implementation: [MVSNet_pl](https://github.com/kwea123/MVSNet_pl)

### Update

1.  Implement groupwise correlation in [Learning Inverse Depth Regression for Multi-View Stereowith Correlation Cost Volume](https://arxiv.org/abs/1912.11746). It achieves almost the same result as original variance-based cost volume, but with fewer parameters and consumes lower memory, so it is highly recommended to use (in contrast, the inverse depth sampling in that paper turns out to have no effect in my experiments, maybe because DTU is indoor dataset, and inverse depth improves outdoor dataset better). To activate, set `--num_groups 8` in training.

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
| This repo <br> (gwc**) |  640x512  | 3       | 4.412mm     | 70.50%     | 83.61%        | 90.41%        | **6.5 / ** |

*Training memory is measured on `batch size=2` and `resolution=640x512`.

**Gwc with `num_groups=8`, see [update](#update) 2. This implementation aims at maintaining the concept of cascade cost volume, and build new operations to further increase the accuracy or to decrease inference time/GPU memory.

## Pretrained model and log
Download the pretrained model and training log in [release](https://github.com/kwea123/CasMVSNet_pl/releases/tag/v1.0).
The above metrics of `This repo (same as paper)` correspond to this training but the model is saved on the 10th epoch (least `val_loss` but not the best in other metrics).

# Testing
## Data download
You need to download [full resolution image](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) and unzip to the same folder as your training data, if you want to test the model for higher resolution (I tested with 1152x864 to be aligned with the paper, but we are able to run full resolution 1600x1184 with 5 views, costing only 8.7GB GPU memory).

## Testing model
For testing depth prediction with val/test set, please see `test.ipynb`.

For depth fusion, run `python eval.py --split test --ckpt_path ckpts/exp2/_ckpt_epoch_10.ckpt (--save_visual)`. It will generate depth prediction files under folder `results/depth`; after the depth prediction for all images finished, it will perform depth fusion for all scans and generate `.ply` files under folder `results/points`.

*  You can comment out the `# Step 1.` to do depth fusion only, after the depth prediction are generated.
*  You can add `--scan {scan_number}` to only do depth fusion on specific scans (specify also the correct `--split`). Otherwise the default will process all scans in the `split`.
*  The speed for one scan is about 80s: 40s for depth prediction of 49 ref views and 40s of depth fusion.

The fusion code is heavily borrowed from [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py) with refactoring and the following modifications (without careful investigations, I just think it looks better):
1.  After the depth of the ref view is refined (this is original mvsnet method), I **use the refined depth** for the following runs (1 run=1 ref view and many src views). For example, depth of view 0 is refined in the first run, then the next run, for ref view 1, if it uses view 0 as src view, this time we don't use the original depth prediction, instead we use the refined depth of the previous run since it is generally better (average across many views).
2.  When projecting points into 3d space, I observe that the color of the same point change a lot across views: this is partly due to non-lambertian sufaces such as metal reflection, but is hugely due to light angle difference bewteen views. Since it's an **indoor** dataset, the light angle changes a lot even with a small displacement, making shadows projected to different spaces. In order to alleviate this color inconsistency, in addition to depth average as suggested in the original paper, I also do **color average** to make the point cloud look more consistent.

Finally, to visualize the point cloud, run `python visualize_ply.py --scan {scan_number}`.

## Demo for scan9
I provide the fusion result for **all 119 scans** with the default parameters in `eval.py` in [release](https://github.com/kwea123/CasMVSNet_pl/releases/). Download and put them under `results/points`. A sample viewpoint (put under `results/`) `viewpoint.json` is also provided: add `--use_viewpoint` to use the same viewpoint to do comparison between scans/different fusion approaches! You can also save your own viewpoint by `--save_viewpoint`.

The default viewpoint looks like:
![teaser](assets/demo.png)

Comparison between some open source methods:
<p align="center">
  <img src="assets/cascade.png", width="48%">
  <img src="assets/rmvsnet.png", width="48%">
  <br>
  <img src="assets/demo.png", width="48%">
  <br>
  <sup>Top left: 
     <a href="https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet">Original casmvsnet</a> Top right: <a href="https://github.com/YoYo000/MVSNet">R-MVSNet</a> Bottom: This repo
  </sup>
</p>

Also a video showing the point cloud in different angles (click to link to YouTube):
[![teaser](assets/demo.gif)](https://youtu.be/wCjMoBR9Nh0)
