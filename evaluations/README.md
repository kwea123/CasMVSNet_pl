Here I provide the description of the general depth fusion method. The same method is applied for all datasets, if not specified otherwise, the testing uses the default parameters in `eval.py`.

# Depth fusion description

The fusion code is heavily borrowed from [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py) with refactoring and the following modifications (without careful investigations, I just think it looks better):
1.  After the depth of the ref view is refined (this is original mvsnet method), I **use the refined depth** for the following runs (1 run=1 ref view and many src views). For example, depth of view 0 is refined in the first run, then the next run, for ref view 1, if it uses view 0 as src view, this time we don't use the original depth prediction, instead we use the refined depth of the previous run since it is generally better (average across many views).
2.  When projecting points into 3d space, I observe that the color of the same point change a lot across views: this is partly due to non-lambertian sufaces such as metal reflection, but is hugely due to light angle difference bewteen views. Since it's an **indoor** dataset, the light angle changes a lot even with a small displacement, making shadows projected to different spaces. In order to alleviate this color inconsistency, in addition to depth average as suggested in the original paper, I also do **color average** to make the point cloud look more consistent.

# Running depth fusion

## Data download

### DTU
You need to download [full resolution image](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) and unzip to the same folder as your training data, if you want to test the model for higher resolution (I tested with 1152x864 to be aligned with the paper, but we are able to run full resolution 1600x1184 with 5 views, costing only 8.7GB GPU memory).

### Tanks and temples
Download [preprocessed files](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) from MVSNet, these files contains camera poses calculated by [COLMAP](https://github.com/colmap/colmap).

### BlendedMVS
Download files from [BlendedMVS](https://github.com/YoYo000/BlendedMVS) and mail the author for high-resolution images if you want (I use high-res images for evaluation).

## Depth fusion

From the base directory of this repo, run
```
python eval.py \
  --dataset_name $DATASET
  --split test
  --ckpt_path ckpts/exp2/_ckpt_epoch_10.ckpt
  (--save_visual)
  (--scan $SCAN)
```

**IMPORTANT**: Currently the script consumes huge RAM. If you set `img_wh` to the max `(2048, 1056)`, it will require up to 20GB RAM if the number of views is about 300.

It will generate depth prediction files under folder `results/$DATASET/depth`; after the depth prediction for all images finished, it will perform depth fusion for all scans and generate `.ply` files under folder `results/$DATASET/points`.

*  You can comment out the `# Step 1.` to do depth fusion only, after the depth prediction are generated.
*  You can add `--scan $SCAN` to only do depth fusion on specific scans (specify also the correct `--split`). Otherwise the default will process all scans in the `split`.
*  For `BlendedMVS`, the `depth_interval` is defined as the number of depths in the coarsest level, which should be equal to `48x4.0=192.0` in the default settings.

# Visualization of point cloud

From the base directory of this repo, run `python visualize_ply.py --dataset_name $DATASET --scan $SCAN`.

# Quantitative and qualitative evaluation

Please see each dataset subdirectory.

* [DTU](dtu/)
* [Tanks and Temples](tanks/)
* [BlendedMVS](blendedmvs/)
