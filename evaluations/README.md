Here I provide the description of the general depth fusion method. The same method is applied for all datasets, if not specified otherwise, the testing uses the default parameters in `eval.py`.

# Depth fusion description

The fusion code is heavily borrowed from [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py) with refactoring and the following modifications (without careful investigations, I just think it looks better):
1.  After the depth of the ref view is refined (this is original mvsnet method), I **use the refined depth** for the following runs (1 run=1 ref view and many src views). For example, depth of view 0 is refined in the first run, then the next run, for ref view 1, if it uses view 0 as src view, this time we don't use the original depth prediction, instead we use the refined depth of the previous run since it is generally better (average across many views).
2.  When projecting points into 3d space, I observe that the color of the same point change a lot across views: this is partly due to non-lambertian sufaces such as metal reflection, but is hugely due to light angle difference bewteen views. Since it's an **indoor** dataset, the light angle changes a lot even with a small displacement, making shadows projected to different spaces. In order to alleviate this color inconsistency, in addition to depth average as suggested in the original paper, I also do **color average** to make the point cloud look more consistent.

# Running depth fusion

Run
```
python eval.py \
  --dataset_name $DATASET
  --split test
  --ckpt_path ckpts/exp2/_ckpt_epoch_10.ckpt
  (--save_visual)
```

It will generate depth prediction files under folder `results/$DATASET/depth`; after the depth prediction for all images finished, it will perform depth fusion for all scans and generate `.ply` files under folder `results/$DATASET/points`.

*  You can comment out the `# Step 1.` to do depth fusion only, after the depth prediction are generated.
*  You can add `--scan {scan_number}` to only do depth fusion on specific scans (specify also the correct `--split`). Otherwise the default will process all scans in the `split`.

# Visualization of point cloud

Run `python visualize_ply.py --dataset_name $DATASET --scan $SCAN`.

# Quantitative evaluation

Please see each dataset subdirectory.
