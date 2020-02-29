from datasets.dtu import DTUDataset
from datasets.utils import save_pfm, read_pfm
import cv2
import torch
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# for depth prediction
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from inplace_abn import ABN

# for point cloud fusion
from numba import jit
from plyfile import PlyData, PlyElement

torch.backends.cudnn.benchmark = True # this increases inference speed a little

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/mvs_training/dtu/',
                        help='root directory of dtu dataset')
    parser.add_argument('--split', type=str,
                        default='val', choices=['train', 'val', 'test'],
                        help='which split to evaluate')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=5,
                        help='number of views (including ref) to be used in training')
    parser.add_argument('--depth_interval', type=float, default=2.65,
                        help='depth interval unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,48],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,2.0,4.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1152, 864],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_depth_visual', default=False, action='store_true',
                        help='save depth visualization or not')

    # for point cloud fusion
    parser.add_argument('--conf', type=float, default=0.99,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--min_geo_consistent', type=int, default=7,
                        help='min number of consistent views for pixel to be valid')
    parser.add_argument('--skip', type=int, default=1,
                        help='''how many points to skip when creating the point cloud.
                                Larger = fewer points and smaller file size.
                                Ref: skip=10 creates ~= 3M points = 50MB file
                                     skip=1 creates ~= 30M points = 500MB file
                             ''')

    return parser.parse_args()


def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    depths = batch['depths']
    masks = batch['masks']
    init_depth_min = batch['init_depth_min']
    depth_interval = batch['depth_interval']
    scan, vid = batch['scan_vid']
    return imgs, proj_mats, depths, masks, \
           init_depth_min, depth_interval, \
           scan, vid


@jit(nopython=True, fastmath=True)
def xy_ref2src(xy_ref, depth_ref, P_world2ref,
               depth_src, P_world2src, img_wh):
    # create ref grid and project to ref 3d coordinate using depth_ref
    xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_ref
    xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xy_ref[:1])))

    P = (P_world2src @ np.ascontiguousarray(np.linalg.inv(P_world2ref)))[:3]
    # project to src 3d coordinate using P_world2ref and P_world2src
    xyz_src_h = P @ xyz_ref_h.reshape(4,-1)
    xy_src = xyz_src_h[:2]/xyz_src_h[2:3]
    xy_src = xy_src.reshape(2, img_wh[1], img_wh[0])

    return xy_src


@jit(nopython=True, fastmath=True)
def xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref,
               depth_src2ref, P_world2src, img_wh):
    # project xy_src back to ref view using the sampled depth
    xyz_src = np.vstack((xy_src, np.ones_like(xy_src[:1]))) * depth_src2ref
    xyz_src_h = np.vstack((xyz_src, np.ones_like(xy_src[:1])))
    P = (P_world2ref @ np.ascontiguousarray(np.linalg.inv(P_world2src)))[:3]
    xyz_ref_h = P @ xyz_src_h.reshape(4,-1)
    depth_ref_reproj = xyz_ref_h[2].reshape(img_wh[1], img_wh[0])
    xy_ref_reproj = xyz_ref_h[:2]/xyz_ref_h[2:3]
    xy_ref_reproj = xy_ref_reproj.reshape(2, img_wh[1], img_wh[0])

    # check |p_reproj-p_1| < 1
    pixel_diff = xy_ref_reproj - xy_ref
    mask_pixel_reproj = (pixel_diff[0]**2+pixel_diff[1]**2)<1

    # check |d_reproj-d_1| / d_1 < 0.01
    mask_depth_reproj = np.abs((depth_ref_reproj-depth_ref)/depth_ref)<0.01

    mask_geo = mask_pixel_reproj & mask_depth_reproj

    return depth_ref_reproj, mask_geo


def check_geo_consistency(depth_ref, P_world2ref,
                          depth_src, P_world2src, img_wh):
    """
    Check the geometric consistency between ref and src views.
    """
    xy_ref = np.mgrid[:img_wh[1],:img_wh[0]][::-1].astype(np.float32)
    xy_src = xy_ref2src(xy_ref, depth_ref, P_world2ref,
                        depth_src, P_world2src, img_wh)

    # Sample the depth of xy_src using bilinear interpolation
    depth_src2ref = cv2.remap(depth_src,
                              xy_src[0].astype(np.float32),
                              xy_src[1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    depth_ref_reproj, mask_geo = \
        xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref, 
                   depth_src2ref, P_world2src, img_wh)
    depth_ref_reproj[~mask_geo] = 0
    
    return depth_ref_reproj, mask_geo


if __name__ == "__main__":
    args = get_opts()
    dataset = DTUDataset(args.root_dir, args.split,
                         n_views=args.n_views, depth_interval=args.depth_interval,
                         img_wh=tuple(args.img_wh))


    # Step 1. Create depth estimation and probability for each scan
    model = CascadeMVSNet(n_depths=args.n_depths,
                          interval_ratios=args.interval_ratios,
                          num_groups=args.num_groups,
                          norm_act=ABN).cuda()
    load_ckpt(model, args.ckpt_path)
    model.eval()

    depth_dir = 'results/depth'
    print('Creating depth predictions...')
    for i in tqdm(range(len(dataset))):
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval, \
            scan, vid = decode_batch(dataset[i])
        
        os.makedirs(os.path.join(depth_dir, scan), exist_ok=True)

        with torch.no_grad():
            results = model(imgs.unsqueeze(0).cuda(), proj_mats.unsqueeze(0).cuda(),
                            init_depth_min, depth_interval)
        
        depth = results['depth_0'][0].cpu().numpy()
        save_pfm(os.path.join(depth_dir, f'{scan}/depth_{vid:04d}.pfm'), depth)
        save_pfm(os.path.join(depth_dir, f'{scan}/proba_{vid:04d}.pfm'),
                 results['confidence_0'][0].cpu().numpy())
        if args.save_depth_visual:
            depth = (depth-depth.min())/(depth.max()-depth.min())
            depth = (255*depth).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/depth_visual_{vid:04d}.jpg'),
                        depth_img)


    # Step 2. Perform depth filtering and fusion
    point_dir = 'results/points'
    os.makedirs(point_dir, exist_ok=True)
    print('Fusing point clouds...')
    for scan in dataset.scans:
        print(f'Processing {scan}...')
        # buffers for the final vertices of this scan
        vs = []
        v_colors = []
        for ref_vid, meta in tqdm(enumerate(filter(lambda x: x[0]==scan, dataset.metas))):
            image_ref = cv2.imread(os.path.join(args.root_dir,
                                    f'Rectified/{scan}/rect_{ref_vid+1:03d}_3_r5000.png'))
            image_ref = cv2.resize(image_ref, tuple(args.img_wh),
                                   interpolation=cv2.INTER_LINEAR)[:,:,::-1] # to RGB
            depth_ref = read_pfm(f'results/depth/{scan}/depth_{ref_vid:04d}.pfm')[0]
            proba_ref = read_pfm(f'results/depth/{scan}/proba_{ref_vid:04d}.pfm')[0]
            mask_conf = proba_ref > args.conf # confidence mask
            P_world2ref = dataset.proj_mats[ref_vid][0][0].numpy()
            
            src_vids = meta[3]
            mask_geos = []
            depth_ref_reprojs = []
            # for each src view, check the consistency and refine depth
            for src_vid in src_vids:
                depth_src = read_pfm(f'results/depth/{scan}/depth_{src_vid:04d}.pfm')[0]
                P_world2src = dataset.proj_mats[src_vid][0][0].numpy()
                depth_ref_reproj, mask_geo = check_geo_consistency(depth_ref, P_world2ref,
                                                                   depth_src, P_world2src,
                                                                   tuple(args.img_wh))
                depth_ref_reprojs += [depth_ref_reproj]
                mask_geos += [mask_geo]
            mask_geo_sum = np.sum(mask_geos, 0)
            mask_geo_final = mask_geo_sum >= args.min_geo_consistent
            depth_ref_average = (np.sum(depth_ref_reprojs, 0)+depth_ref)/(mask_geo_sum+1)
            mask_final = mask_conf & mask_geo_final
            depth_ref_average[~mask_final] = 0
            
            # create the final points
            xy_ref = np.mgrid[:args.img_wh[1],:args.img_wh[0]][::-1]
            xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_ref_average
            xyz_ref = xyz_ref.transpose(1,2,0)[mask_final].T # (3, N)
            color = image_ref[mask_final] # (N, 3)
            xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xyz_ref[:1])))
            xyz_world = (np.linalg.inv(P_world2ref) @ xyz_ref_h).T # (N, 4)
            xyz_world = xyz_world[::args.skip, :3]
            color = color[::args.skip]
            
            # append to buffers
            vs += [xyz_world]
            v_colors += [color]
            
        # process all points in the buffers
        vs = np.concatenate(vs, axis=0)
        v_colors = np.concatenate(v_colors, axis=0)
        print(scan, 'contains', len(vs), 'points')
        vs = np.array([tuple(v) for v in vs],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        v_colors = np.array([tuple(v) for v in v_colors],
                            dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_all = np.empty(len(vs), vs.dtype.descr+v_colors.dtype.descr)
        for prop in vs.dtype.names:
            vertex_all[prop] = vs[prop]
        for prop in v_colors.dtype.names:
            vertex_all[prop] = v_colors[prop]

        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(f'{point_dir}/{scan}.ply')


    print('Done!')