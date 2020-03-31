from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
from collections import defaultdict
import cv2
from PIL import Image
import torch
from torchvision import transforms as T

class BlendedMVSDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=192.0,
                 img_wh=(768, 576)):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'all'], \
            'split must be either "train", "val" or "all"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.n_depths = depth_interval
        self.build_metas()
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        self.ref_views_per_scan = defaultdict(list)
        if self.split == 'train':
            list_txt = os.path.join(self.root_dir, '../training_list.txt')
        elif self.split == 'val':
            list_txt = os.path.join(self.root_dir, '../validation_list.txt')
        else:
            list_txt = os.path.join(self.root_dir, '../all_list.txt')

        with open(list_txt) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    self.ref_views_per_scan[scan] += [ref_view]
                    line = f.readline().rstrip().split()
                    n_views_valid = int(line[0]) # valid views
                    if n_views_valid < self.n_views: # skip no enough valid views
                        continue
                    src_views = [int(x) for x in line[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]

    def build_proj_mats(self):
        self.proj_mats = {} # proj mats for each scan
        self.scale_factors = {} # depth scale factors for each scan
        if self.root_dir.endswith('dataset_low_res') \
            or self.root_dir.endswith('dataset_low_res/'):
            img_w, img_h = 768, 576
        else:
            img_w, img_h = 2048, 1536
        for scan in self.scans:
            self.proj_mats[scan] = {}
            for vid in self.ref_views_per_scan[scan]:
                proj_mat_filename = os.path.join(self.root_dir, scan,
                                                 f'cams/{vid:08d}_cam.txt')
                intrinsics, extrinsics, depth_min = \
                    self.read_cam_file(scan, proj_mat_filename)
                intrinsics[0] *= self.img_wh[0]/img_w/4
                intrinsics[1] *= self.img_wh[1]/img_h/4

                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat_ls = []
                for l in reversed(range(self.levels)):
                    proj_mat_l = np.eye(4)
                    proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                    intrinsics[:2] *= 2 # 1/4->1/2->1
                    proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
                # (self.levels, 4, 4) from fine to coarse
                proj_mat_ls = torch.stack(proj_mat_ls[::-1])
                self.proj_mats[scan][vid] = (proj_mat_ls, depth_min)

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        if scan not in self.scale_factors:
            # use the first cam to determine scale factor
            self.scale_factors[scan] = 100/depth_min

        depth_min *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan]
        return intrinsics, extrinsics, depth_min

    def read_depth_and_mask(self, scan, filename, depth_min):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth *= self.scale_factors[scan]
        if self.img_wh is not None:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)

        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}

        masks = {"level_0": torch.BoolTensor(depth_0>depth_min),
                 "level_1": torch.BoolTensor(depth_1>depth_min),
                 "level_2": torch.BoolTensor(depth_2>depth_min)}

        depth_max = depth_0.max()
        
        return depths, masks, depth_max

    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ColorJitter(brightness=0.25,
                                                      contrast=0.5),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                       ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir,
                                f'{scan}/blended_images/{vid:08d}.jpg')
            depth_filename = os.path.join(self.root_dir,
                                f'{scan}/rendered_depth_maps/{vid:08d}.pfm')

            img = Image.open(img_filename)
            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[scan][vid]

            if i == 0:  # reference view
                depths, masks, depth_max = self.read_depth_and_mask(scan, depth_filename, depth_min)
                depth_interval = (depth_max-depth_min)/self.n_depths
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                sample['depth_interval'] = torch.FloatTensor([depth_interval])
                
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats
        sample['depths'] = depths
        sample['masks'] = masks
        sample['scan_vid'] = (scan, ref_view)

        return sample