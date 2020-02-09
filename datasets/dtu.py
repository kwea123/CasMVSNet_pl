from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T

class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, init_depth_interval=2.8):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val'], \
            'split must be either "train" or "val"!'
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.init_depth_interval = init_depth_interval
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        with open(f'datasets/lists/dtu/{self.split}.txt') as f:
            scans = [line.rstrip() for line in f.readlines()]

        pair_file = "Cameras/pair.txt"
        for scan in scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        self.metas += [(scan, light_idx, ref_view, src_views)]

    def build_proj_mats(self):
        proj_mats = []
        for vid in range(49): # total 49 view ids
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            proj_mat_ls = torch.stack(proj_mat_ls[::-1]) # (self.levels, 4, 4) from fine to coarse
            proj_mats += [(proj_mat_ls, depth_min)]

        self.proj_mats = proj_mats

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        return intrinsics, extrinsics, depth_min

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_NEAREST) # (800, 600)
        depth_0 = depth[44:556, 80:720] # (640, 512)
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST) # (320, 256)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST) # (160, 128)
        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}
        
        return depths

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                          interpolation=cv2.INTER_NEAREST) # (800, 600)
        mask_0 = mask[44:556, 80:720] # (640, 512)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (320, 256)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (160, 128)

        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2)}

        return masks

    def define_transforms(self):
        if self.split == 'train':
            self.transform = T.Compose([T.ToTensor(),
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
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
            mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_visual_{vid:04d}.png')
            depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_map_{vid:04d}.pfm')

            img = Image.open(img_filename)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[vid]

            if i == 0:  # reference view
                masks = self.read_mask(mask_filename)
                depths = self.read_depth(depth_filename)
                proj_mats += [torch.inverse(proj_mat_ls)]
            else:
                proj_mats += [proj_mat_ls]

        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats) # (V, self.levels, 4, 4) from fine to coarse

        return imgs, proj_mats, depths, masks, depth_min, self.init_depth_interval