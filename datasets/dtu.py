from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, n_depths=256, interval_scale=1.06):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val'], \
            'split must be either "train" or "val"!'
        self.build_metas()
        self.n_views = n_views
        self.n_depths = n_depths
        self.interval_scale = interval_scale
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
            intrinsics, extrinsics, depth_min, depth_interval = \
                self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics
            proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
            proj_mats += [(torch.FloatTensor(proj_mat), depth_min, depth_interval)]

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
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_depth(self, filename):
        return np.array(read_pfm(filename)[0], dtype=np.float32)

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
                                f'Depths/{scan}_train/depth_visual_{vid:04d}.png')
            depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}_train/depth_map_{vid:04d}.pfm')

            img = Image.open(img_filename)
            img = self.transform(img)
            imgs += [img]

            proj_mat, depth_min, depth_interval = self.proj_mats[vid]

            if i == 0:  # reference view
                depth_values = torch.arange(depth_min,
                                            depth_interval*self.n_depths+depth_min,
                                            depth_interval,
                                            dtype=torch.float32)
                mask = Image.open(mask_filename)
                mask = torch.BoolTensor(np.array(mask))
                depth = torch.FloatTensor(self.read_depth(depth_filename))
                proj_mats += [torch.inverse(proj_mat)]
            else:
                proj_mats += [proj_mat]

        imgs = torch.stack(imgs)
        proj_mats = torch.stack(proj_mats)

        return imgs, proj_mats, depth, depth_values, mask