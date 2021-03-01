from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms as T

class TanksDataset(Dataset):
    def __init__(self, root_dir, split='intermediate', n_views=3, levels=3, depth_interval=-1,
                 img_wh=(1152, 864)):
        """
        For testing only! You can write training data loader by yourself.
        @depth_interval has no effect. The depth_interval is predefined for each view.
        """
        self.root_dir = root_dir
        self.img_wh = img_wh
        assert img_wh[0]%32==0 and img_wh[1]%32==0, \
            'img_wh must both be multiples of 32!'
        self.split = split
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                          'M60', 'Panther', 'Playground', 'Train']
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080)}
            self.depth_interval = {'Family': 2.5e-3,
                                   'Francis': 1e-2,
                                   'Horse': 1.5e-3,
                                   'Lighthouse': 1.5e-2,
                                   'M60': 5e-3,
                                   'Panther': 5e-3,
                                   'Playground': 7e-3,
                                   'Train': 5e-3} # depth interval for each scan (hand tuned)
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}
            self.depth_interval = {'Auditorium': 3e-2,
                                   'Ballroom': 2e-2,
                                   'Courtroom': 2e-2,
                                   'Museum': 2e-2,
                                   'Palace': 1e-2,
                                   'Temple': 1e-2} # depth interval for each scan (hand tuned)
        self.ref_views_per_scan = defaultdict(list)

        for scan in self.scans:
            with open(os.path.join(self.root_dir, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]
                    self.ref_views_per_scan[scan] += [ref_view]

    def build_proj_mats(self):
        self.proj_mats = {} # proj mats for each scan
        for scan in self.scans:
            self.proj_mats[scan] = {}
            img_w, img_h = self.image_sizes[scan]
            for vid in self.ref_views_per_scan[scan]:
                proj_mat_filename = os.path.join(self.root_dir, self.split, scan,
                                                 f'cams/{vid:08d}_cam.txt')
                intrinsics, extrinsics, depth_min = \
                    self.read_cam_file(proj_mat_filename)
                intrinsics[0] *= self.img_wh[0]/img_w/4
                intrinsics[1] *= self.img_wh[1]/img_h/4
                # self.depth_interval[scan][vid] = depth_interval

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

    def define_transforms(self):
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
            img_filename = os.path.join(self.root_dir, self.split, scan, f'images/{vid:08d}.jpg')

            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[scan][vid]

            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_ls)
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                sample['depth_interval'] = torch.FloatTensor([self.depth_interval[scan]])
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats
        sample['scan_vid'] = (scan, ref_view)

        return sample
