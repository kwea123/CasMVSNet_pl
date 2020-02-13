import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential( 
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, 
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x) # (B, 8, H, W)
        conv1 = self.conv1(conv0) # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1) # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2) # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1) # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0) # (B, 8, H, W)

        feats = {"level_0": feat0,
                 "level_1": feat1,
                 "level_2": feat2}

        return feats

class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x

class CascadeMVSNet(nn.Module):
    def __init__(self, n_depths=[8, 32, 48],
                       interval_ratios=[1, 2, 4],
                       norm_act=InPlaceABN):
        super(CascadeMVSNet, self).__init__()
        self.levels = 3 # 3 depth levels
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.feature = FeatureNet(norm_act)
        for l in range(self.levels):
            setattr(self, f'cost_reg_{l}', CostRegNet(8*2**l, norm_act))

    def predict_depth(self, feats, proj_mats, depth_values, cost_reg):
        # feats: (B, V, F, H, W)
        # proj_mats: (B, V, 4, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, F, D, h, w) and output (B, 1, D, h, w)
        B, V, _, H, W = feats.shape
        D = depth_values.shape[1]

        # step 1. feature extraction
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        ref_proj, src_projs = proj_mats[:, 0], proj_mats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4) # (V-1, B, F, h, w)
        src_projs = src_projs.permute(1, 0, 2, 3) # (V-1, B, 4, 4)

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1) # (B, F, D, h, w)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_volume = homo_warp(src_feat, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2) # the memory of warped_volume is modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
        del volume_sq_sum, volume_sum
        
        # step 3. cost volume regularization
        cost_reg = cost_reg(volume_variance).squeeze(1)
        prob_volume = F.softmax(cost_reg, 1) # (B, D, h, w)
        depth = depth_regression(prob_volume, depth_values)
        
        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                      pad=(0, 0, 0, 0, 1, 2)),
                                                (4, 1, 1), stride=1).squeeze(1) # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(prob_volume,
                                           torch.arange(D,
                                                        device=prob_volume.device,
                                                        dtype=prob_volume.dtype)
                                          ).long() # (B, h, w)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(prob_volume_sum4, 1, 
                                      depth_index.unsqueeze(1)).squeeze(1) # (B, h, w)

        return depth, confidence

    def forward(self, imgs, proj_mats, init_depth_min, depth_interval):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, self.levels, 4, 4) from fine to coarse
        # init_depth_min, depth_interval: (B,) assumed to be the same for all samples
        B, V, _, H, W = imgs.shape
        results = {}

        imgs = imgs.reshape(B*V, 3, H, W)
        feats = self.feature(imgs) # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        # TODO: any better way?
        if not isinstance(init_depth_min, float):
            init_depth_min = float(init_depth_min[0].cpu().numpy())
        if not isinstance(depth_interval, float):
            depth_interval = float(depth_interval[0].cpu().numpy())

        for l in reversed(range(self.levels)): # (2, 1, 0)
            feats_l = feats[f"level_{l}"] # (B*V, F, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:]) # (B, V, F, h, w)
            proj_mats_l = proj_mats[:, :, l] # (B, V, 4, 4)
            depth_interval_l = depth_interval * self.interval_ratios[l]
            if l == self.levels-1: # coarsest level
                h, w = feats_l.shape[-2:]

                depth_values = init_depth_min + depth_interval_l * \
                               torch.arange(0, self.n_depths[l],
                                            device=imgs.device,
                                            dtype=imgs.dtype)
                depth_values = depth_values.reshape(1, -1, 1, 1).repeat(B, 1, h, w)
            else:
                depth_l_1 = depth_l.detach() # the depth of previous level
                depth_l_1 = F.interpolate(depth_l_1.unsqueeze(1),
                                          scale_factor=2, mode='bilinear',
                                          align_corners=True) # (B, 1, h, w)
                depth_values = get_depth_values(depth_l_1,
                                                self.n_depths[l],
                                                depth_interval_l)
                del depth_l_1
            depth_l, confidence_l = self.predict_depth(feats_l, proj_mats_l, depth_values,
                                                       getattr(self, f'cost_reg_{l}'))
            del feats_l, proj_mats_l, depth_values
            results[f"depth_{l}"] = depth_l
            results[f"confidence_{l}"] = confidence_l

        return results