from einops import reduce, rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from kornia.utils import create_meshgrid

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def get_depth_values(current_depth, n_depths, depth_interval):
    """
    get the depth values of each pixel : [depth_min, depth_max) step is depth_interval
    current_depth: (B, 1, H, W), current depth map
    n_depth: int, number of channels of depth
    depth_interval: (B, 1) or float, interval between each depth channel
    return: (B, D, H, W)
    """
    if not isinstance(depth_interval, float):
        depth_interval = rearrange(depth_interval, 'b 1 -> b 1 1 1')
    depth_min = torch.clamp_min(current_depth - n_depths/2 * depth_interval, 1e-7)
    depth_values = depth_min + depth_interval * \
                   rearrange(torch.arange(0, n_depths,
                                          device=current_depth.device,
                                          dtype=current_depth.dtype), 'd -> 1 d 1 1')
    return depth_values


def homo_warp(src_feat, proj_mat, depth_values):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device

    R = proj_mat[:, :, :3] # (B, 3, 3)
    T = proj_mat[:, :, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
    ref_grid = rearrange(ref_grid, '1 h w c -> 1 c (h w)') # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = repeat(ref_grid, 'b c x -> b c (d x)', d=D) # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T/rearrange(depth_values, 'b d h w -> b 1 (d h w)')
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory
    
    # project negative depth pixels to somewhere outside the image
    negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
    src_grid_d[:, 0:1][negative_depth_mask] = W
    src_grid_d[:, 1:2][negative_depth_mask] = H
    src_grid_d[:, 2:3][negative_depth_mask] = 1

    src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:] # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0]/((W-1)/2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((H-1)/2) - 1 # scale to -1~1
    src_grid = rearrange(src_grid, 'b c (d h w) -> b d (h w) c', d=D, h=H, w=W)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = rearrange(warped_src_feat, 'b c d (h w) -> b c d h w', h=H, w=W)

    return warped_src_feat


def depth_regression(p, depth_values):
    """
    p: probability volume (B, D, H, W)
    depth_values: discrete depth values (B, D, H, W) or (D)
    inverse: depth_values is inverse depth or not
    """
    if depth_values.dim() == 1:
        depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
    depth = reduce(p*depth_values, 'b d h w -> b h w', 'sum').to(depth_values.dtype)
    return depth