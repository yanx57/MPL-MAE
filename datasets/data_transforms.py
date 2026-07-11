import numpy as np
import torch
import random

import torch

def rotate_point_cloud_by_angle(data, angle):
    """
    针对 Batch 点云进行 Y 轴固定角度旋转
    输入:
        data:  [B, N, 3] Tensor, 原始点云坐标
        angle: float, 旋转弧度 (例如: np.pi / 5)
    输出:
        rotated_data: [B, N, 3] Tensor, 旋转后的点云
    """
    device = data.device
    cos_val = torch.cos(torch.tensor(angle)).to(device)
    sin_val = torch.sin(torch.tensor(angle)).to(device)
    
    # 构造 Y 轴旋转矩阵 (围绕 Y 轴旋转)
    # R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    rot_mat = torch.tensor([
        [cos_val,  0, sin_val],
        [0,        1, 0      ],
        [-sin_val, 0, cos_val]
    ], dtype=data.dtype).to(device)
    
    # 执行旋转变换: [B, N, 3] @ [3, 3] -> [B, N, 3]
    rotated_data = torch.matmul(data, rot_mat)
    
    return rotated_data

class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudScaleAndTranslate_Uniform(object):
    def __init__(self,
                 scale_low=0.95,
                 scale_high=1.05,
                 translate_range=0.02):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        # pc: (B, N, 3)
        B = pc.size(0)
        device = pc.device

        for i in range(B):
            # uniform scale (same for x,y,z)
            scale = torch.empty(1, device=device).uniform_(
                self.scale_low, self.scale_high
            )

            # translate only on x,z
            shift = torch.zeros(3, device=device)
            shift[0] = torch.empty(1).uniform_(
                -self.translate_range, self.translate_range
            )
            shift[2] = torch.empty(1).uniform_(
                -self.translate_range, self.translate_range
            )

            pc[i, :, 0:3] = pc[i, :, 0:3] * scale + shift

        return pc

def normalize_point_cloud(pc):
    """
    pc: torch.Tensor, shape (B, N, 3)
    """
    # 1. 中心化 (Centering)
    # 计算每个 batch 样本的质心: (B, 1, 3)
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    
    # 2. 归一化 (Normalization) 到单位球 (Unit Ball)
    # 计算每个点到原点的距离，找到最远的那个点的距离作为缩放因子
    # dist shape: (B, N)
    # dist = torch.norm(pc, dim=-1, p=2)
 
    # max_dist = torch.max(dist, dim=1, keepdim=True)[0].unsqueeze(-1)
    
    # pc = pc / max_dist
    
    return pc

class PointcloudJitter_xz(object):
    def __init__(self, std=0.005, clip=0.02):
        self.std = std
        self.clip = clip

    def __call__(self, pc):
        # pc: (B, N, 3)
        B, N, _ = pc.shape

        jitter = pc.new(B, N, 3).normal_(
            mean=0.0, std=self.std
        ).clamp_(-self.clip, self.clip)

        # no jitter on y-axis
        jitter[:, :, 1] = 0.0

        pc[:, :, 0:3] += jitter
        return pc


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

class RandomHorizontalFlip(object):


  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])


  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords