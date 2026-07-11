import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from timm.models.layers import DropPath, trunc_normal_

# 自动添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试 import logger，失败则用内联版本
try:
    from logger import get_missing_parameters_message, get_unexpected_parameters_message
except ImportError:
    def get_missing_parameters_message(keys):
        msg = "Some model parameters or buffers are not found in the checkpoint:\n"
        msg += "\n".join("  " + k for k in sorted(keys))
        return msg

    def get_unexpected_parameters_message(keys):
        msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
        msg += "\n".join("  " + k for k in sorted(keys))
        return msg

from pointnet2_ops import pointnet2_utils
from knn import knn_point
from pointnet2_utils import PointNetFeaturePropagation

from models.pos import get_pos_embed


# ============================================================
# DCP 协议对齐: 用 scipy 实现欧拉角转换 (与 DCP 论文一致)
# ============================================================

def npmat2euler_via_scipy(R, seq='zyx'):
    """
    通过 scipy 实现旋转矩阵 -> 欧拉角转换。
    与 DCP 论文使用的 scipy.spatial.transform.Rotation.as_euler('zyx', degrees=True) 完全一致。
    
    Args:
        R: (B, 3, 3) torch tensor
    Returns:
        (B, 3) torch tensor in degrees, [z_angle, y_angle, x_angle]
    """
    R_np = R.detach().cpu().numpy()
    eulers = []
    for i in range(R_np.shape[0]):
        e = Rotation.from_matrix(R_np[i]).as_euler(seq, degrees=True)
        eulers.append(e)
    eulers = np.stack(eulers, axis=0).astype(np.float32)
    return torch.from_numpy(eulers).to(R.device).to(R.dtype)


# ============================================================
# 工具函数
# ============================================================

def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


# ============================================================
# 模型组件
# ============================================================

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            output: B G M 3, center: B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


# ============================================================
# SVD Head (与 DCP 一致)
# ============================================================

class SVDHead(nn.Module):
    def __init__(self, emb_dims):
        super(SVDHead, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.linear_pro = nn.Linear(self.emb_dims, self.emb_dims)

    def forward(self, src_embedding, tgt_embedding, src, tgt):
        """
        src_embedding: (B, C, N)
        tgt_embedding: (B, C, M)
        src: (B, 3, N)
        tgt: (B, 3, M)
        """
        B, C, N = src_embedding.shape
        src_embedding = self.linear_pro(src_embedding.transpose(2, 1)).transpose(2, 1)
        tgt_embedding = self.linear_pro(tgt_embedding.transpose(2, 1)).transpose(2, 1)

        # 1. soft correspondence
        scores = torch.matmul(src_embedding.transpose(2, 1), tgt_embedding) / math.sqrt(C)
        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1))

        # 2. center the point clouds
        src_mean = src.mean(dim=2, keepdim=True)
        src_corr_mean = src_corr.mean(dim=2, keepdim=True)
        src_centered = src - src_mean
        src_corr_centered = src_corr - src_corr_mean

        # 3. covariance
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1))

        # 4. SVD
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        V = Vh.transpose(1, 2)
        R = torch.matmul(V, U.transpose(1, 2))

        # 5. reflection fix
        det = torch.det(R)
        mask = (det < 0).view(-1)
        if mask.any():
            V_fixed = V.clone()
            V_fixed[mask, :, 2] *= -1
            R[mask] = torch.matmul(V_fixed[mask], U[mask].transpose(1, 2))

        # 6. translation
        t = src_corr_mean - torch.matmul(R, src_mean)
        t = t.squeeze(-1)

        return R, t


# ============================================================
# 主模型 (DCP-v1 风格: 单方向预测)
# ============================================================

class get_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = 32
        self.num_group = 128

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.svd_head = SVDHead(self.trans_dim)

    # ============================================================
    # Loss (DCP-v1 标准: identity-based MSE)
    # ============================================================
    
    def pose_loss(self, R_pred, t_pred, R_gt, t_gt):
        """
        DCP 标准 loss:
            loss_R = ||R_pred^T @ R_gt - I||² (identity-based)
            loss_t = ||t_pred - t_gt||²
        """
        B = R_pred.shape[0]
        identity = torch.eye(3, device=R_pred.device).unsqueeze(0).repeat(B, 1, 1)
        
        loss_R = F.mse_loss(torch.matmul(R_pred.transpose(2, 1), R_gt), identity)
        loss_t = F.mse_loss(t_pred, t_gt)
        
        return loss_R + loss_t

    # ============================================================
    # DCP 协议对齐的评估指标
    # ============================================================
    
    def get_euler_angles(self, R):
        """旋转矩阵 -> ZYX 欧拉角 (degrees), 与 scipy 完全一致"""
        return npmat2euler_via_scipy(R, seq='zyx')

    def rotation_error_per_sample(self, R_pred, R_gt):
        """DCP 协议: 返回欧拉角逐维度差异 (B, 3)"""
        euler_pred = self.get_euler_angles(R_pred)
        euler_gt = self.get_euler_angles(R_gt)
        return euler_pred - euler_gt

    def translation_error_per_sample(self, t_pred, t_gt):
        """DCP 协议: 返回平移逐维度差异 (B, 3)"""
        return t_pred - t_gt

    def rotation_geodesic_per_sample(self, R_pred, R_gt):
        """辅助指标: SO(3) geodesic angle (B,)"""
        R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        theta_deg = torch.rad2deg(torch.acos(cos_theta))
        return theta_deg

    # ============================================================
    # Checkpoint loading
    # ============================================================
    
    def load_model_from_ckpt(self, bert_ckpt_path, model_prefix):
        assert model_prefix in ['MAE_encoder']
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith(model_prefix):
                    base_ckpt[k[len(model_prefix + '.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    # ============================================================
    # Forward (DCP-v1 风格: 一次 forward, 单方向)
    # ============================================================
    
    def forward(self, src_pts, tgt_pts):
        """
        src_pts: (B, N, 3) source
        tgt_pts: (B, N, 3) target
        returns: R_ab, t_ab (source -> target transform)
        """
        # Source 分支
        src_neighborhood, src_center = self.group_divider(src_pts)
        src_tokens = self.encoder(src_neighborhood)
        src_pos = self.pos_embed(get_pos_embed(self.trans_dim, src_center))
        src_feat = self.blocks(src_tokens, src_pos)

        # Target 分支 (注意: 用 tgt_tokens, 不是 src_tokens)
        tgt_neighborhood, tgt_center = self.group_divider(tgt_pts)
        tgt_tokens = self.encoder(tgt_neighborhood)
        tgt_pos = self.pos_embed(get_pos_embed(self.trans_dim, tgt_center))
        tgt_feat = self.blocks(tgt_tokens, tgt_pos)

        # SVD 求解
        R, t = self.svd_head(
            src_feat.transpose(2, 1),
            tgt_feat.transpose(2, 1),
            src_center.transpose(2, 1),
            tgt_center.transpose(2, 1)
        )

        return R, t