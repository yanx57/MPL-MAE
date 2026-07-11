"""
Point cloud completion model strictly aligned with PoinTr architecture.

Key alignment with PoinTr/models/PoinTr.py:
- Same Fold class (hidden_dim=256, 2-stage folding)
- Same coarse prediction strategy (global feature -> Linear(1024, 3*M))
- Same query feature construction (global + coarse_xyz -> mlp_query)
- Same decoder: 8 layers with self-attn + cross-attn (depth=[6,8])
- Same loss: ChamferDistanceL1 (sparse + dense)
- Same output: coarse=2M points, dense=14336+2048=16384 points

Difference from PoinTr:
- Encoder is REPLACED:
    PoinTr's DGCNN_Grouper + Transformer encoder (6 layers)
    -> Your pretrained SSL encoder (12-layer Transformer with mini-PointNet token encoder)

Reason: This evaluates the quality of pretrained representations on completion task,
keeping decoder/head identical to PoinTr for fair comparison.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === 添加 extensions 所在的父目录到 sys.path ===
EXTENSIONS_PARENT = '/home/yanxu_2023/yanxu_2023/eccv_rebuttal/MPL-MAE-recon_task'
if EXTENSIONS_PARENT not in sys.path:
    sys.path.insert(0, EXTENSIONS_PARENT)

try:
    from logger import get_missing_parameters_message, get_unexpected_parameters_message
except ImportError:
    def get_missing_parameters_message(keys):
        return "Missing keys:\n" + "\n".join("  " + k for k in sorted(keys))
    def get_unexpected_parameters_message(keys):
        return "Unexpected keys:\n" + "\n".join("  " + k for k in sorted(keys))

from pointnet2_ops import pointnet2_utils
from knn import knn_point as knn_point_external

# ChamferDistance imported from PoinTr extensions (you said you can directly import)
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from models.pos import get_pos_embed


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


# ============================================================
# Pretrained encoder components (与你 SSL 预训练保持一致)
# ============================================================

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_point_external(self.group_size, xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class TokenEncoder(nn.Module):
    """Mini-PointNet token encoder (你 SSL 预训练用的版本)"""
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
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
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
        x = self.proj(x); x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """与你 SSL 预训练完全一致的 Block (encoder block)"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """你的 SSL 预训练 encoder (12 layer Transformer)"""
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


# ============================================================
# Decoder components - 严格对齐 PoinTr/models/Transformer.py
# ============================================================

def square_distance(src, dst):
    """与 PoinTr Transformer.py 一致"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point_internal(nsample, xyz, new_xyz):
    """与 PoinTr Transformer.py 中 knn_point 一致"""
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def get_knn_index(coor_q, coor_k=None):
    """与 PoinTr Transformer.py 一致"""
    coor_k = coor_k if coor_k is not None else coor_q
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        idx = knn_point_internal(8,
                                  coor_k.transpose(-1, -2).contiguous(),
                                  coor_q.transpose(-1, -2).contiguous())  # B G M
        idx = idx.transpose(-1, -2).contiguous()
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)

    return idx


def get_graph_feature(x, knn_index, x_q=None):
    """与 PoinTr Transformer.py 一致"""
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature  # b k np c


class CrossAttention(nn.Module):
    """严格对齐 PoinTr Transformer.py 的 CrossAttention"""
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    """严格对齐 PoinTr Transformer.py 的 DecoderBlock
    
    包含:
    - Self-attention + 可选 KNN graph feature merge
    - Cross-attention + 可选 KNN graph feature merge  
    - MLP
    - 所有都带 LayerNorm + DropPath + residual
    """
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # KNN feature merge layers (geometry-aware)
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(dim * 2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map_cross = nn.Linear(dim * 2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # Self-attention with optional KNN merge
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)

        q = q + self.drop_path(q_1)

        # Cross-attention with optional KNN merge
        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # MLP
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


# ============================================================
# Fold - 严格对齐 PoinTr/models/PoinTr.py 的 Fold class
# ============================================================

class Fold(nn.Module):
    """严格复制 PoinTr.py 中的 Fold class
    
    Hidden dim = 256 (不是 512!)
    Structure: Conv1d(in+2, 256) -> BN -> ReLU -> Conv1d(256, 128) -> BN -> ReLU -> Conv1d(128, 3)
    然后第二个 fold 同样结构 (但 input 是 in+3 而不是 in+2)
    """
    def __init__(self, in_channel, step, hidden_dim=256):
        super().__init__()
        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0)  # (2, step^2)

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        """
        x: (BM, in_channel)
        returns: (BM, 3, step^2)
        """
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


# ============================================================
# Main model: PoinTr architecture with pretrained encoder
# ============================================================

class get_model(nn.Module):
    """
    Strictly aligned with PoinTr architecture, but with pretrained encoder.
    
    Architecture:
    1. [REPLACED] Pretrained encoder: your SSL 12-layer Transformer
       - Replaces PoinTr's DGCNN_Grouper + 6-layer encoder
       - Output: (B, 128, 384) features + (B, 128, 3) centers
    
    2. [SAME AS PoinTr] increase_dim + global_feature
       - Conv1d(384, 1024, 1) -> BN -> LeakyReLU -> Conv1d(1024, 1024, 1)
       - max pool -> (B, 1024)
    
    3. [SAME AS PoinTr] coarse_pred:
       - Linear(1024, 1024) -> ReLU -> Linear(1024, 3*M)
       - M=224 coarse points predicted from global feature
    
    4. [SAME AS PoinTr] mlp_query:
       - Construct query feature: cat(global_repeat, coarse_xyz)
       - Conv1d(1027, 1024) -> LeakyReLU -> Conv1d(1024, 1024) -> LeakyReLU -> Conv1d(1024, 384)
    
    5. [SAME AS PoinTr] Decoder: 8 layers DecoderBlock
       - Each block: self-attn + cross-attn + mlp
       - Layer 0 uses KNN graph feature (knn_layer=1)
    
    6. [SAME AS PoinTr] Reduce_map + Fold:
       - rebuild_feat = cat(global_repeat, q, coarse_xyz) -> reduce_map (Linear)
       - relative_xyz = Fold(rebuild_feat) (B, M, 3, S)
       - rebuild_points = relative_xyz + coarse_xyz
    
    7. [SAME AS PoinTr] Final concat with input partial:
       - coarse = cat(coarse_pred, fps(partial, M))  -> (B, 2M, 3)
       - dense  = cat(rebuild_points, partial)        -> (B, 14336+2048=16384, 3)
    
    Loss: ChamferDistanceL1 on both coarse and dense (1:1 weight, exactly as PoinTr)
    """
    def __init__(self, num_pred=14336, num_query=224, knn_layer=1, trans_dim=384,
                 encoder_depth=12, encoder_num_heads=6, decoder_depth=8, decoder_num_heads=6):
        super().__init__()

        # ============ Hyperparameters (从 PoinTr.yaml 来) ============
        self.trans_dim = trans_dim
        self.knn_layer = knn_layer
        self.num_pred = num_pred
        self.num_query = num_query
        self.fold_step = int(pow(num_pred // num_query, 0.5) + 0.5)  # 8 (224*64=14336)

        # ============ Pretrained encoder (REPLACES PoinTr's DGCNN+encoder) ============
        # 你 SSL 预训练 encoder 的配置
        self.group_size = 32
        self.num_group = 128

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = TokenEncoder(encoder_channel=self.trans_dim)

        # Pos embed (与你 SSL 预训练保持一致)
        self.pos_embed = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.1, encoder_depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            drop_path_rate=dpr,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # ============ PoinTr decoder components (严格对齐) ============
        # increase_dim: 与 PoinTr Transformer.py L314 一致
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        # coarse_pred: 与 PoinTr Transformer.py L322 一致
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )

        # mlp_query: 与 PoinTr Transformer.py L327 一致
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, self.trans_dim, 1)
        )

        # Decoder: 8 layers DecoderBlock
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=self.trans_dim, num_heads=decoder_num_heads, mlp_ratio=2.,
                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            for _ in range(decoder_depth)
        ])

        # ============ Foldingnet head (严格对齐 PoinTr.py L82-83) ============
        self.foldingnet = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)

        # increase_dim_2 + reduce_map (严格对齐 PoinTr.py L85-92)
        self.increase_dim_2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

        # ============ Loss function (严格对齐 PoinTr.py L94) ============
        self.build_loss_func()

        # Init weights for new modules
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """初始化非预训练部分 (decoder, fold, etc.)"""
        for name, m in self.named_modules():
            # 跳过预训练部分的初始化 (它们会被 ckpt 覆盖)
            is_pretrained = any(name.startswith(p) for p in
                                ['group_divider', 'encoder', 'pos_embed', 'blocks', 'norm'])
            if is_pretrained:
                continue

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def build_loss_func(self):
        """与 PoinTr.py L94 一致"""
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch=0):
        """
        与 PoinTr.py L96-99 一致
        ret: (coarse_point_cloud, rebuild_points)
        gt: (B, 16384, 3)
        """
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def load_model_from_ckpt(self, bert_ckpt_path, model_prefix='MAE_encoder'):
        """加载 SSL 预训练 encoder, 严格只取 model_prefix 部分"""
        if bert_ckpt_path is None:
            return
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

        # ============ 修改: 只保留 model_prefix.* 的 keys ============
        encoder_ckpt = {}
        for k, v in base_ckpt.items():
            if k.startswith(model_prefix + '.'):
                new_k = k[len(model_prefix + '.'):]
                encoder_ckpt[new_k] = v
            elif k.startswith('base_model.'):
                new_k = k[len('base_model.'):]
                # 进一步检查 base_model.MAE_encoder.* 才算
                if new_k.startswith(model_prefix + '.'):
                    new_k = new_k[len(model_prefix + '.'):]
                    encoder_ckpt[new_k] = v
                # 否则丢弃 (decoder 等部分)

        print(f'[Pretrain] Filtered to {len(encoder_ckpt)} encoder keys '
            f'(from {len(base_ckpt)} total)')

        incompatible = self.load_state_dict(encoder_ckpt, strict=False)

        # New decoder/head modules will appear in missing_keys (expected)
        if incompatible.missing_keys:
            print('[Pretrain] Missing keys (expected for new decoder/head modules):')
            # 只打印前几个避免日志过长
            for k in incompatible.missing_keys[:10]:
                print(f'  {k}')
            if len(incompatible.missing_keys) > 10:
                print(f'  ... and {len(incompatible.missing_keys) - 10} more')
        if incompatible.unexpected_keys:
            print('[Pretrain] Unexpected keys (these are encoder keys not in current model):')
            for k in incompatible.unexpected_keys:
                print(f'  {k}')

        print(f'[Pretrain] Loaded encoder from {bert_ckpt_path}')

    def encode_partial(self, partial):
        """
        Encode partial point cloud using pretrained encoder.
        
        Args:
            partial: (B, N=2048, 3)
        Returns:
            x: (B, 128, 384) - encoded tokens
            center: (B, 128, 3) - center coordinates
        """
        neighborhood, center = self.group_divider(partial)
        tokens = self.encoder(neighborhood)
        pos = self.pos_embed(get_pos_embed(self.trans_dim, center))
        x = self.blocks(tokens, pos)
        x = self.norm(x)
        return x, center

    def forward(self, xyz):
        """
        严格对齐 PoinTr.py forward (L101-130).
        
        Args:
            xyz: (B, 2048, 3) partial input
        Returns:
            (coarse_point_cloud, rebuild_points)
            coarse_point_cloud: (B, 2*M=448, 3)
            rebuild_points: (B, 14336+2048=16384, 3)
        """
        # ============ Step 1: Encode partial (REPLACES PoinTr's PCTransformer) ============
        # x: (B, 128, 384), coor: (B, 128, 3)
        x, coor = self.encode_partial(xyz)

        # ============ Step 2: Predict coarse points (与 PoinTr Transformer.py L408-411 一致) ============
        global_feature = self.increase_dim(x.transpose(1, 2))  # (B, 1024, 128)
        global_feature = torch.max(global_feature, dim=-1)[0]  # (B, 1024)

        coarse_point_cloud = self.coarse_pred(global_feature).reshape(xyz.size(0), -1, 3)
        # (B, M, 3) where M=num_query=224

        # ============ Step 3: Build query feature (与 PoinTr Transformer.py L416-419 一致) ============
        bs = xyz.size(0)
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1),
            coarse_point_cloud
        ], dim=-1)  # (B, M, 1024+3=1027)

        q = self.mlp_query(query_feature.transpose(1, 2)).transpose(1, 2)
        # (B, M, trans_dim=384)

        # ============ Step 4: Decoder with KNN (与 PoinTr Transformer.py L412-414, L420-425 一致) ============
        # KNN indices for geometry-aware decoder
        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())  # self KNN on coarse
        cross_knn_index = get_knn_index(
            coor_k=coor.transpose(1, 2).contiguous(),
            coor_q=coarse_point_cloud.transpose(1, 2).contiguous()
        )  # cross KNN: coarse -> partial centers

        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, x, new_knn_index, cross_knn_index)
            else:
                q = blk(q, x)

        # q is now (B, M, 384)

        # ============ Step 5: Build rebuild_feature (与 PoinTr.py L107-115 一致) ============
        B, M, C = q.shape

        global_feature_2 = self.increase_dim_2(q.transpose(1, 2)).transpose(1, 2)  # (B, M, 1024)
        global_feature_2 = torch.max(global_feature_2, dim=1)[0]  # (B, 1024)

        rebuild_feature = torch.cat([
            global_feature_2.unsqueeze(-2).expand(-1, M, -1),  # (B, M, 1024)
            q,                                                   # (B, M, C)
            coarse_point_cloud                                   # (B, M, 3)
        ], dim=-1)  # (B, M, 1027 + C)

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # (BM, C)

        # ============ Step 6: Folding (与 PoinTr.py L121-122 一致) ============
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)  # (B, M, 3, S=64)
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3).reshape(B, -1, 3)
        # (B, M*S=14336, 3)

        # ============ Step 7: Concat with input (与 PoinTr.py L128-130 一致) ============
        inp_sparse = fps(xyz, self.num_query)  # (B, M, 3)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        # (B, 2M=448, 3)
        rebuild_points = torch.cat([rebuild_points, xyz], dim=1).contiguous()
        # (B, 14336+2048=16384, 3)

        ret = (coarse_point_cloud, rebuild_points)
        return ret