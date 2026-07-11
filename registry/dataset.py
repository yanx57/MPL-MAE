"""
ModelNet40 Registration Dataset, FULLY aligned with DCP protocol.

Returns 8 values per sample (matching DCP official):
    (src, tgt, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba)

Supports DCP Settings 1/2/3:
- Setting 1 (Clean Unseen Shapes): gaussian_noise=False, unseen=False
- Setting 2 (Clean Unseen Categories): gaussian_noise=False, unseen=True  
- Setting 3 (Gaussian Noise): gaussian_noise=True, unseen=False

DCP protocol details:
- Input points: 1024 (first-N from 2048 in H5)
- Rotation: per-axis uniform in [0, π/factor], default factor=4 -> [0°, 45°]
- Translation: per-dim uniform in [-0.5, 0.5]
- Rotation matrix: R_ab = Rx @ Ry @ Rz (DCP convention)
- Gaussian noise: σ=0.01, clip [-0.05, 0.05], applied to BOTH src and tgt
- Test set uses np.random.seed(item) for reproducibility
"""
import os
import sys
import glob
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# ============================================================
# Data Loading: 优先用 DCP H5, 备用 normal_resampled
# ============================================================

def _load_data_h5(data_dir, partition):
    """从 DCP 标准的 modelnet40_ply_hdf5_2048 加载数据"""
    import h5py
    
    h5_dir = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')
    if not os.path.exists(h5_dir):
        return None, None
    
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(h5_dir, f'ply_data_{partition}*.h5'))):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    
    if len(all_data) == 0:
        return None, None
    
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze()
    return all_data, all_label


def _load_data_normal_resampled(data_dir, partition):
    """从 modelnet40_normal_resampled 加载数据 (备用)"""
    catfile = os.path.join(data_dir, 'modelnet40_shape_names.txt')
    if not os.path.exists(catfile):
        return None, None
    
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    
    split_file = os.path.join(data_dir, f'modelnet40_{partition}.txt')
    shape_ids = [line.rstrip() for line in open(split_file)]
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
    
    all_data = []
    all_label = []
    for i, shape_id in enumerate(shape_ids):
        fn = os.path.join(data_dir, shape_names[i], shape_id + '.txt')
        point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
        point_set = point_set[:2048, :3]
        all_data.append(point_set)
        all_label.append(classes[shape_names[i]])
    
    all_data = np.stack(all_data, axis=0)
    all_label = np.array(all_label, dtype=np.int64)
    return all_data, all_label


def load_modelnet40(data_root, partition):
    """优先 H5, 备用 normal_resampled"""
    data, label = _load_data_h5(data_root, partition)
    if data is not None:
        print(f'[Dataset] Loaded {partition}: {len(data)} samples from H5 format')
        return data, label
    
    data, label = _load_data_normal_resampled(data_root, partition)
    if data is not None:
        print(f'[Dataset] Loaded {partition}: {len(data)} samples from normal_resampled format')
        return data, label
    
    raise RuntimeError(
        f'Could not load ModelNet40 from {data_root}.\n'
        f'Expected one of:\n'
        f'  - {data_root}/modelnet40_ply_hdf5_2048/ply_data_{partition}*.h5\n'
        f'  - {data_root}/modelnet40_{partition}.txt + per-class folders\n'
    )


# ============================================================
# Helper
# ============================================================

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    """DCP 官方的 Gaussian noise 实现 (σ=0.01, clip=0.05)"""
    N, C = pointcloud.shape
    pointcloud = pointcloud + np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return pointcloud


# ============================================================
# Dataset (returns 8 values, fully aligned with DCP)
# ============================================================

class ModelNet40Registration(Dataset):
    """
    Returns (per __getitem__):
        src:        (N, 3) source point cloud
        tgt:        (N, 3) target point cloud (rotated + translated + optional noise)
        R_ab:       (3, 3) GT rotation source -> target
        t_ab:       (3,)   GT translation source -> target
        R_ba:       (3, 3) GT rotation target -> source (= R_ab.T)
        t_ba:       (3,)   GT translation target -> source (= -R_ab.T @ t_ab)
        euler_ab:   (3,)   euler angles for R_ab [angle_z, angle_y, angle_x]
        euler_ba:   (3,)   euler angles for R_ba (= -euler_ab[::-1])
    """
    
    def __init__(self, root, split='train', num_points=1024, 
                 gaussian_noise=True, unseen=False, factor=4):
        super().__init__()
        
        partition = split
        assert partition in ['train', 'test']
        
        self.data, self.label = load_modelnet40(root, partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.factor = factor
        
        # Setting 2: Unseen Categories
        if self.unseen:
            if self.partition == 'test':
                mask = self.label >= 20
                self.data = self.data[mask]
                self.label = self.label[mask]
            elif self.partition == 'train':
                mask = self.label < 20
                self.data = self.data[mask]
                self.label = self.label[mask]
        
        print(f'[Dataset] {partition} set: {len(self.data)} samples '
              f'(noise={gaussian_noise}, unseen={unseen}, factor={factor})')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        # Step 1: 加载点云 (与 DCP 一致: 取前 num_points 个点)
        pointcloud = self.data[item][:self.num_points].copy()  # (N, 3)
        
        # Step 2: 测试集固定 seed (与 DCP 一致)
        if self.partition != 'train':
            np.random.seed(item)
        
        # Step 3: 生成随机旋转角度 (DCP 标准)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor
        
        # Step 4: 构造旋转矩阵 (DCP 顺序: Rx @ Ry @ Rz)
        cosx, cosy, cosz = np.cos(anglex), np.cos(angley), np.cos(anglez)
        sinx, siny, sinz = np.sin(anglex), np.sin(angley), np.sin(anglez)
        
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T  # 反向旋转
        
        # Step 5: 生成随机平移
        translation_ab = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ], dtype=np.float32)
        translation_ba = -R_ba.dot(translation_ab)
        
        # Step 6: 应用变换生成 target
        pointcloud1 = pointcloud.copy()  # source: (N, 3)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1) + translation_ab[np.newaxis, :]
        
        # Step 7: Gaussian noise (Setting 3)
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)
        
        # Step 8: Shuffle 点的顺序
        idx1 = np.random.permutation(self.num_points)
        idx2 = np.random.permutation(self.num_points)
        pointcloud1 = pointcloud1[idx1]
        pointcloud2 = pointcloud2[idx2]
        
        # Step 9: Euler angles (与 DCP 一致)
        euler_ab = np.array([anglez, angley, anglex], dtype=np.float32)
        euler_ba = -euler_ab[::-1].copy()  # 注意: 必须 copy, 否则 negative stride 报错
        
        # Step 10: 返回 8 个值 (完全对齐 DCP)
        return (
            pointcloud1.astype(np.float32),     # src: (N, 3)
            pointcloud2.astype(np.float32),     # tgt: (N, 3)
            R_ab.astype(np.float32),            # (3, 3)
            translation_ab.astype(np.float32),  # (3,)
            R_ba.astype(np.float32),            # (3, 3)
            translation_ba.astype(np.float32),  # (3,)
            euler_ab.astype(np.float32),        # (3,)
            euler_ba.astype(np.float32),        # (3,)
        )


# ============================================================
# Sanity check
# ============================================================

if __name__ == '__main__':
    import torch
    
    print("=" * 70)
    print("Testing DCP Setting 3 (Gaussian Noise) - 8 returns")
    print("=" * 70)
    
    # 修改路径为你的实际路径
    DATA_ROOT = '../data/modelnet40_normal_resampled'
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = '../data'
    
    train_set = ModelNet40Registration(
        root=DATA_ROOT, split='train', num_points=1024,
        gaussian_noise=True, unseen=False, factor=4
    )
    
    sample = train_set[0]
    src, tgt, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba = sample
    
    print(f"\nReturn structure (8 values):")
    print(f"  1. src:      {src.shape} {src.dtype}")
    print(f"  2. tgt:      {tgt.shape} {tgt.dtype}")
    print(f"  3. R_ab:     {R_ab.shape} {R_ab.dtype}")
    print(f"  4. t_ab:     {t_ab.shape} {t_ab.dtype}")
    print(f"  5. R_ba:     {R_ba.shape} {R_ba.dtype}")
    print(f"  6. t_ba:     {t_ba.shape} {t_ba.dtype}")
    print(f"  7. euler_ab: {euler_ab.shape} {euler_ab.dtype}")
    print(f"  8. euler_ba: {euler_ba.shape} {euler_ba.dtype}")
    
    # 验证 R_ba = R_ab.T
    assert np.allclose(R_ba, R_ab.T), "R_ba != R_ab.T!"
    print(f"\n✅ R_ba == R_ab.T")
    
    # 验证 R_ab @ R_ba = I
    assert np.allclose(R_ab @ R_ba, np.eye(3), atol=1e-5), "R_ab @ R_ba != I!"
    print(f"✅ R_ab @ R_ba == I")
    
    # 验证 t_ba = -R_ba @ t_ab
    assert np.allclose(t_ba, -R_ba @ t_ab, atol=1e-5), "t_ba relation incorrect!"
    print(f"✅ t_ba == -R_ba @ t_ab")
    
    # DataLoader test
    loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    src_b, tgt_b, R_ab_b, t_ab_b, R_ba_b, t_ba_b, euler_ab_b, euler_ba_b = batch
    print(f"\nBatch shapes:")
    print(f"  src:      {src_b.shape}")
    print(f"  tgt:      {tgt_b.shape}")
    print(f"  R_ab:     {R_ab_b.shape}")
    print(f"  t_ab:     {t_ab_b.shape}")
    print(f"  R_ba:     {R_ba_b.shape}")
    print(f"  t_ba:     {t_ba_b.shape}")
    print(f"  euler_ab: {euler_ab_b.shape}")
    print(f"  euler_ba: {euler_ba_b.shape}")
    
    print("\n✅ All checks passed!")