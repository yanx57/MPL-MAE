"""
PCN dataset for point cloud completion.

Strictly aligned with PoinTr/datasets/PCNDataset.py, but simplified:
- Removed distributed/multi-GPU dependencies
- Inlined data_transforms (RandomSamplePoints, RandomMirrorPoints, ToTensor)
- Loads .pcd / .npy / .h5 files using open3d / numpy / h5py
- Uses the same PCN.json category index file from PoinTr
- Same 8-rendering training, 1-rendering testing protocol
- Same 16384 GT points, 2048 partial points after RandomSamplePoints

Expected directory structure (PoinTr's PCN format from DATASET.md):
    PCN/
        PCN.json   <- category index (also called category_file in PoinTr)
        train/
            partial/<taxonomy_id>/<model_id>/00.pcd ... 07.pcd
            complete/<taxonomy_id>/<model_id>.pcd
        test/
            partial/<taxonomy_id>/<model_id>/00.pcd
            complete/<taxonomy_id>/<model_id>.pcd
        val/
            ...
"""
import os
import json
import random
import numpy as np
import torch
import torch.utils.data as data


# ============================================================
# IO utilities
# ============================================================

def _load_pcd(file_path):
    """Load .pcd file using open3d"""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Please install open3d: pip install open3d")
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points, dtype=np.float32)


def _load_npy(file_path):
    return np.load(file_path).astype(np.float32)


def _load_h5(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        return f['data'][:].astype(np.float32)


def _load_point_cloud(file_path):
    """Auto-dispatch based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pcd':
        return _load_pcd(file_path)
    elif ext == '.npy':
        return _load_npy(file_path)
    elif ext == '.h5':
        return _load_h5(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ============================================================
# Data transforms (与 PoinTr/datasets/data_transforms.py 一致)
# ============================================================

class RandomSamplePoints:
    """随机采样到固定点数 (上采样时重复采样)"""
    def __init__(self, n_points):
        self.n_points = n_points

    def __call__(self, points):
        n = len(points)
        if n >= self.n_points:
            indices = np.random.choice(n, self.n_points, replace=False)
        else:
            # 不足时重复采样
            indices = np.random.choice(n, self.n_points, replace=True)
        return points[indices]


class RandomMirrorPoints:
    """随机镜像 (训练增强)"""
    def __call__(self, partial, gt):
        # 50% 概率沿 X 轴镜像, 50% 概率沿 Z 轴镜像 (与 PoinTr 一致)
        rnd_value = np.random.uniform(0, 1)
        trfm_mat = np.eye(3, dtype=np.float32)
        if rnd_value <= 0.25:
            trfm_mat[0, 0] = -1  # X mirror
        elif rnd_value <= 0.5:
            trfm_mat[2, 2] = -1  # Z mirror
        # 0.5 概率不变
        partial = partial @ trfm_mat
        gt = gt @ trfm_mat
        return partial, gt


# ============================================================
# PCN Dataset (与 PoinTr/datasets/PCNDataset.py 严格对齐)
# ============================================================

class PCN(data.Dataset):
    """
    PCN dataset class (与 PoinTr 完全对齐).
    
    Args:
        root: PCN data root directory (containing train/test/val and PCN.json)
        category_file: path to PCN.json (default: <root>/PCN.json)
        subset: 'train' / 'val' / 'test'
        n_points: number of GT points (default 16384, PCN standard)
        cars: if True, only use car category (KITTI fine-tune)
        partial_n_points: target partial points after RandomSamplePoints (default 2048)
        file_format: 'pcd' / 'npy' / 'h5' (auto-detected if None)
    """
    def __init__(self, root, category_file=None, subset='train',
                 n_points=16384, cars=False, partial_n_points=2048,
                 file_format='pcd'):
        super().__init__()
        self.root = root
        self.subset = subset
        self.npoints = n_points
        self.cars = cars
        self.partial_n_points = partial_n_points
        self.file_format = file_format

        # Default category file path
        if category_file is None:
            category_file = os.path.join(root, 'PCN.json')
        self.category_file = category_file

        # PoinTr's path templates (与 PoinTr cfgs/dataset_configs/PCN.yaml 一致)
        # 实际格式: <subset>/partial/<taxonomy_id>/<model_id>/<rendering_id>.pcd
        self.partial_points_path = os.path.join(
            root, '%s/partial/%s/%s/%02d.' + file_format
        )
        self.complete_points_path = os.path.join(
            root, '%s/complete/%s/%s.' + file_format
        )

        # Load category index
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        if cars:
            self.dataset_categories = [
                dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343'
            ]

        # 8 renderings for train, 1 for val/test (与 PoinTr 一致)
        self.n_renderings = 8 if subset == 'train' else 1

        # Build file list
        self.file_list = self._get_file_list(subset, self.n_renderings)

        # Transforms
        self.random_sample = RandomSamplePoints(n_points=partial_n_points)
        self.random_mirror = RandomMirrorPoints() if subset == 'train' else None

        print(f'[PCN] {subset}: {len(self.file_list)} samples '
              f'across {len(self.dataset_categories)} categories '
              f'(n_renderings={self.n_renderings})')

    def _get_file_list(self, subset, n_renderings=1):
        file_list = []
        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_paths': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # Random viewpoint for train, fixed (0) for test (与 PoinTr 一致)
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        # Load partial and gt
        partial = _load_point_cloud(sample['partial_paths'][rand_idx])
        gt = _load_point_cloud(sample['gt_path'])

        # Sanity check
        assert gt.shape[0] == self.npoints, \
            f'GT point count mismatch: got {gt.shape[0]}, expected {self.npoints}'

        # RandomSamplePoints on partial (PoinTr: 'objects': ['partial'])
        partial = self.random_sample(partial)

        # RandomMirrorPoints on both (only train)
        if self.random_mirror is not None:
            partial, gt = self.random_mirror(partial, gt)

        # ToTensor
        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()

        # Return format: (taxonomy_id, model_id, (partial, gt))
        # 与 PoinTr/datasets/PCNDataset.py 一致, runner.py L93 期望此格式
        return sample['taxonomy_id'], sample['model_id'], (partial, gt)


# ============================================================
# Sanity check
# ============================================================

if __name__ == '__main__':
    # 修改为你的实际路径
    DATA_ROOT = '/path/to/PCN'

    print(f'Testing PCN dataset at {DATA_ROOT}')

    train_set = PCN(root=DATA_ROOT, subset='train', file_format='pcd')
    test_set = PCN(root=DATA_ROOT, subset='test', file_format='pcd')

    print(f'Train: {len(train_set)}')
    print(f'Test: {len(test_set)}')

    sample = train_set[0]
    tax_id, model_id, (partial, gt) = sample
    print(f'\nSample 0:')
    print(f'  Taxonomy: {tax_id}')
    print(f'  Model: {model_id}')
    print(f'  Partial: {partial.shape} {partial.dtype}')
    print(f'  GT: {gt.shape} {gt.dtype}')

    loader = torch.utils.data.DataLoader(train_set, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    tax_ids, model_ids, (p, g) = batch
    print(f'\nBatch shapes:')
    print(f'  partial: {p.shape}')
    print(f'  gt: {g.shape}')