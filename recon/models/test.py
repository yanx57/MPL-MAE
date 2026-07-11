# verify_euler.py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def npmat2euler_via_scipy(R, seq='zyx'):
    R_np = R.detach().cpu().numpy()
    eulers = []
    for i in range(R_np.shape[0]):
        e = Rotation.from_matrix(R_np[i]).as_euler(seq, degrees=True)
        eulers.append(e)
    eulers = np.stack(eulers, axis=0)
    return torch.from_numpy(eulers).to(R.device).to(R.dtype)

np.random.seed(0)
torch.manual_seed(0)

# 测试 100 个随机旋转矩阵
max_diff = 0
for _ in range(100):
    R_np = Rotation.random().as_matrix()  # (3, 3)
    R_torch = torch.from_numpy(R_np).float().unsqueeze(0)  # (1, 3, 3)
    
    # scipy 版本 (DCP 用的)
    euler_scipy = Rotation.from_matrix(R_np).as_euler('zyx', degrees=True)  # (3,)
    
    # 你的 PyTorch 版本
    euler_torch = npmat2euler_via_scipy(R_torch).squeeze(0).numpy()  # (3,)
    
    diff = np.abs(euler_scipy - euler_torch).max()
    max_diff = max(max_diff, diff)
    
print(f"Max diff between scipy and torch: {max_diff:.6e}")
print(f"✅ Aligned" if max_diff < 0.01 else f"❌ NOT aligned!")