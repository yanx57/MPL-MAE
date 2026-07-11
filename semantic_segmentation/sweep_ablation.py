import subprocess
import os
import time
import torch

def main():
    # --- 1. 资源检测 (修正版) ---
    # 不要直接用 range(80)，因为系统可能只允许你用其中的一部分
    try:
        # 获取当前进程真正有权访问的核心编号集合
        allowed_cores = sorted(list(os.sched_getaffinity(0)))
    except AttributeError:
        # 兜底方案（如在非 Linux 系统上）
        allowed_cores = list(range(os.cpu_count()))
    
    total_allowed = len(allowed_cores)
    print(f"🔍 系统检测到 80 核，但当前进程实际可用核心数: {total_allowed}")
    print(f"📍 实际可用核心编号: {allowed_cores if total_allowed < 20 else '列表过长已省略'}")

    available_gpus = list(range(torch.cuda.device_count()))
    
    # 参数设置
    cores_per_process = 8 
    seeds = [4]
    
    # --- 2. 基于“可用核心”的分组逻辑 ---
    # 从 allowed_cores 列表中切片，确保 taskset 申请的每一个核都在权限范围内
    core_groups = [allowed_cores[n:n + cores_per_process] for n in range(0, total_allowed, cores_per_process)]
    
    if not core_groups:
        print("❌ 错误：没有检测到可用的 CPU 核心！")
        return

    processes = []

    # --- 3. 启动进程 ---
    # 如果你希望子进程不输出到终端，建议重新启用 fnull
    # 或者为了调试暂时保持 stdout=None
    for i, seed in enumerate(seeds):
        # A. 轮询分配 GPU
        gpu_idx = available_gpus[i % len(available_gpus)]
        
        # B. 轮询获取 CPU 核心组
        current_group = core_groups[i % len(core_groups)]
        # 将列表转为字符串，如 "0,1,2,3,4,5,6,7"
        core_range = ",".join(map(str, current_group))

        current_log_dir = f"rotate_first_seed_sweep_{seed}_repeoduce"
        # os.makedirs(current_log_dir, exist_ok=True) # 建议开启以自动创建目录

        # C. 构建命令
        # 强制指定 --gpu 0，因为 CUDA_VISIBLE_DEVICES 已经做了物理隔离
        cmd = [
            "taskset", "-c", core_range,
            "python", "main.py",
            "--ckpts", "/irip/yanxu_2023/gumble-test/ckpt-epoch-300.pth",
            "--root", "../data/s3dis",
            "--learning_rate", "0.0002",
            "--gpu", str(gpu_idx), 
            "--epoch", "60",
            "--log_dir", current_log_dir,
            "--seed", str(seed)
        ]

        env = os.environ.copy()
        # 物理隔离 GPU：子进程只能看到被分配的那张卡
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        
        # D. 启动
        try:
            # 如果不希望终端被刷屏，请将 stdout/stderr 改回 os.devnull 或文件句柄
            proc = subprocess.Popen(
                cmd, 
                env=env, 
                stdout=None, 
                stderr=None
            )
            processes.append(proc)
            print(f"✅ [Launch] Seed {seed} | GPU {gpu_idx} | Cores {core_range}")
        except Exception as e:
            print(f"❌ [Error] Failed to launch seed {seed}: {e}")
        
        time.sleep(2)

    print(f"\n🚀 所有进程已在后台运行 (共 {len(processes)} 个)")
    
    # 阻塞主进程直到全部结束
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()