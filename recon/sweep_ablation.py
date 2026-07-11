import os
import copy
import itertools
import subprocess
import time
import argparse
import sys
import torch
import pynvml
import numpy as np  # 用于最后计算均值和标准差

# ================= 配置网格 =================
# 生成 70 个种子
seed_list = list(range(2,12))

new_seeds = [
    # 段 1：顺序续接（44~60）
    7, 137, 168, 199,
    222, 256, 314, 365, 404, 512, 666, 777,
    888, 999, 1024, 1234, 2024, 2025, 3407, 42424,
]

SEEDS = {
    "seed": seed_list
}
 
def keep_cfg_C(cfg):
    return True

def generate_exp_grid(grid, keep_fn):
    keys, values = list(grid.keys()), list(grid.values())
    exp_grid = []
    for v in itertools.product(*values):
        cfg = dict(zip(keys, v))
        if keep_fn(cfg):
            exp_grid.append(cfg)
    return exp_grid

def get_free_gpu_memory(gpu_idx):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = info.free / 1024**2
        pynvml.nvmlShutdown()
        return free_mb
    except Exception:
        return 0

def collect_acc(output_dir):
    path = os.path.join(output_dir, "best_metric.txt")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                for line in f:
                    if "acc" in line.lower() and "=" in line:
                        return float(line.split("=")[1].strip())
        except: pass
    return None # 返回 None 表示没跑完或没结果

def main(grid):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="./experiments/seeds")
    parser.add_argument("--max-jobs", type=int, default=10) 
    args = parser.parse_args()

    exp_grid = grid
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    
    available_gpus = copy.deepcopy(gpu_ids)
    processes = []
    all_results = [] # 存储所有 Acc 用于最后计算

    print(f"🚀 任务队列已就绪 | 总计: {len(exp_grid)} | 并行: {args.max_jobs} | GPU 池: {gpu_ids}")

    for i, overrides in enumerate(exp_grid, 1):
        # 1. 保持 tag 唯一
        tag = "aug_".join([f"{k.split('.')[-1]}{v}" for k, v in overrides.items()]).replace(".", "p")
        out_dir = os.path.abspath(os.path.join(args.runs_root, tag))
        
        if os.path.exists(out_dir) and collect_acc(out_dir) is not None:
            print(f"⏭️  Skip #{i}: {tag} 已存在结果")
            all_results.append(collect_acc(out_dir))
            continue
            
        os.makedirs(out_dir, exist_ok=True)

        # 2. 资源调度
        while len(available_gpus) == 0 or len(processes) >= args.max_jobs:
            for pinfo in processes[:]:
                ret = pinfo["proc"].poll()
                if ret is not None:
                    pinfo["log_file"].close()
                    released_gpu = pinfo["gpu_id"]
                    available_gpus.append(released_gpu)
                    acc = collect_acc(pinfo["out_dir"])
                    if acc is not None:
                        all_results.append(acc)
                        print(f"✅ Run #{pinfo['idx']} (Seed {pinfo['seed']}) 完成 | Acc: {acc:.4f}")
                    processes.remove(pinfo)
            time.sleep(10)

        # 3. 核心：保持你原有的 cmd 内容，仅动态添加种子
        gpu_idx = available_gpus.pop(0)

        cmd = [
            sys.executable, "-u", "main.py",
            "--model", "pt",
            "--ckpts", "/irip/yanxu_2023/eccv_rebuttal/MPL-MAE-recon_task/experiments/base/pretrain/pretrain/ckpt-epoch-300.pth",
            "--max_epoch", "200",
            "--root", "/irip/yanxu_2023/eccv_rebuttal/ShapeNetCompletion_npy",
            "--file_format", "npy",
            "--num_workers", "8",
            "--decay_step", "4",
            "--batch_size", "24",
            "--lr", "0.0005",
            "--encoder_lr_scale", "0.1",
            "--weight_decay", "0.0005",
            "--lr_decay", "0.9",
            "--lowest_decay", "0.02",
            "--num_query", "224",
            "--num_pred", "14336",
            "--knn_layer", "1",
            "--trans_dim", "384",
            "--decoder_depth", "8",
            "--seed", str(overrides['seed']),
            "--log_dir", "epoch200_seed_"+str(overrides['seed']),
        ]

        # 动态注入 overrides 里的所有参数
        for k, v in overrides.items():
            if k == "likely_hit":  # 跳过非训练参数
                continue
            cmd.extend([f"--{k}", str(v)])

        # 4. 显存检查
        while get_free_gpu_memory(gpu_idx) < 1024:
            print(f"⏳ GPU {gpu_idx} 显存尚未完全回收，等待中...", end='\r')
            time.sleep(5)


        # 5. 启动
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        log_file = open(os.path.join(out_dir, "run.log"), "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        
        processes.append({
            "proc": proc, "out_dir": out_dir, "seed": overrides['seed'],
            "idx": i, "log_file": log_file, "gpu_id": gpu_idx
        })
        print(f"🔥 [Launch] task {i} on GPU {gpu_idx}")
        time.sleep(5)

    # 6. 收尾与统计
    while len(processes) > 0:
        for pinfo in processes[:]:
            ret = pinfo["proc"].poll()
            if ret is not None:
                pinfo["log_file"].close()
                acc = collect_acc(pinfo["out_dir"])
                if acc is not None: all_results.append(acc)
                available_gpus.append(pinfo["gpu_id"])
                processes.remove(pinfo)
        time.sleep(10)

    

if __name__ == "__main__":
    exp_grid = generate_exp_grid(SEEDS, keep_fn=keep_cfg_C)
    # exp_grid = Params_grid2
    # exp_grid=SEEDS
    print(len(exp_grid))

    # exp_grid_1 = exp_grid[:10]
    # exp_grid_2 = exp_grid[10:]
    main(exp_grid)