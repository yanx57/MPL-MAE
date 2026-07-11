import os
import copy
import itertools
import subprocess
import time
import argparse
import yaml
import sys
import torch
import pynvml

# ================= 配置网格 =================
seed_list1 = []
for i in range(105, 301):
    seed_list1.append(i)





def keep_cfg_C(cfg):
    return True


best_GRID = {
    "optimizer.kwargs.lr": [2.0e-5],  # backbone 固定
    "optimizer.head.lr": [2.7e-4],  # 大 ratio, 8 个值
    "optimizer.kwargs.weight_decay": [0.05],
    "optimizer.head.weight_decay": [0.015],  # 4 个
    "model.drop_path_rate": [0.0],
    "model.head_dp": [0.5],
    "seed": [20]
}


def generate_exp_grid(grid, keep_fn):
    keys, values = list(grid.keys()), list(grid.values())
    exp_grid = []
    for v in itertools.product(*values):
        cfg = dict(zip(keys, v))
        if keep_fn(cfg):
            exp_grid.append(cfg)
    return exp_grid

def get_free_gpu_memory(gpu_idx):
    """获取指定 GPU 的剩余显存 (MB)"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = info.free / 1024**2
        pynvml.nvmlShutdown()
        return free_mb
    except Exception as e:
        print(f"监控 GPU 失败: {e}")
        return 0

def set_nested(cfg, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        cfg = cfg.setdefault(p, {})
    cfg[parts[-1]] = value

def collect_acc(output_dir):
    path = os.path.join(output_dir, "best_metric.txt")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                for line in f:
                    if "acc" in line.lower() and "=" in line:
                        return float(line.split("=")[1].strip())
        except: pass
    return 0.0




def main(grid):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,  default="./tmp_cfgs/finetune_scan_objbg.yaml")
    parser.add_argument("--runs_root", type=str, default="./experiments/ft_objbg-finetune_grid_droppath")
    parser.add_argument("--max-jobs", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

  
    exp_grid = grid

    
    # 获取可用 GPU 列表并初始化资源池
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    
    # 确保并行数不超过 GPU 数量（除非单卡跑多任务）
    available_gpus = copy.deepcopy(gpu_ids)
    
    print(f"🚀 任务队列已就绪 | 总计: {len(exp_grid)} | 并行限制: {args.max_jobs} | 可用 GPU 池: {gpu_ids}")
    
    processes = []
    best_acc_record = {"value": 0.0, "idx": 0}

    for i, overrides in enumerate(exp_grid, 1):
        # 1. 生成唯一标识和输出路径
        tag = "_".join([f"{k.split('.')[-1]}{v}" for k, v in overrides.items()]).replace(".", "p")
        run_name = f"aug_test_{tag}"
        out_dir = os.path.abspath(os.path.join(args.runs_root, run_name))
        
        if os.path.exists(out_dir) and os.path.exists(os.path.join(out_dir, "best_metric.txt")):
            print(f"⏭️  Skip #{i}: {run_name} (Already exists)")
            continue
            
        os.makedirs(out_dir, exist_ok=True)

        # 2. 调度监控逻辑：如果没有空闲 GPU 或进程达到上限，则等待
        while len(available_gpus) == 0 or len(processes) >= args.max_jobs:
            for pinfo in processes[:]:
                ret = pinfo["proc"].poll()
                if ret is not None:
                    pinfo["log_file"].close()
                    
                    # 【核心】回收该进程占用的 GPU ID 回资源池
                    released_gpu = pinfo["gpu_id"]
                    available_gpus.append(released_gpu)
                    
                    if ret != 0:
                        print(f"\n❌ Run #{pinfo['idx']} 异常退出 (GPU {released_gpu})")
                    else:
                        acc = collect_acc(pinfo["out_dir"])
                        if acc > best_acc_record["value"]:
                            best_acc_record = {"value": acc, "idx": pinfo["idx"]}
                        # 重点高亮 95.7 以上的结果
                        star = " ⭐⭐⭐" if acc >= 95.7 else ""
                        print(f"✅ Run #{pinfo['idx']} 完成 [GPU {released_gpu}] | Acc: {acc:.4f}{star} | 🏆 最佳: {best_acc_record['value']:.4f}")
                    
                    if os.path.exists(pinfo["cfg_path"]): os.remove(pinfo["cfg_path"])
                    processes.remove(pinfo)
            
            if len(available_gpus) == 0:
                time.sleep(10) # 暂无空闲卡，稍后轮询

        # 3. 准备配置和指令
        cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items(): set_nested(cfg, k, v)
        
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), f"tmp_{run_name}.yaml")
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)

        # 4. 从池中取出一张显卡
        gpu_idx = available_gpus.pop(0)

        # 物理显存检查，确保真正可用
        wait_start = time.time()
        while get_free_gpu_memory(gpu_idx) < 1024:
            print(f"⏳ GPU {gpu_idx} 显存尚未完全回收，等待中...", end='\r')
            time.sleep(5)

        cmd = [
            sys.executable, "-u", "main.py",
            "--config", cfg_path,
            "--finetune_model",
            "--exp_name", out_dir,
            "--ckpts", "/irip/yanxu_2023/gumble-test/ckpt-epoch-300.pth"
        ]

        # 5. 启动进程
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # 防止碎片化
        
        log_file = open(os.path.join(out_dir, "run.log"), "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        
        processes.append({
            "proc": proc, "out_dir": out_dir, "cfg_path": cfg_path, 
            "idx": i, "log_file": log_file, "gpu_id": gpu_idx
        })
        print(f"🔥 [Launch] Run #{i}/{len(exp_grid)} on GPU {gpu_idx} | Active: {len(processes)} | Free GPUs: {available_gpus}")
        time.sleep(5) # 错峰启动，保护 IO 和显存申请

    # 6. 收尾工作：处理最后运行中的任务
    print("\n⌛ 所有任务已提交，等待最后进程结束...")
    while len(processes) > 0:
        for pinfo in processes[:]:
            ret = pinfo["proc"].poll()
            if ret is not None:
                pinfo["log_file"].close()
                acc = collect_acc(pinfo["out_dir"])
                released_gpu = pinfo["gpu_id"]
                if acc > best_acc_record["value"]:
                    best_acc_record = {"value": acc, "idx": pinfo["idx"]}
                print(f"✅ Final Run #{pinfo['idx']} 完成 [GPU {released_gpu}] | Acc: {acc:.4f}")
                if os.path.exists(pinfo["cfg_path"]): os.remove(pinfo["cfg_path"])
                processes.remove(pinfo)
        time.sleep(10)

    print(f"\n🎯 暴力搜索圆满结束！")
    print(f"🏆 最高精度: {best_acc_record['value']:.4f} (来自 Run #{best_acc_record['idx']})")

if __name__ == "__main__":
    # exp_grid_A = generate_exp_grid(GRID_A, keep_fn=keep_cfg_A)
    # main(exp_grid_A)
    # exp_grid_B = generate_exp_grid(GRID_B, keep_fn=keep_cfg_B)
    # main(exp_grid_B)
    exp_grid_F = generate_exp_grid(droppath_GRID, keep_fn=keep_cfg_C)
    main(exp_grid_F)