"""
DCP-v1 protocol aligned point cloud registration training.

Strict alignment with DCP-v1:
- Single forward pass per batch (src -> tgt only)
- Only AB direction loss (no BA loss, no cycle loss)
- Identity-based pose loss (DCP standard)
- Full backbone fine-tuning with grouped LR
- DCP-aligned MSE/RMSE/MAE metrics (Euler-based for R, per-dim for t)
- Direct comparison with DCP-v1 Table 1 numbers
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import datetime
import logging
import importlib
import shutil
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import ModelNet40Registration
import misc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt')
    parser.add_argument('--model-prefix', type=str, default='MAE_encoder', choices=['MAE_encoder'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=250, type=int)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--warmup_epoch', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--encoder_lr_scale', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--grad_clip', default=10.0, type=float)
    parser.add_argument('--log_dir', type=str, default='./exp')
    parser.add_argument('--ckpts', type=str, default='../model_zoo/pretrain.pth')
    parser.add_argument('--root', type=str, default='../data/modelnet40_normal_resampled/')
    
    # DCP 协议参数
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--setting', type=str, default='1', choices=['1', '2', '3'],
                        help='DCP setting: 1=clean unseen shapes, 2=clean unseen categories, 3=gaussian noise')
    parser.add_argument('--rotation_factor', type=int, default=4,
                        help='rotation range = pi/factor per axis')
    
    return parser.parse_args()


# ============================================================
# Validate (DCP-v1 协议: 只评估 AB 方向)
# ============================================================

def validate(logger, classifier, testDataLoader):
    """
    DCP-v1 协议对齐的评估:
    - 只评估 AB 方向 (与 DCP-v1 论文一致)
    - 收集所有样本的 (B, 3) 欧拉角差和平移差
    - 在所有样本所有维度上统一计算 MSE/RMSE/MAE
    
    返回的指标对应 DCP 论文 Table 1 的 6 列:
    MSE(R), RMSE(R), MAE(R), MSE(t), RMSE(t), MAE(t)
    """
    classifier.eval()
    
    all_loss = []
    all_r_diff = []      # (N, 3) 欧拉角差异 (degrees)
    all_t_diff = []      # (N, 3) 平移差异
    all_geo = []         # (N,) 辅助 geodesic angle
    
    with torch.no_grad():
        for batch in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            src, tgt, R_ab, t_ab, _, _, _, _ = batch  # 只用 AB
            
            src = src.float().cuda()
            tgt = tgt.float().cuda()
            R_ab = R_ab.float().cuda()
            t_ab = t_ab.float().cuda()
            
            # 单方向 forward
            R_pred, t_pred = classifier(src, tgt)
            loss = classifier.pose_loss(R_pred, t_pred, R_ab, t_ab)
            
            r_diff = classifier.rotation_error_per_sample(R_pred, R_ab)
            t_diff = classifier.translation_error_per_sample(t_pred, t_ab)
            geo = classifier.rotation_geodesic_per_sample(R_pred, R_ab)
            
            all_loss.append(loss.item())
            all_r_diff.append(r_diff.cpu())
            all_t_diff.append(t_diff.cpu())
            all_geo.append(geo.cpu())
    
    # 拼接
    all_r_diff = torch.cat(all_r_diff)      # (N, 3)
    all_t_diff = torch.cat(all_t_diff)      # (N, 3)
    all_geo = torch.cat(all_geo)            # (N,)
    
    # DCP 标准指标
    r_diff_flat = all_r_diff.flatten().float()  # (N*3,)
    mse_R = (r_diff_flat ** 2).mean().item()
    rmse_R = float(np.sqrt(mse_R))
    mae_R = r_diff_flat.abs().mean().item()
    
    t_diff_flat = all_t_diff.flatten().float()  # (N*3,)
    mse_t = (t_diff_flat ** 2).mean().item()
    rmse_t = float(np.sqrt(mse_t))
    mae_t = t_diff_flat.abs().mean().item()
    
    median_R = r_diff_flat.abs().median().item()
    median_t = t_diff_flat.abs().median().item()
    
    metrics = {
        'loss': np.mean(all_loss),
        'mse_R': mse_R,
        'rmse_R': rmse_R,
        'mae_R': mae_R,
        'mse_t': mse_t,
        'rmse_t': rmse_t,
        'mae_t': mae_t,
        'median_R': median_R,
        'median_t': median_t,
        'geo_mean': all_geo.mean().item(),
        'geo_median': all_geo.median().item(),
    }
    return metrics


def print_dcp_metrics(log_fn, metrics, prefix='[Eval]'):
    """以 DCP 论文 Table 1 的格式打印指标"""
    log_fn('%s Loss: %.6f' % (prefix, metrics['loss']))
    log_fn('%s Rotation:    MSE=%.6f  RMSE=%.6f°  MAE=%.6f°  Median=%.6f°' % (
        prefix, metrics['mse_R'], metrics['rmse_R'], metrics['mae_R'], metrics['median_R']))
    log_fn('%s Translation: MSE=%.6e  RMSE=%.6f   MAE=%.6f   Median=%.6f' % (
        prefix, metrics['mse_t'], metrics['rmse_t'], metrics['mae_t'], metrics['median_t']))
    log_fn('%s Geodesic R:  Mean=%.6f°  Median=%.6f° (auxiliary)' % (
        prefix, metrics['geo_mean'], metrics['geo_median']))


# ============================================================
# Main
# ============================================================

def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('registry')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    if args.seed != -1:
        log_string(f'Set random seed to {args.seed}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic)
    
    '''DATA'''
    setting_config = {
        '1': {'gaussian_noise': False, 'unseen': False},
        '2': {'gaussian_noise': False, 'unseen': True},
        '3': {'gaussian_noise': True, 'unseen': False},
    }
    config = setting_config[args.setting]
    log_string(f'DCP Setting {args.setting}: noise={config["gaussian_noise"]}, unseen={config["unseen"]}')
    log_string('Training mode: DCP-v1 style (single direction, no cycle loss)')
    
    TRAIN_DATASET = ModelNet40Registration(
        root=args.root, split='train',
        num_points=args.num_points,
        gaussian_noise=config['gaussian_noise'],
        unseen=config['unseen'],
        factor=args.rotation_factor,
    )
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
        num_workers=8, drop_last=True
    )
    TEST_DATASET = ModelNet40Registration(
        root=args.root, split='test',
        num_points=args.num_points,
        gaussian_noise=config['gaussian_noise'],
        unseen=config['unseen'],
        factor=args.rotation_factor,
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=8, drop_last=False
    )
    log_string("Train: %d | Test: %d" % (len(TRAIN_DATASET), len(TEST_DATASET)))
    
    '''MODEL'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    classifier = MODEL.get_model().cuda()
    log_string('# parameters: %d' % sum(param.numel() for param in classifier.parameters()))
    
    if args.test:
        state_dict = torch.load(args.ckpts, map_location='cpu')
        classifier.load_state_dict(state_dict['model_state_dict'], strict=True)
        test_metrics = validate(logger, classifier, testDataLoader)
        print_dcp_metrics(log_string, test_metrics, prefix='[Test]')
        return
    
    '''Load pretrained'''
    if args.ckpts is not None:
        assert args.model_prefix is not None
        pre_load_weight = classifier.pos_embed[0].weight.data.clone()
        classifier.load_model_from_ckpt(args.ckpts, model_prefix=args.model_prefix)
        post_load_weight = classifier.pos_embed[0].weight.data
        diff = (pre_load_weight - post_load_weight).abs().sum().item()
        if diff > 1e-6:
            log_string("✅ pos_embed loaded from pretrained checkpoint")
        else:
            log_string("⚠️ pos_embed NOT loaded - still randomly initialized!")
    
    # 全模型微调
    for name, param in classifier.named_parameters():
        param.requires_grad = True
    log_string('All parameters are trainable (full fine-tuning).')
    
    '''Optimizer with grouped LR'''
    decay = []
    no_decay = []
    head_params = []
    
    for name, param in classifier.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('svd_head'):
            head_params.append(param)
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    encoder_lr = args.learning_rate * args.encoder_lr_scale
    
    param_groups = [
        {'params': head_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': decay, 'lr': encoder_lr, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'lr': encoder_lr, 'weight_decay': 0.},
    ]
    
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    log_string(f'Encoder LR: {encoder_lr:.6f} | Head LR: {args.learning_rate:.6f}')
    log_string(f'Trainable params: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}')
    
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epoch, t_mul=1, lr_min=1e-6,
        decay_rate=0.1, warmup_lr_init=1e-6, warmup_t=args.warmup_epoch,
        cycle_limit=1, t_in_epochs=True
    )
    
    '''Initial validation'''
    test_metrics = validate(logger, classifier, testDataLoader)
    print_dcp_metrics(log_string, test_metrics, prefix='[Init]')
    best_metrics = test_metrics.copy() 
    global_epoch = 0
    
    best_mae_R = test_metrics['mae_R']
    best_mae_t = test_metrics['mae_t']
    global_epoch = 0
    
    '''Training loop (DCP-v1 风格: 单方向 forward + 单方向 loss)'''
    for epoch in range(args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        loss_batch = []
        
        for i, batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # ============ 只取 AB 方向数据 (DCP-v1 风格) ============
            src, tgt, R_ab, t_ab, _, _, _, _ = batch
            
            src = src.float().cuda()
            tgt = tgt.float().cuda()
            R_ab = R_ab.float().cuda()
            t_ab = t_ab.float().cuda()
            
            # ============ 标准训练循环 ============
            optimizer.zero_grad()
            
            # 单方向 forward
            R_pred, t_pred = classifier(src, tgt)
            
            # DCP-v1 标准 loss (identity-based)
            loss = classifier.pose_loss(R_pred, t_pred, R_ab, t_ab)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip, norm_type=2)
            optimizer.step()
            
            loss_batch.append(loss.detach().cpu().item())
        
        scheduler.step(epoch)
        
        train_loss = np.mean(loss_batch)
        log_string('Train loss: %.5f' % train_loss)
        log_string('LR: head=%.6f, encoder=%.6f' % (
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        
        # Validation
        test_metrics = validate(logger, classifier, testDataLoader)
        print_dcp_metrics(log_string, test_metrics, prefix=f'[Epoch {epoch}]')
        
        # ============ 用 MAE(R) 作为 best 判定 (类似 DCP 用 loss) ============
        if test_metrics['mae_R'] < best_metrics['mae_R']:
            # 同步更新所有指标 (来自同一个 epoch)
            best_metrics = test_metrics.copy()
            best_metrics['epoch'] = epoch  # 记录 best epoch
            
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string(f'Saving best model -> {savepath}')
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
            }
            torch.save(state, savepath)
        
        # 打印 best (所有指标都来自同一 epoch)
        log_string('=== Best metrics (from epoch %d) ===' % best_metrics.get('epoch', 0))
        log_string('  MSE(R) = %.6f, RMSE(R) = %.6f°, MAE(R) = %.6f°' % (
            best_metrics['mse_R'], best_metrics['rmse_R'], best_metrics['mae_R']))
        log_string('  MSE(t) = %.6e, RMSE(t) = %.6f, MAE(t) = %.6f' % (
            best_metrics['mse_t'], best_metrics['rmse_t'], best_metrics['mae_t']))
        
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)