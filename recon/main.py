"""
Point cloud completion training, strictly aligned with PoinTr/tools/runner.py.

Key alignment with PoinTr runner:
- Loss: ChamferDistanceL1 for both sparse and dense (1:1, no weighting)
- Total loss: sparse_loss + dense_loss (与 runner.py L118 一致)
- Validation: report SparseLossL1, SparseLossL2, DenseLossL1, DenseLossL2 + per-category metrics
- LR scheduler: LambdaLR (decay_step=21, lr_decay=0.9, lowest_decay=0.02)
- Optimizer: AdamW (lr=0.0005, wd=0.0005)
- Best model selection: based on consider_metric (default CDL1) using Metrics.better_than()
- Loss reporting: × 1000 (与 PoinTr 一致)
- grad_norm_clip: 10

Difference from PoinTr runner:
- Single-GPU only (no distributed)
- Uses pretrained encoder loaded from SSL ckpt
- Grouped LR: encoder_lr = lr * encoder_lr_scale (default 0.1) for fine-tuning encoder
- Logger format simpler than runner.py
"""
import argparse
import os
import sys
import time
import json
import shutil
import datetime
import logging
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from dataset import PCN
import misc

# 你需要确保 extensions/chamfer_dist 已经按 PoinTr README 编译过
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('PCN Completion (PoinTr-aligned)')

    # Model
    parser.add_argument('--model', type=str, default='pt')
    parser.add_argument('--model_prefix', type=str, default='MAE_encoder',
                        choices=['MAE_encoder'])
    parser.add_argument('--ckpts', type=str, default=None,
                        help='SSL pretrained encoder checkpoint')

    # Data
    parser.add_argument('--root', type=str, required=True,
                        help='PCN dataset root directory')
    parser.add_argument('--category_file', type=str, default=None,
                        help='PCN.json path (default: <root>/PCN.json)')
    parser.add_argument('--file_format', type=str, default='pcd',
                        choices=['pcd', 'npy', 'h5'])
    parser.add_argument('--n_points', type=int, default=16384,
                        help='Number of GT points (default 16384, PCN standard)')
    parser.add_argument('--partial_n_points', type=int, default=2048,
                        help='Partial points after RandomSamplePoints')

    # Model config (与 PoinTr.yaml 一致)
    parser.add_argument('--num_pred', type=int, default=14336,
                        help='Number of points predicted by foldingnet (default 14336 = 224*64)')
    parser.add_argument('--num_query', type=int, default=224)
    parser.add_argument('--knn_layer', type=int, default=1)
    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--encoder_depth', type=int, default=12,
                        help='Pretrained encoder depth (12 for your SSL model)')
    parser.add_argument('--decoder_depth', type=int, default=8,
                        help='Decoder depth (8 in PoinTr)')

    # Training
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Total batch size (PoinTr.yaml: total_bs=48)')
    parser.add_argument('--max_epoch', type=int, default=300,
                        help='Max epochs (PoinTr.yaml: 300)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial LR (PoinTr.yaml: 0.0005)')
    parser.add_argument('--encoder_lr_scale', type=float, default=0.1,
                        help='Encoder LR = lr * encoder_lr_scale (for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay (PoinTr.yaml: 0.0005)')
    parser.add_argument('--decay_step', type=int, default=21,
                        help='LR decay step (PoinTr.yaml: 21)')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='LR decay multiplier per decay_step (PoinTr.yaml: 0.9)')
    parser.add_argument('--lowest_decay', type=float, default=0.02,
                        help='Min LR ratio (PoinTr.yaml: 0.02)')
    parser.add_argument('--grad_norm_clip', type=float, default=10.,
                        help='Grad clipping (PoinTr default: 10)')

    # Logging / saving
    parser.add_argument('--log_dir', type=str, default='pcn_completion')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    return parser.parse_args()


# ============================================================
# Validate (严格对齐 runner.py validate)
# ============================================================

def validate(model, test_dataloader, ChamferDisL1, ChamferDisL2, log_fn, epoch=0):
    """
    Validation loop, strictly aligned with PoinTr runner.py validate().
    
    Reports:
    - SparseLossL1, SparseLossL2, DenseLossL1, DenseLossL2 (× 1000)
    - Per-category DenseLossL1 (CD-L1, the main PCN metric)
    - Overall avg
    """
    model.eval()

    # Per-category accumulators
    cat_metrics = {}  # taxonomy_id -> [list of dense_loss_l1 values]
    cat_counts = {}

    # Overall sparse/dense L1/L2
    all_sparse_l1 = []
    all_sparse_l2 = []
    all_dense_l1 = []
    all_dense_l2 = []

    # Category name mapping
    cat_name_map = {
        '02691156': 'airplane', '02933112': 'cabinet',
        '02958343': 'car',      '03001627': 'chair',
        '03636649': 'lamp',     '04256520': 'sofa',
        '04379243': 'table',    '04530566': 'watercraft',
    }

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(
                tqdm(test_dataloader, desc='[Val]', smoothing=0.9)):
            partial = data[0].cuda()
            gt = data[1].cuda()

            ret = model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            # × 1000 (与 PoinTr 一致)
            all_sparse_l1.append(sparse_loss_l1.item() * 1000)
            all_sparse_l2.append(sparse_loss_l2.item() * 1000)
            all_dense_l1.append(dense_loss_l1.item() * 1000)
            all_dense_l2.append(dense_loss_l2.item() * 1000)

            # Per-category dense L1
            for tax_id in taxonomy_ids:
                if tax_id not in cat_metrics:
                    cat_metrics[tax_id] = []
                    cat_counts[tax_id] = 0
                cat_metrics[tax_id].append(dense_loss_l1.item() * 1000)
                cat_counts[tax_id] += 1

    # ============ Summary ============
    log_fn('=' * 70)
    log_fn(f'[Validation Epoch {epoch}]')
    log_fn(f'  Avg SparseLossL1: {np.mean(all_sparse_l1):.4f}  '
           f'Avg SparseLossL2: {np.mean(all_sparse_l2):.4f}')
    log_fn(f'  Avg DenseLossL1:  {np.mean(all_dense_l1):.4f}  '
           f'Avg DenseLossL2:  {np.mean(all_dense_l2):.4f}')

    log_fn('  ============ Per-category CD-L1 (×1000) ============')
    cat_avg_l1 = []
    for tax_id in sorted(cat_metrics.keys()):
        cat_l1 = np.mean(cat_metrics[tax_id])
        cat_avg_l1.append(cat_l1)
        cat_name = cat_name_map.get(tax_id, tax_id)
        log_fn(f'  {tax_id} ({cat_name:12s}, n={cat_counts[tax_id]}): '
               f'CD-L1 = {cat_l1:.4f}')

    avg_cat_l1 = np.mean(cat_avg_l1)
    log_fn(f'  ----------------------------------------------------')
    log_fn(f'  Average across categories (PCN standard): {avg_cat_l1:.4f}')
    log_fn('=' * 70)

    metrics = {
        'cdl1': avg_cat_l1,                      # main metric (consider_metric: CDL1)
        'avg_dense_l1': np.mean(all_dense_l1),   # micro-avg
        'avg_dense_l2': np.mean(all_dense_l2),
        'avg_sparse_l1': np.mean(all_sparse_l1),
        'avg_sparse_l2': np.mean(all_sparse_l2),
        'per_category': {tid: np.mean(vals) for tid, vals in cat_metrics.items()},
    }
    return metrics


# ============================================================
# LR scheduler (与 PoinTr LambdaLR 一致)
# ============================================================

def build_lambda_scheduler(optimizer, decay_step, lr_decay, lowest_decay):
    """
    LambdaLR with continuous decay (与 PoinTr utils/builder.py 一致).
    
    LR multiplier at epoch e:
        max(lr_decay ** (e // decay_step), lowest_decay)
    
    For PCN: decay_step=21, lr_decay=0.9, lowest_decay=0.02
    -> LR drops by 0.9 every 21 epochs, min ratio = 0.02
    """
    def lr_lambda(epoch):
        return max(lr_decay ** (epoch / decay_step), lowest_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Main
# ============================================================

def main(args):
    # ============ Setup ============
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/completion').joinpath(args.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Logger
    logger = logging.getLogger("PCN-Completion")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(str(log_dir / f'{args.model}.txt'))
    fh.setLevel(logging.INFO); fh.setFormatter(formatter)
    logger.addHandler(fh)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string('=' * 70)
    log_string(f'[Args] {args}')
    log_string('=' * 70)

    # Seed
    if args.seed != -1:
        misc.set_random_seed(args.seed, deterministic=args.deterministic)

    # ============ Dataset ============
    train_set = PCN(
        root=args.root,
        category_file=args.category_file,
        subset='train',
        n_points=args.n_points,
        partial_n_points=args.partial_n_points,
        file_format=args.file_format,
    )
    test_set = PCN(
        root=args.root,
        category_file=args.category_file,
        subset='test',
        n_points=args.n_points,
        partial_n_points=args.partial_n_points,
        file_format=args.file_format,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )
    # PoinTr 测试 batch_size 也用 1 (per-sample evaluation),
    # 但为了效率我们用 batch (与 PoinTr runner validate 一致, 它用 args.batch)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    log_string(f'Train: {len(train_set)} | Test: {len(test_set)}')

    # ============ Model ============
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(exp_dir))
    model = MODEL.get_model(
        num_pred=args.num_pred,
        num_query=args.num_query,
        knn_layer=args.knn_layer,
        trans_dim=args.trans_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    log_string(f'# parameters: {n_params:,} ({n_params/1e6:.2f}M)')
    log_string(f'Output: num_query={args.num_query}, fold_step={model.fold_step}, '
               f'num_pred={args.num_pred} (= num_query × fold_step^2)')

    # ============ Test mode ============
    if args.test:
        state_dict = torch.load(args.ckpts, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        ChamferDisL1 = ChamferDistanceL1()
        ChamferDisL2 = ChamferDistanceL2()
        validate(model, test_loader, ChamferDisL1, ChamferDisL2, log_string, epoch=-1)
        return

    # ============ Load pretrained encoder ============
    if args.ckpts is not None:
        # Verify by tracking pos_embed[0].weight before/after
        pre_w = model.pos_embed[0].weight.data.clone()
        model.load_model_from_ckpt(args.ckpts, model_prefix=args.model_prefix)
        post_w = model.pos_embed[0].weight.data
        diff = (pre_w - post_w).abs().sum().item()
        if diff > 1e-6:
            log_string('✅ pos_embed loaded from pretrained checkpoint')
        else:
            log_string('⚠️ pos_embed NOT loaded - still randomly initialized!')

    # ============ Optimizer (grouped LR for encoder/decoder) ============
    # 预训练部分 (encoder) 用小 LR, 新增部分 (decoder/head) 用主 LR
    pretrained_module_names = [
        'group_divider', 'encoder', 'pos_embed', 'blocks', 'norm'
    ]

    encoder_params_decay = []
    encoder_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_pretrained = any(name.startswith(m) for m in pretrained_module_names)
        # weight decay 排除 1D 参数 (LayerNorm, BN, bias)
        no_decay = (param.ndim == 1) or name.endswith('.bias') or 'token' in name

        if is_pretrained:
            if no_decay:
                encoder_params_no_decay.append(param)
            else:
                encoder_params_decay.append(param)
        else:
            if no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)

    encoder_lr = args.lr * args.encoder_lr_scale
    param_groups = [
        {'params': head_params_decay, 'lr': args.lr,
         'weight_decay': args.weight_decay, 'name': 'head_decay'},
        {'params': head_params_no_decay, 'lr': args.lr,
         'weight_decay': 0.0, 'name': 'head_no_decay'},
        {'params': encoder_params_decay, 'lr': encoder_lr,
         'weight_decay': args.weight_decay, 'name': 'encoder_decay'},
        {'params': encoder_params_no_decay, 'lr': encoder_lr,
         'weight_decay': 0.0, 'name': 'encoder_no_decay'},
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    log_string(f'Encoder LR: {encoder_lr:.6f} ({len(encoder_params_decay)+len(encoder_params_no_decay)} params)')
    log_string(f'Head LR:    {args.lr:.6f} ({len(head_params_decay)+len(head_params_no_decay)} params)')

    # ============ LR Scheduler (LambdaLR, 与 PoinTr 一致) ============
    scheduler = build_lambda_scheduler(
        optimizer,
        decay_step=args.decay_step,
        lr_decay=args.lr_decay,
        lowest_decay=args.lowest_decay,
    )

    # ============ Loss functions ============
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    # ============ Resume ============
    start_epoch = 0
    best_metric = float('inf')  # 越小越好 (CD-L1)
    if args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_metric = ckpt.get('best_metric', float('inf'))
        log_string(f'Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}')

    # ============ Initial validation ============
    log_string(f'\n[Init] Running initial validation...')
    init_metrics = validate(model, test_loader, ChamferDisL1, ChamferDisL2, log_string, epoch=-1)

    # ============ Training loop ============
    for epoch in range(start_epoch, args.max_epoch):
        model.train()

        epoch_start_time = time.time()
        sparse_losses = []
        dense_losses = []

        pbar = tqdm(train_loader, desc=f'[Epoch {epoch}/{args.max_epoch}]', smoothing=0.9)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(pbar):
            partial = data[0].cuda(non_blocking=True)
            gt = data[1].cuda(non_blocking=True)

            optimizer.zero_grad()

            # Forward (与 PoinTr runner.py L114 一致)
            ret = model(partial)
            sparse_loss, dense_loss = model.get_loss(ret, gt, epoch)

            # Total loss (与 PoinTr runner.py L118 一致: 1:1 weighting, 不加 weight)
            loss = sparse_loss + dense_loss

            loss.backward()

            # Grad clipping (与 PoinTr runner.py L123 一致)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            args.grad_norm_clip, norm_type=2)
            optimizer.step()

            # Loss × 1000 (与 PoinTr runner.py L131-133 一致)
            sparse_losses.append(sparse_loss.item() * 1000)
            dense_losses.append(dense_loss.item() * 1000)

            if idx % 100 == 0:
                pbar.set_postfix(
                    sparse=f'{sparse_loss.item()*1000:.4f}',
                    dense=f'{dense_loss.item()*1000:.4f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.6f}'
                )

        # Step scheduler at end of epoch (与 PoinTr 一致)
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        log_string(f'\n[Epoch {epoch}] Time: {epoch_time:.1f}s | '
                   f'Avg SparseLoss: {np.mean(sparse_losses):.4f} | '
                   f'Avg DenseLoss: {np.mean(dense_losses):.4f} | '
                   f'LR_head: {optimizer.param_groups[0]["lr"]:.6f} | '
                   f'LR_encoder: {optimizer.param_groups[2]["lr"]:.6f}')

        # ============ Validation ============
        if epoch % args.val_freq == 0:
            metrics = validate(model, test_loader, ChamferDisL1, ChamferDisL2,
                               log_string, epoch=epoch)

            # Save best (basis: CDL1 = avg of per-category CD-L1, 与 PoinTr 一致)
            current_metric = metrics['cdl1']
            if current_metric < best_metric:
                best_metric = current_metric
                save_path = checkpoints_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'metrics': metrics,
                }, str(save_path))
                log_string(f'✓ Saved best model (CDL1={best_metric:.4f}) -> {save_path}')

        # Save last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
        }, str(checkpoints_dir / 'last_model.pth'))

        log_string(f'[Best] CDL1: {best_metric:.4f}')

    log_string('\n' + '=' * 70)
    log_string(f'Training complete. Best CDL1: {best_metric:.4f}')
    log_string('=' * 70)


if __name__ == '__main__':
    args = parse_args()
    main(args)