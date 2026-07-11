"""
Convert PCN .pcd dataset to .npy for fast loading.

Why convert:
- open3d 在 DataLoader 多 worker 下经常 segfault (已知问题)
- numpy.load 在多进程下完全稳定
- npy 加载速度比 pcd 快 5-10 倍

Usage:
    python convert_pcn_to_npy.py \
        --src /home/yanxu_2023/yanxu_2023/eccv_rebuttal/ShapeNetCompletion \
        --dst /home/yanxu_2023/yanxu_2023/eccv_rebuttal/ShapeNetCompletion_npy \
        --workers 16

Speed: ~30-60 min for full PCN dataset (~265K files) on a fast disk.
"""
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm


def convert_one(args):
    """Convert one .pcd file to .npy. Single-process function."""
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return None  # Already converted, skip
    
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(src_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            return f'EMPTY: {src_path}'
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, pts)
        return None
    except Exception as e:
        return f'ERROR ({type(e).__name__}: {e}): {src_path}'


def collect_tasks(src_root, dst_root, cats):
    """Build list of (src, dst) pairs to convert."""
    tasks = []
    for split in ['train', 'val', 'test']:
        for cat in cats:
            tax_id = cat['taxonomy_id']
            models = cat[split]
            n_views = 8 if split == 'train' else 1
            
            for model_id in models:
                # Complete
                src = os.path.join(src_root, split, 'complete', tax_id, f'{model_id}.pcd')
                dst = os.path.join(dst_root, split, 'complete', tax_id, f'{model_id}.npy')
                if os.path.exists(src):
                    tasks.append((src, dst))
                
                # Partial (multiple views)
                for view in range(n_views):
                    src = os.path.join(src_root, split, 'partial', tax_id,
                                       model_id, f'{view:02d}.pcd')
                    dst = os.path.join(dst_root, split, 'partial', tax_id,
                                       model_id, f'{view:02d}.npy')
                    if os.path.exists(src):
                        tasks.append((src, dst))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='ShapeNetCompletion (pcd) root')
    parser.add_argument('--dst', required=True, help='Output ShapeNetCompletion_npy root')
    parser.add_argument('--workers', type=int, default=16,
                        help='Parallel workers (use ~CPU core count)')
    parser.add_argument('--category_file', type=str, default=None)
    args = parser.parse_args()
    
    cat_file = args.category_file or os.path.join(args.src, 'PCN.json')
    
    print(f'[Source]      {args.src}')
    print(f'[Destination] {args.dst}')
    print(f'[Workers]     {args.workers}')
    print(f'[Categories]  {cat_file}')
    
    # Load categories
    with open(cat_file) as f:
        cats = json.load(f)
    print(f'[Loaded]      {len(cats)} categories')
    
    # Collect all conversion tasks
    print('\n[Step 1] Scanning files...')
    tasks = collect_tasks(args.src, args.dst, cats)
    print(f'[Tasks]       {len(tasks)} files to convert')
    
    # Filter tasks already done
    pending = [t for t in tasks if not os.path.exists(t[1])]
    skipped = len(tasks) - len(pending)
    print(f'[Pending]     {len(pending)} (skipping {skipped} already converted)')
    
    if not pending:
        print('\n✅ All files already converted!')
        # Just copy PCN.json
        dst_json = os.path.join(args.dst, 'PCN.json')
        if not os.path.exists(dst_json):
            os.makedirs(args.dst, exist_ok=True)
            import shutil
            shutil.copy(cat_file, dst_json)
            print(f'[Copy]        PCN.json -> {dst_json}')
        return
    
    # Parallel conversion
    print(f'\n[Step 2] Converting with {args.workers} workers...')
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 用 submit + as_completed 配合 tqdm 看进度
        futures = [executor.submit(convert_one, t) for t in pending]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Converting'):
            result = future.result()
            if result is not None:
                errors.append(result)
    
    # Copy PCN.json
    dst_json = os.path.join(args.dst, 'PCN.json')
    if not os.path.exists(dst_json):
        import shutil
        shutil.copy(cat_file, dst_json)
        print(f'[Copy]        PCN.json -> {dst_json}')
    
    # Report
    print(f'\n[Report]')
    print(f'  Total tasks:     {len(tasks)}')
    print(f'  Skipped:         {skipped}')
    print(f'  Successful:      {len(pending) - len(errors)}')
    print(f'  Errors:          {len(errors)}')
    
    if errors:
        print(f'\n[Errors] First 10:')
        for e in errors[:10]:
            print(f'  {e}')
    
    print(f'\n✅ Conversion complete! Use --root {args.dst} --file_format npy')


if __name__ == '__main__':
    main()