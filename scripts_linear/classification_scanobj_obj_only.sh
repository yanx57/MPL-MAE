#!/bin/bash
weights_path=$1

# Check the both arguments are provided
if [ -z "$weights_path" ] 
then
    echo "Please provide the gpu id and the model weights path"
    echo "Example usage: bash scripts/classification_scanobj_obj_bg.sh /home/yanxu_2023/yanxu_2023/PCP-MAE-main/experiments/base/pretrain/experiment/pcp-pretrain/ckpt-epoch-300.pth"
    exit 1
fi

for i in `seq 1 10`
do
    python main.py \
    --config cfgs/finetune_scan_objonly.yaml --finetune_model \
    --exp_name /home/yanxu_2023/yanxu_2023/PCP-MAE-main/experiments/ft_objonly-official-300/test_$i \
    --ckpts $weights_path --seed $RANDOM
done
