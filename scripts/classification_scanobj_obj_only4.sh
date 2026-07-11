#!/bin/bash
# weights_path=$1

# # Check the both arguments are provided
# if [ -z "$weights_path" ] 
# then
#     echo "Please provide the gpu id and the model weights path"
#     echo "Example usage: bash scripts/classification_scanobj_obj_bg.sh /home/yanxu_2023/yanxu_2023/PCP-MAE-main/experiments/base/pretrain/experiment/pcp-pretrain/ckpt-epoch-300.pth"
#     exit 1
# fi

for i in `seq 12 16`
do
    python main.py \
    --config cfgs/finetune_scan_objonly.yaml --finetune_model \
    --exp_name ./ft_objonly-finetune/mod_wd_test_$i\
    --ckpts  /home/yanxu_2023/yanxu_2023/PCP-MAE-new-pe-learnable-mlp-dropout_add_2_loss-gate-sigmoid1/experiments/base/pretrain/pretrain/ckpt-epoch-300.pth --seed $i
done
