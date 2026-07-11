#!/bin/bash
# weights_path=$1

# # Check the both arguments are provided
# if [ -z "$weights_path" ] 
# then
#     echo "Please provide the gpu id and the model weights path"
#     echo "Example usage: bash scripts/classification_scanobj_obj_bg.sh /home/yanxu_2023/yanxu_2023/PCP-MAE-main/experiments/base/pretrain/experiment/pcp-pretrain/ckpt-epoch-300.pth"
#     exit 1
# fi

for i in `seq 0 0`
do
    python main.py \
    --config /irip/yanxu_2023/PCP-MAE-new-pe-learnable-mlp-dropout_add_2_loss-gate-gumble-softmax/experiments/aug_test_hlr2e-04_hwd0p05_dpr0p2_elr2e-05_ewd0p05_seed1/generated/ft_objbg-finetune/aug_test_hlr2e-04_hwd0p05_dpr0p2_elr2e-05_ewd0p05_seed1/config.yaml --finetune_model \
    --exp_name ./ft_objbg-finetune/aug_test_$i\
    --ckpts  /irip/yanxu_2023/PCP-MAE-new-pe-learnable-mlp-dropout_add_2_loss-gate-gumble-softmax/experiments/base/pretrain/pretrain/ckpt-epoch-300.pth --seed $i
done
