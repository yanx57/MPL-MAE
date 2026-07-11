#!/bin/bash
# weights_path=$1

# # Check the both arguments are provided
# if [ -z "$weights_path" ] 
# then
#     echo "Please provide the gpu id and the model weights path"
#     echo "Example usage: bash scripts/classification_scanobj_obj_bg.sh /home/yanxu_2023/yanxu_2023/PCP-MAE-main/experiments/base/pretrain/experiment/pcp-pretrain/ckpt-epoch-300.pth"
#     exit 1
# fi

for i in `seq 1 10`
do
    python main.py \
    --config /home/yanxu_2023/yanxu_2023/PCP-MAE-new-pe-dropout/cfgs/linear/finetune_scan_objbg.yaml --finetune_model \
    --exp_name ./ft_objbg-finetune/test_$i\
    --ckpts  /home/yanxu_2023/yanxu_2023/PCP-MAE-new-pe-dropout/experiments/base/pretrain/rank_base_dropout/ckpt-epoch-300.pth --seed $i
done
