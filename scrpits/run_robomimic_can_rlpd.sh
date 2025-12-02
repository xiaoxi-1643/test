#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

project_name=qc_online
env_name=can-mh-low_dim
run_group=can_QC-RLPD-AC

# RLPD-AC

for seed in 1 2 3 4 5
do
  python main_online.py \
    --env_name=$env_name \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False 
done


