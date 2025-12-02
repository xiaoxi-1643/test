#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

project_name=qc_online
env_name=puzzle-3x3-play-singletask-task4-v0
run_group=puzzle4_QC-FQL

# QC-FQL
for seed in 0 1 2 3 4
do
  python main.py \
    --retain_offline_data=True \
    --offline_steps=0 \
    --agent=agents/acfql.py \
    --agent.actor_type=distill-ddpg \
    --agent.alpha=10000 \
    --env_name=$env_name \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False
done
