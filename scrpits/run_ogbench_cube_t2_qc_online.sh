#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

project_name=qc_online
env_name=cube-triple-play-singletask-task2-v0
run_group=cube_t2_QC

# QC
for seed in 1 2 3 4 5
do
  python main.py \
    --retain_offline_data=True \
    --offline_steps=0 \
    --start_training=5000 \
    --agent=agents/acfql.py \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=$env_name \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False
done

# QC-FQL
# for seed in 0 1 2 3 4
# do
#   python main.py \
#     --offline_steps=0 \
#     --agent=agents/acfql.py \
#     --agent.actor_type=distill-ddpg \
#     --agent.alpha=10000 \
#     --env_name=$env_name \
#     --horizon_length=5 \
#     --project_name=$project_name \
#     --run_group=square_QC-FQL \
#     --exp_name=seed-$seed \
#     --seed=$seed 
# done
