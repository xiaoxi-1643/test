#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

beta=10.0
temp=1.0

project_name=qc_online
env_name=cube-triple-play-singletask-task4-v0
run_group=cube_t4_QC-MPO-beta$beta-temp$temp

# QC-FQL-MPO
for seed in 1 2 3 4 5
do
  python main.py \
    --offline_steps=0 \
    --start_training=5000 \
    --agent=agents/acmpo.py \
    --agent.beta=$beta \
    --agent.temperature=$temp \
    --sparse=True \
    --env_name=$env_name \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False 
done



