#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

beta=10.0
temp=0.1

project_name=qc_online
run_group=square_QC-MPO-beta$beta-temp$temp-clip-BCbug

# QC-FQL-MPO
for seed in 1 2 3 4 5
do
  python main.py \
    --offline_steps=0 \
    --start_training=5000 \
    --agent=agents/acmpo.py \
    --agent.beta=$beta \
    --agent.temperature=$temp \
    --env_name=square-mh-low_dim \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False 
done



