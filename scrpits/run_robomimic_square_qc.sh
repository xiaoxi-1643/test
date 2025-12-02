#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# QC-FQL
for seed in 0 1 2 3 4
do
  python main.py \
    --agent=agents/acfql.py \
    --agent.actor_type=distill-ddpg \
    --agent.alpha=10000 \
    --env_name=square-mh-low_dim \
    --horizon_length=5 \
    --project_name=qc_test \
    --run_group=square_QC-FQL \
    --exp_name=seed-$seed \
    --seed=$seed 
done

# QC
for seed in 0 1 2 3 4
do
  python main.py \
    --agent=agents/acfql.py \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=16 \
    --env_name=square-mh-low_dim \
    --horizon_length=5 \
    --project_name=qc_test \
    --run_group=square_QC \
    --exp_name=seed-$seed \
    --seed=$seed 
done

