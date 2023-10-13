#!/bin/bash

# record trajectory data
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir ./data_save/ --num_episodes 50 --onscreen_render 
python3 record_sim_episodes.py --task_name sim_insertion_scripted --dataset_dir ./data_save/ --num_episodes 50 --onscreen_render 


# train 
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ./ckpt/sim_transfer_cube_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 
python3 imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 
python3 imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir ./ckpt/sim_insertion_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 
python3 imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 


# evaluation
python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir ./ckpt/sim_transfer_cube_scripted \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0 --eval --onscreen_render \
--temporal_agg


python3 imitate_episodes.py \
--task_name sim_transfer_cube_human \
--ckpt_dir ./ckpt/sim_transfer_cube_human \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0 --eval --onscreen_render \
--temporal_agg


python3 imitate_episodes.py \
--task_name sim_insertion_scripted \
--ckpt_dir ./ckpt/sim_insertion_scripted \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0 --eval --onscreen_render \
--temporal_agg


python3 imitate_episodes.py \
--task_name sim_insertion_human \
--ckpt_dir ./ckpt/sim_insertion_human \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0 --eval --onscreen_render \
--temporal_agg


