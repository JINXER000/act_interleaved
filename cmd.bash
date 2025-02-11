#!/bin/bash

# record trajectory data
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir ./data_save/ --num_episodes 50 --onscreen_render 
python3 record_sim_episodes.py --task_name sim_insertion_scripted --dataset_dir ./data_save/ --num_episodes 50 --onscreen_render 

python3 record_sim_episodes.py \
    --task_name sim_insertion_scripted \
    --dataset_dir ./dataset/sim_insertion_scripted/ \
    --num_episodes 50

python3 record_sim_episodes.py \
    --task_name sim_insertion_tamp \
    --dataset_dir ~/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_tamp/ \
    --num_episodes 50 --onscreen_render


# train 
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ./ckpt_dir/transfer_cube_scripted_sim --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

python3 imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
python3 imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir ./ckpt/sim_insertion_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 
python3 imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 

# train - aloha
python3 imitate_episodes.py --task_name aloha_insert_10s --ckpt_dir ./ckpt_dir/aloha_insert_10s --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

python3 imitate_episodes.py --task_name aloha_transfer_tape --ckpt_dir ./ckpt_dir/aloha_transfer_tape --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

python3 imitate_episodes.py --task_name sim_insertion_tamp --ckpt_dir ./ckpt/sim_insertion_tamp --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0  

python3 imitate_episodes.py --task_name cup_random --ckpt_dir /ssd1/chenyizhou/act_ckpts --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 8000 --lr 1e-5 --seed 0  

# evaluation
python3 imitate_episodes_sim.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir ./ckpt_dir/sim_transfer_cube_scripted \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0 --eval 
# --onscreen_render \
# --temporal_agg


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


