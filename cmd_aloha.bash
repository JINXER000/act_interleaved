''' Task info edited in ./aloha/aloha_scripts/constants'''


# train - aloha
python3 imitate_episodes.py \
--task_name aloha_insert_10s \
--ckpt_dir ./ckpt_dir/aloha_insert_10s_t2 \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 5000 --lr 1e-5 --seed 0 \
> log_insert_10s_t2.log 2>&1 &

# visualize

# evaluation - aloha
python3 imitate_episodes.py \
--task_name aloha_insert_10s_t2 \
--ckpt_dir ./ckpt_dir/aloha_insert_10s_t2 \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \


python3 imitate_episodes.py \
--task_name aloha_battery_allcam \
--ckpt_dir ./ckpt_dir/aloha_battery_allcam \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval


python3 imitate_episodes.py \
--task_name aloha_battery \
--ckpt_dir ./ckpt_dir/aloha_battery \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval

python3 imitate_episodes.py \
--task_name aloha_towel \
--ckpt_dir ./ckpt_dir/aloha_towel \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
# --temporal_agg


python3 imitate_episodes.py \
--task_name aloha_ziploc \
--ckpt_dir ./ckpt_dir/aloha_ziploc \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval
# --temporal_agg

python3 imitate_episodes.py \
--task_name aloha_board_ram \
--ckpt_dir ./ckpt_dir/aloha_board_ram \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg

python3 imitate_episodes.py \
--task_name aloha_ram2 \
--ckpt_dir ./ckpt_dir/aloha_ram2 \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg


python3 imitate_episodes.py \
--task_name aloha_screwdriver \
--ckpt_dir ./ckpt_dir/aloha_screwdriver \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg


python3 imitate_episodes.py \
--task_name aloha_conveyor \
--ckpt_dir ./ckpt_dir/aloha_conveyor \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg


python3 imitate_episodes.py \
--task_name aloha_starbucks \
--ckpt_dir ./ckpt_dir/aloha_starbucks \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg


python3 imitate_episodes.py \
--task_name aloha_transfer_tape \
--ckpt_dir ./ckpt_dir/aloha_transfer_tape \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 --seed 0 \
--eval \
--temporal_agg