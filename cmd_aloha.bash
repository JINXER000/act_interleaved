
# train - aloha
python3 imitate_episodes.py --task_name aloha_insert_10s --ckpt_dir ./ckpt_dir/aloha_insert_10s_t2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 5000 --lr 1e-5 --seed 0 > log_insert_10s_t2.log 2>&1 &