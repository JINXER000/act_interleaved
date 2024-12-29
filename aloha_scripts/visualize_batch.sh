for (( i=0; i<35; i++ ))
do
  echo "Starting visualize $i"
  python visualize_episodes.py --dataset_dir /ssd1/aloha_data/aloha_transfer_tape/transfer_cup/  --episode_idx $i
done
