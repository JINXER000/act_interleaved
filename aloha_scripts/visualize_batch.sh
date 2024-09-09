for (( i=0; i<50; i++ ))
do
  echo "Starting visualize $i"
  python visualize_episodes.py --dataset_dir /ssd1/aloha_data/cup_random/  --episode_idx $i
done
