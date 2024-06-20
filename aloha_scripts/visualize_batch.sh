for (( i=40; i<50; i++ ))
do
  echo "Starting visualize $i"
  python3 visualize_episodes.py --dataset_dir ~/Desktop/aloha_data/aloha_ziploc --episode_idx $i
done