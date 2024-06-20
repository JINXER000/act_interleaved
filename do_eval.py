'''
python3 imitate_episodes.py
--task_name aloha_insert_10s
--ckpt_dir ./ckpt_dir/aloha_insert_10s_t2
--policy_class ACT --kl_weight 10 --chunk_size 100
--hidden_dim 512 --batch_size 8 --dim_feedforward 3200
--num_epochs 5000 --lr 1e-5 --seed 0
> log_insert_10s_t2.log 2>&1 &

'''

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

import ipdb

import os
import sys
# # sys.path.append("/home/xuhang/interbotix_ws/src/aloha/")
# sys.path.append("/home/xuhang/Desktop/xh-codes/ACT/aloha/")

from aloha.aloha_scripts import constants

import torch
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


def main(args):

    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']



    # get task parameters
    is_sim = task_name[:4] == 'sim_' ## key point
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        #import constants.TASK_CONFIGS as TASK_CONFIGS
        task_config = constants.TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        # ckpt_names = [f'policy_best.ckpt']
        ckpt_names = [f'policy_last.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()



def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy



def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:

        from aloha.aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha.aloha_scripts.real_env import make_real_env # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        inference_time_list = []

        ep_t0 = time.perf_counter()
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)

                t0 = time.perf_counter()
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                t1 = time.perf_counter()

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                inference_time_list.append(t1 - t0)
                print(f'Step {t} inference time: {t1 - t0} s')

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass
        ep_t1 = time.perf_counter()
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        print(f'Avg inference time: {np.mean(inference_time_list[1:]):.4f} +/- {np.std(inference_time_list[1:]):.4f} s')
        print(f'Episode time: {ep_t1 - ep_t0:.4f} s')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    # parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='/home/xuhang/interbotix_ws/src/ACT/ckpt_dir/aloha_transfer_tape')
    parser.add_argument('--policy_class',  type=str, default='ACT')
    parser.add_argument('--task_name', type=str, help='task_name', default='aloha_transfer_tape')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=8)
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--num_epochs', type=int, help='num_epochs', default=2000)
    parser.add_argument('--lr', type=float, help='lr', default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', type=int, help='KL Weight', default=10)
    parser.add_argument('--chunk_size', type=int, help='chunk_size', default=100)
    parser.add_argument('--hidden_dim', type=int, help='hidden_dim', default=512)
    parser.add_argument('--dim_feedforward', type=int, help='dim_feedforward', default=3200)
    parser.add_argument('--temporal_agg', action='store_false')    


    main(vars(parser.parse_args()))

    # param_dict = {}
    # param_dict['eval'] = True
    # param_dict['onscreen_render'] = False
    # param_dict['ckpt_dir'] = './ckpt_dir/aloha_transfer_tape'
    # param_dict['policy_class'] = 'ACT'
    # param_dict['task_name'] = 'aloha_transfer_tape'
    # param_dict['batch_size'] = 8
    # param_dict['seed'] = 0
    # param_dict['num_epochs'] = 2000
    # param_dict['lr'] = 1e-5
    # param_dict['kl_weight'] = 10
    # param_dict['chunk_size'] = 100
    # param_dict['hidden_dim'] = 512
    # param_dict['dim_feedforward'] = 3200
    # param_dict['temporal_agg'] = True

    # main(param_dict)

