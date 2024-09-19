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

from act_utils import load_data # data functions
from act_utils import sample_box_pose, sample_insertion_pose # robot functions
from act_utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed


import os
import sys
# # sys.path.append("/home/xuhang/interbotix_ws/src/aloha/")
# sys.path.append("/home/xuhang/Desktop/xh-codes/ACT/aloha/")

from aloha.aloha_scripts import constants

import torch



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



def limit_step_diff(target_qpos, qpos, max_diff = 0.06):
    
    cur_diff = target_qpos - qpos
    rectified_diff = np.clip(cur_diff, -max_diff, max_diff)
    rectified_target_qpos = qpos + rectified_diff
    # gripper remain fixed
    rectified_target_qpos[6] = target_qpos[6] 
    rectified_target_qpos[13] = target_qpos[13]
    return rectified_target_qpos



class ACT_Evaluator(object):
    def __init__(self, with_planning = False):
        self.with_planning = with_planning
        arg_dict = self.get_default_args()
        self.initialize(arg_dict)

    def get_default_args(self):
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


        return vars(parser.parse_args())
    
    def initialize(self, args):
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

        ##NOTE: drop all name containning 'depth' in camera_names
        camera_names = [name for name in camera_names if 'depth' not in name]

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

        self.config = {
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

        ckpt_name = f'policy_best.ckpt'


        # start eval
        set_seed(1000)
        ckpt_dir = self.config['ckpt_dir']
        self.state_dim = self.config['state_dim']
        real_robot = self.config['real_robot']
        policy_class = self.config['policy_class']
        onscreen_render = self.config['onscreen_render']
        policy_config = self.config['policy_config']
        self.camera_names = self.config['camera_names']
        self.max_timesteps = self.config['episode_len']
        task_name = self.config['task_name']
        temporal_agg = self.config['temporal_agg']
        onscreen_cam = 'angle'

        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = make_policy(policy_class, policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # load environment
        if real_robot:

            from aloha.aloha_scripts.robot_utils import move_grippers # requires aloha
            from aloha.aloha_scripts.real_env import make_real_env # requires aloha

            self.env = make_real_env(init_node=True, setup_robots= not self.with_planning)
            env_max_reward = 0
        else:
            from sim_env import make_sim_env
            self.env = make_sim_env(task_name)
            env_max_reward = self.env.task.max_reward

        self.query_frequency = policy_config['num_queries']
        if temporal_agg:
            self.query_frequency = 1


        self.num_queries = policy_config['num_queries']


        self.reset_all()



    def inference(self):
        with torch.inference_mode():    
        #     
            ### process previous timestep to get qpos and image_list
            obs = self.ts.observation
            qpos_numpy = np.array(obs['qpos'])
            qpos = self.pre_process(qpos_numpy)

            t0 = time.perf_counter()
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_image = get_image(self.ts, self.camera_names)

            ### query policy
            if self.config['policy_class'] == "ACT":
                if self.t % self.query_frequency == 0:
                    self.all_actions = self.policy(qpos, curr_image)
                if self.config['temporal_agg']:
                    self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[:, self.t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = self.all_actions[:, self.t % self.query_frequency]
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            
            action = self.post_process(raw_action)
            target_qpos = action

            ### limit the step difference to 0.06
            target_qpos = limit_step_diff(target_qpos, qpos_numpy, max_diff = 0.09)

            ### step the environment
            self.ts = self.env.step(target_qpos)

            self.t += 1

    def reset_all(self, reset_grippers = True):
        self.ts = self.env.reset(fake=self.with_planning)
        if reset_grippers:
            self.env.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.env.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        ### evaluation loop
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()

        self.t = 0


if __name__ == '__main__':
    evaluator = ACT_Evaluator(with_planning=False)
    for i in range (evaluator.max_timesteps):
        evaluator.inference()
        print(f'Step {i} done')

    evaluator.reset_all()


  
