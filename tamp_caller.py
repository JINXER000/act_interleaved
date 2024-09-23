import numpy as np


import numpy as np
import os
import collections
import matplotlib.pyplot as plt

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE

from act_utils import sample_box_pose, sample_insertion_pose,\
     plt_render # robot functions
from sim_env import make_sim_env, BOX_POSE

from eval_act_wrapper import ACT_Evaluator

NORMALIZED_OPEN = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_OPEN)
NORMALIZED_CLOSE = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_CLOSE)
import sys
CUR_FOLDER = os.getcwd()
EXE_FOLDER = '/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha/'
sys.path.append(EXE_FOLDER)
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import load_yaml_param, read_pickle, Pose, Euler, Point #, qpos_to_eepose
from examples.pybullet.utils.pybullet_tools.aloha_primitives import BodyPath, multiply


JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
def visualize_joints(qpos_list, plot_path=None, ylim=None, label_overwrite=None):

    qpos = np.array(qpos_list) # ts, dim
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx])
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()


    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def qpos_16d_to_14d(qpos_16d):
    qpos_14d = qpos_16d.copy()
    qpos_14d = np.delete(qpos_14d, 7)
    qpos_14d = np.delete(qpos_14d, 14)
    return qpos_14d
    
class Tamp_replayer(ACT_Evaluator):
    def __init__(self):
        super().__init__()
        self.ts = self.env.reset()

    def replay_tamp_step(self, qpos):
        self.ts = self.env.step(qpos)
        return self.ts


# def simulate_tamp_14d(seq, init_obj_states, env = None):
#     BOX_POSE[0] = init_obj_states # used in sim reset

#     if env is None:
#         # setup the environment
#         env = make_sim_env('sim_insertion_tamp')
#     ts = env.reset()

#     # setup plotting
#     ax = plt.subplot()
#     plt_img = ax.imshow(ts.observation['images']['angle'])
#     plt.ion()

#     # action_seq_14d = convert_action_seq_14d(env, seq, init_obj_states)
#     # visualize_joints(action_seq_14d, plot_path='/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/qpos.png')
#     # for id, action in enumerate(action_seq_14d):
#     for action in iterate_sequence(seq, init_obj_states):

#         qpos = action[:16]
#         ts = env.step(qpos)

#         plt_img.set_data(ts.observation['images']['angle'])
#         plt.pause(0.02)

#     return ts






def iterate_sequence(seq, init_obj_states= None):
    starting_qpos = qpos_16d_to_14d(START_ARM_POSE)
    action_seq = [starting_qpos]
    # initiually, open the gripper
    action_seq[0][6] = NORMALIZED_OPEN
    action_seq[0][-1] = NORMALIZED_OPEN

    cur_conf_14d = action_seq[-1]
    for i, primitive in enumerate(seq.commands):
        
        # attach or detach
        if not hasattr(primitive, 'path'):
            continue

        if hasattr(primitive, 'refined_qpos'):
            path_to_execute = primitive.refined_qpos
        else:
            path_to_execute = primitive.path

        for cfg in path_to_execute:
            next_conf_14d = cur_conf_14d.copy()
            
            if primitive.group == "left_arm":
                next_conf_14d[0:6] = cfg[0:6]

            elif primitive.group == "right_arm":
                next_conf_14d[7:7+6] = cfg[0:6]

            elif primitive.group == "left_gripper":
                next_conf_14d[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(cfg[0])
            elif primitive.group == "right_gripper":
                next_conf_14d[-1] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(cfg[0])

            cur_conf_14d = next_conf_14d
            yield next_conf_14d





def get_box_poses(obj_info_ls):
    socket_stable_height = peg_stable_height= 0.05
    y_bias = 0.49
    socket_info, peg_info, colObs_info = obj_info_ls
    socket_pose = Pose(Point(x=socket_info.x, y=socket_info.y + y_bias, z = socket_stable_height), euler=Euler( pitch=1.57))
    peg_pose = Pose(Point(x=peg_info.x, y=peg_info.y + y_bias, z = peg_stable_height), euler=Euler(roll=1.57))
    # NOTE: pybullet quat is xyzw, mujoco quat is wxyz
    socket_quat = [socket_pose[1][3], socket_pose[1][0], socket_pose[1][1], socket_pose[1][2]]
    peg_quat = [peg_pose[1][3], peg_pose[1][0], peg_pose[1][1], peg_pose[1][2]]
    
    socket_arr = np.concatenate([socket_pose[0], socket_quat])
    peg_arr = np.concatenate([peg_pose[0], peg_quat])
    init_obj_states = np.concatenate([peg_arr, socket_arr])
    return init_obj_states
    
    
def call_act(act_evaluator, ts):
    act_evaluator.reset_all(reset_grippers = False, at_start=False)
    for i in range(act_evaluator.max_timesteps):
        ts = act_evaluator.inference(ts)



if __name__ == "__main__":
    CFG_PATH = os.path.join(EXE_FOLDER, 'config/aloha_scene.yaml')
    # load param from yaml
    CMD_PATH, obj_info_ls, is_record = load_yaml_param(CFG_PATH)
    
    pkl_path = os.path.join(CMD_PATH, "finegrand_transfer.pkl")
    seq = read_pickle(pkl_path)

    init_obj_states = get_box_poses(obj_info_ls)
    BOX_POSE[0] = init_obj_states # used in sim reset

    os.chdir(CUR_FOLDER)

    act_evaluator = Tamp_replayer()
    ts = act_evaluator.ts

    # ### debug: generate scene point clouds
    # act_evaluator.env.task.generate_scene_point_clouds(act_evaluator.env.physics)

    # setup plotting
    plotter = plt_render(ts, dt=DT, img_type='rgb', cam_name='angle')
    # plotter = plt_render(ts, dt=DT, img_type='depth', cam_name='top')
    # plotter = plt_render(ts, dt=DT, img_type='seg', cam_name='top', env=act_evaluator.env)

    for tamp_id, action in enumerate(iterate_sequence(seq, init_obj_states)):

        qpos = action[:16]
        ts = act_evaluator.replay_tamp_step(qpos)

        plotter.update(ts)

    act_evaluator.reset_all(reset_grippers = False, at_start=False)
    for i in range(tamp_id, act_evaluator.max_timesteps):
        ts = act_evaluator.inference()

        plotter.update(ts)


    plt.ioff()