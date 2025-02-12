import numpy as np


import numpy as np
import os
import collections
import matplotlib.pyplot as plt

from constants import DT, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE
from constants import MJ2BULLET_OFFSET

from act_utils import sample_box_pose, sample_insertion_pose,\
     plt_render # robot functions
from sim_env import make_sim_env, BOX_POSE

from eval_act_wrapper import ACT_Evaluator, qpos_16d_to_14d

import sys
CUR_FOLDER = os.getcwd()
EXE_FOLDER = '/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha/'
sys.path.append(EXE_FOLDER)
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import load_insertion_param, read_pickle, Pose, Euler, Point #, qpos_to_eepose


NORMALIZED_OPEN = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_OPEN)
NORMALIZED_CLOSE = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_CLOSE)


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



    

## input obj pose in the pyb frame, should transform to mujoco frame
def get_box_poses(obj_info_ls):
    socket_stable_height = peg_stable_height= 0.05
    y_bias = 0.0
    socket_info, peg_info, colObs_info = obj_info_ls
    # socket_pose = Pose(Point(x=socket_info.x, y=socket_info.y + y_bias, z = socket_stable_height), euler=Euler( pitch=1.57))
     # peg_pose = Pose(Point(x=peg_info.x, y=peg_info.y + y_bias, z = peg_stable_height), euler=Euler(roll=1.57))
    # in mujoco, the origin is not the center of the table 
    socket_xyz = Point(x=socket_info.x, y=socket_info.y + y_bias, z = socket_stable_height) - MJ2BULLET_OFFSET
    socket_pose = Pose(socket_xyz, euler=Euler(yaw=socket_info.yaw))
    peg_xyz = Point(x=peg_info.x, y=peg_info.y + y_bias, z = peg_stable_height) - MJ2BULLET_OFFSET
    peg_pose = Pose(peg_xyz, euler=Euler(yaw=peg_info.yaw))

    
    # NOTE: pybullet quat is xyzw, mujoco quat is wxyz
    socket_quat = [socket_pose[1][3], socket_pose[1][0], socket_pose[1][1], socket_pose[1][2]]
    peg_quat = [peg_pose[1][3], peg_pose[1][0], peg_pose[1][1], peg_pose[1][2]]
    
    socket_arr = np.concatenate([socket_pose[0], socket_quat])
    peg_arr = np.concatenate([peg_pose[0], peg_quat])
    init_obj_states = np.concatenate([peg_arr, socket_arr])
    return init_obj_states
    


## input pc in the mujoco frame, should transform to pybullet frame
def save_mj_obsevation(ts, task_name, npz_path = None, has_col =False):
    pc_dict = {}
    if task_name == 'sim_insertion_tamp':
        peg_pc = ts.observation['peg_pc']['top'] + MJ2BULLET_OFFSET # + np.array([0, -0.01, 0.0])
        socket_pc = ts.observation['socket_pc']['top'] + MJ2BULLET_OFFSET # + np.array([0, 0.05, 0.0])

        pc_dict.update({'peg': peg_pc, 'socket': socket_pc})

        if has_col:
            colObs_pc = ts.observation['highcol_pc']['top'] + MJ2BULLET_OFFSET
            pc_dict.update({'colObs': colObs_pc})
    else:
        raise NotImplementedError("This task is not supported")
    
    if npz_path is not None:
        np.savez(npz_path, **pc_dict)
    return pc_dict





def iterate_sequence(seq):
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
                # next_conf_14d[-1] = NORMALIZED_CLOSE if cfg[0] < 0.5 else NORMALIZED_OPEN

            cur_conf_14d = next_conf_14d
            yield next_conf_14d


class Tamp_replayer(ACT_Evaluator):
    def __init__(self, task_name, init_obj_states_arr = None, use_viewer = True):
        super().__init__(task_name=task_name, init_obj_states_arr = init_obj_states_arr)
        self.ts = self.env.reset()

        if use_viewer:
            import mujoco
            import mujoco.viewer
            model = self.env.physics.model.ptr
            data = self.env.physics.data.ptr
            self.viewer = mujoco.viewer.launch_passive(model, data)
        else:
            self.viewer = None

    def replay_tamp_step(self, qpos):
        self.ts = self.env.step(qpos)

        if self.viewer is not None:
            self.viewer.sync()
        return self.ts
    
    def inference(self):
        ts = super().inference()
        if self.viewer is not None:
            self.viewer.sync()

        return ts


if __name__ == "__main__":

    CFG_PATH = os.path.join(EXE_FOLDER, 'config/aloha_scene.yaml')
    # load param from yaml
    CMD_PATH, obj_info_ls, is_record = load_insertion_param(CFG_PATH)
    
    pkl_path = os.path.join(CMD_PATH, "finegrand_transfer.pkl")
    seq = read_pickle(pkl_path)

    init_obj_states = get_box_poses(obj_info_ls)
    BOX_POSE[0] = init_obj_states # used in sim reset

    os.chdir(CUR_FOLDER)

    act_evaluator = Tamp_replayer(task_name='sim_insertion_tamp')
    ts = act_evaluator.ts

    # ### debug: generate scene point clouds
    # act_evaluator.env.task.generate_scene_point_clouds(act_evaluator.env.physics)

    # setup plotting
    plotter = plt_render(ts, dt=DT, img_type='rgb', cam_name='angle')
    # plotter = plt_render(ts, dt=DT, img_type='depth', cam_name='top')
    # plotter = plt_render(ts, dt=DT, img_type='seg', cam_name='top', env=act_evaluator.env)

    for tamp_id, action in enumerate(iterate_sequence(seq)):

        # qpos = action[:16]
        ts = act_evaluator.replay_tamp_step(action)

        plotter.update(ts)

    act_evaluator.reset_all(reset_grippers = False, at_start=False)
    for i in range(tamp_id, act_evaluator.max_timesteps):
        ts = act_evaluator.inference()

        plotter.update(ts)


    plt.ioff()