import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from constants import DT, START_ARM_POSE, TASK_CONFIGS, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS



from record_episodes import get_auto_index, print_dt_diagnosis,\
    debug
import sys
sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')

from examples.pybullet.aloha_real.scripts.ros_openworld_base import openworld_base
from examples.pybullet.aloha_real.scripts.constants import qpos_to_eetrans, RBT_ID

import IPython
e = IPython.embed


import open3d as o3d
import os
import cv2
import json

GRIPPER_THRETH = 0.03

def record_grasp_pose(robot_id, jval_14d = [], tgt_obj_point = None,  \
                      ee_trans_ls = [], ee_rot_ls = [], obj_point_ls = [], **kwargs):
    if RBT_ID[robot_id] == 'left':
        joint_vals = jval_14d[:6]
        gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(jval_14d[6])
    else:
        joint_vals = jval_14d[7:13]
        gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(jval_14d[13])
    
    if gripper_val >= GRIPPER_THRETH:
        return False
    # compute the eepose
    eetrans = qpos_to_eetrans(joint_vals, robot_id)
    eetrans_t = eetrans[:3, 3].reshape(3)
    eepose_R =  eetrans[:3, :3].reshape(9)

    assert tgt_obj_point is not None
    grasp_dist = np.linalg.norm(eetrans_t - tgt_obj_point)
    if grasp_dist > 0.1:
        return False
    
    ee_trans_ls.append(eetrans_t)
    ee_rot_ls.append(eepose_R)
    obj_point_ls.append(tgt_obj_point)
    
    return True


# def capture_pointcloud(task_name= 'aloha_transfer_tape', episode_idx = 101, sensor_dir = None, save_dir = None, initilized = False, **kwargs):

def initialize_bots(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, \
                      **kwargs):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)



def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, \
                      **kwargs):

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)



    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')


def sense_tabletop(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right,\
                  save_dir = None, sensor_dir = None, estimator = None, **kwargs):
        # before demo, do perception
    percept_arm_pose = [0, -1.6015, 0.5727, 0.0838,1.7418,0]
    move_arms([puppet_bot_left, puppet_bot_right], [percept_arm_pose] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)

    if not os.path.isdir(sensor_dir):
        os.makedirs(sensor_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    acting_base = openworld_base({}, only_perception = True)
    acting_base.obtain_sensor_data(sensor_dir)

    # read sensor data
    depth_file = os.path.join(sensor_dir, 'depth_image.png')
    color_file = os.path.join(sensor_dir, 'color_image.png')
    camera_intrinsics_file = os.path.join(sensor_dir, 'color_info.json')

    color_img = cv2.imread(color_file)
    depth_img = cv2.imread(depth_file)
    depth_img_mm = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    depth_img = depth_img_mm.astype(np.float32) / 1000.0
    with open(camera_intrinsics_file, 'r') as f:
        camera_info_color = json.load(f)

    return color_img, depth_img, camera_info_color



    # if estimator is None:
    #     connect(use_gui=True)
    #     robot_body, names, movable_bodies, stackable_bodies = load_world_0obj()
    #     estimator = estimation_policy(robot_body, img_src = 'real', \
    #                                 file_path=sensor_dir,  teleport=False, client=CLIENT)
        
    #     # belief = estimator.estimate_state(graspdata_dir = save_dir, episode_idx = episode_idx)
    # belief = estimator.estimate_state()
    # tgt_obj = belief.estimated_objects[0]

    # return tgt_obj, estimator



def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,\
                         **kwargs):
    print(f'Dataset name: {dataset_name}')

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    initialize_bots(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, **kwargs)
   
    # do perception, record the point cloud and the mesh
    start_color_img, start_depth_img, camera_info = sense_tabletop(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, **kwargs)
    

    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []


    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time() #
        ts = env.step(action)
        t2 = time.time() #
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])


    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    
    # Open puppet grippers
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    # do perception again for final grasp detection
    end_color_img, end_depth_img, _ = sense_tabletop(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, **kwargs)



    # # postprocess recorded values to obtain the contact switch
    # obj_poses = [start_obj.observed_pose[0], end_obj.observed_pose[0]]
    # l_grasp_ids, l_release_ids, r_grasp_ids, r_release_ids = actions2grasps(np.array(actions), obj_poses, plot = True)


    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/color_img': [],
        '/depth_img': [],
        # '/camera_K': []
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    data_dict['/color_img'] = [start_color_img, end_color_img]
    data_dict['/depth_img'] = [start_depth_img, end_depth_img]
    # data_dict['/camera_info'] = camera_info
    # data_dict['/camera_K'] = camera_info['K']

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))

        # TODO: below error. Try only save depth images and camera intrinsicss into hdf5. 
        # objects = root.create_group('objects')
        # _ = objects.create_dataset('start_obj', data = start_obj)
        # _ = objects.create_dataset('end_obj', data = end_obj)

        _ = root.create_dataset('color_img', (2, 480, 640, 3), dtype='uint8')
        _ = root.create_dataset('depth_img', (2, 480, 640), dtype='float32')
        # _ = root.create_dataset('camera_K', (1, 9), dtype='float32')

        for name, array in data_dict.items():
            root[name][...] = array

        _ = root.create_dataset('camera_info', data = json.dumps(camera_info))

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')

    sensor_dir = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/realrobot'
    save_dir = '/home/xuhang/interbotix_ws/src/ACT/aloha/depth_data/'+ args['task_name'] 
    
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,\
                                         task_name = args['task_name'], episode_idx = args['episode_idx'],
                                         sensor_dir = sensor_dir, save_dir = save_dir, only_npz=True)
        if is_healthy:
            break



def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='aloha_transfer_tape', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=99, required=False)
    main(vars(parser.parse_args()))
    # debug()


