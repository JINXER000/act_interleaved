import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from constants import DT, START_ARM_POSE, TASK_CONFIGS, MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS

from record_episodes import get_auto_index, print_dt_diagnosis,\
    debug
import sys
sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import estimation_policy
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_world_0obj
from examples.pybullet.utils.pybullet_tools.utils import CLIENT, connect
from examples.pybullet.aloha_real.scripts.ros_openworld_base import openworld_base
from examples.pybullet.aloha_real.scripts.constants import qpos_to_eetrans, RBT_ID

import IPython
e = IPython.embed

import open3d as o3d
import os

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
    
    # # save eepose_R  and eetrans_t into txt
    # ee_file = os.path.join(save_dir, 'graspPose_{}.npz'.format(episode_idx))
    # np.savez(ee_file, axes = eepose_R, seg_center = eetrans_t, obj_point = tgt_obj_point)

    # # load ply from data dir
    # file_name = os.path.join(save_dir, 'scene.ply')
    # scene_pc = o3d.io.read_point_cloud(file_name)
    # xyz = np.asarray(scene_pc.points)
    # rgb = np.asarray(scene_pc.colors)

    # npz_dict = {'seg_center': eetrans_t, 'axes': eepose_R, 'xyz': xyz, 'rgb': rgb}
    # npz_file = os.path.join(save_dir, 'riemann_scene{}.npz'.format(episode_idx))
    # np.savez(npz_file, **npz_dict)

    # # save another version of npz file
    # soleobj_filename = os.path.join(save_dir, 'graspobj_{}.ply'.format(episode_idx))
    # obj_pc = o3d.io.read_point_cloud(soleobj_filename)
    # xyz = np.asarray(obj_pc.points)
    # rgb = np.asarray(obj_pc.colors)

    # npz_dict = {'seg_center': eetrans_t, 'axes': eepose_R, \
    #             'xyz': xyz, 'rgb': rgb, 'tgt_obj_point': tgt_obj_point}
    # npz_file = os.path.join(save_dir, 'riemannobj_{}.npz'.format(episode_idx))
    # np.savez(npz_file, **npz_dict)
    return True



def capture_pointcloud(task_name= 'aloha_transfer_tape', episode_idx = 101, sensor_dir = None, save_dir = None,**kwargs):
    if not os.path.isdir(sensor_dir):
        os.makedirs(sensor_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    acting_base = openworld_base({}, only_perception = True)
    acting_base.obtain_sensor_data(sensor_dir)

    # do segmentation 
    connect(use_gui=True)
    robot_body, names, movable_bodies, stackable_bodies = load_world_0obj()
    estimator = estimation_policy(robot_body, mode = 'data_process', img_src = 'real', \
                                  file_path=sensor_dir,  teleport=False, client=CLIENT)
    # belief = estimator.estimate_state(graspdata_dir = save_dir, episode_idx = episode_idx)
    belief = estimator.estimate_state()
    tgt_obj = belief.estimated_objects[0]
    return tgt_obj

def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, \
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

    percept_arm_pose = [0, -1.6015, 0.5727, 0.0838,1.7418,0]
    move_arms([puppet_bot_left, puppet_bot_right], [percept_arm_pose] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)

    tgt_obj = capture_pointcloud(**kwargs)
    print('recorded desk and obj point cloud')

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

    return tgt_obj

def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,\
                        only_npz  = False, save_dir = None, episode_idx = 101, **kwargs):
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

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    tgt_obj = opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, \
                               save_dir = save_dir, episode_idx = 101,**kwargs)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    data_cnt = 0
    ee_trans_ls = []
    ee_rot_ls = []
    obj_point_ls = []
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time() #
        ts = env.step(action)
        t2 = time.time() #
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

        # extract the gripper value
        if data_cnt < 20:
            npz_saved= record_grasp_pose(robot_id=1,  jval_14d=action, \
                            tgt_obj_point= tgt_obj.observed_pose[0], \
                                ee_trans_ls = ee_trans_ls, ee_rot_ls = ee_rot_ls, \
                                    obj_point_ls = obj_point_ls, **kwargs)
            if npz_saved:
                data_cnt += 1
                print('saving data:', data_cnt)


    rgb_points = [lp.color for lp in tgt_obj.points]
    xyz_points = [lp.point for lp in tgt_obj.points]
    # save the point cloud for debug
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_points)
    file_name = os.path.join(save_dir, 'graspobj_{}.ply'.format(episode_idx))
    o3d.io.write_point_cloud(file_name, pcd)
    # save demo file
    demo_file = os.path.join(save_dir, 'graspDemo_{}.npz'.format(episode_idx))
    np.savez(demo_file, xyz = np.asarray(xyz_points), rgb = np.asarray(rgb_points),  axes = np.asarray(ee_rot_ls), \
             seg_center = np.asarray(ee_trans_ls), obj_point = np.asarray(obj_point_ls))

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    if only_npz:
        return True
    
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

        for name, array in data_dict.items():
            root[name][...] = array
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='aloha_transfer_tape', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=101, required=False)
    main(vars(parser.parse_args()))
    # debug()


