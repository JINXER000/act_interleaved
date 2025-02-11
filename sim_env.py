import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.mujoco.wrapper.mjbindings import mjlib

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN


from act_utils import sample_box_pose, sample_insertion_pose, \
    get_geom_ids, rgbd_to_pointcloud, resize_point_cloud
import math
import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside
INIT_ARM_POSE = [None] # to be changed from outside

def make_sim_env(task_name, init_obj_states_arr = None):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_insertion_tamp' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TAMPInsertionTask(random=False, init_obj_states_arr=init_obj_states_arr)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)

    elif 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)

    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='back')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            if BOX_POSE[0] is  None:
                BOX_POSE[0] = sample_box_pose()

            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward



class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            if BOX_POSE[0] is  None:
                BOX_POSE[0] = np.concatenate(sample_insertion_pose(has_col = True))

            # physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
            physics.named.data.qpos[-7*3:] = BOX_POSE[0] # last is highcol
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward



class TAMPInsertionTask(InsertionTask):
    def __init__(self, random=None, init_obj_states_arr = None):
        super().__init__(random=random)
        self.obs_idx = 0
        self.recorded_pc = False
        self.init_obj_states_arr = init_obj_states_arr

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)

            if BOX_POSE[0] is  None:
                    if self.init_obj_states_arr is not None:
                        BOX_POSE[0] = self.init_obj_states_arr
                    else:
                        BOX_POSE[0] = np.concatenate(sample_insertion_pose(has_col = True))

            print('initial obj poses:', BOX_POSE[0])

            # physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            physics.named.data.qpos[-7*3:] = BOX_POSE[0] # last is highcol
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    # #https://github.com/google-deepmind/mujoco/issues/1863
    # def generate_scene_point_clouds(self, physics):
    #     # Simulate for 10 seconds and capture RGB-D images at fps Hz.
    #     xyzrgbs: list[np.ndarray] = []
    #     model = physics.model.ptr
    #     data = physics.data.ptr
    #     # mujoco.mj_forward(model, data)
    #     physics.forward()
    #     # mujoco.mj_resetData(model, data)
    #     physics.reset()
    #     while data.time < 10:
    #         # mujoco.mj_step(model, data)
    #         physics.step()
    #         rgb = physics.render(height=480, width=640, camera_id='top')
    #         depth = physics.render(height=480, width=640, camera_id='top', depth=True)
    #         cam_intrinsics = self.get_cam_intrinsics(physics, 'top')
    #         cam_extrinsics = self.get_cam_extrinsics(physics, 'top')
    #         xyzrgb = rgbd_to_pointcloud(rgb, depth, cam_intrinsics, cam_extrinsics)
    #         xyzrgbs.append(xyzrgb)

    #     # Visualize in open3d.
    #     import open3d as o3d
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(xyzrgbs[0][:, :3])
    #     pcd.colors = o3d.utility.Vector3dVector(xyzrgbs[0][:, 3:])
    #     vis.add_geometry(pcd)
    #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    #     vis.add_geometry(frame)

    #     counter: int = 1

    #     def update_pc(vis):
    #         global counter
    #         if counter < len(xyzrgbs) - 1:
    #             pcd.points = o3d.utility.Vector3dVector(xyzrgbs[counter][:, :3])
    #             pcd.colors = o3d.utility.Vector3dVector(xyzrgbs[counter][:, 3:])
    #             vis.update_geometry(pcd)
    #             counter += 1

    #     vis.register_animation_callback(update_pc)
    #     vis.run()
    #     vis.destroy_window()

    def vis_frame_pc(self, xyz):
        # xyzrgb, xyz_cam = rgbd_to_pointcloud(obs['images']['top'], obs['depth']['top'], \
        #     cam_intrinsics, cam_extrinsics)
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
        pcd.points = o3d.utility.Vector3dVector(xyz)
        vis.add_geometry(pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        vis.add_geometry(frame)
        vis.run()
        vis.destroy_window()


    def merge_to_top(self, pc_dict, key, new_pc):
        if key in pc_dict:
            pc_dict[key] = np.vstack([pc_dict[key], new_pc])
        else:
            pc_dict[key] = new_pc

    def get_observation(self, physics):
        self.obs_idx += 1

        obs = super().get_observation(physics)

        if not self.recorded_pc and self.obs_idx == 50:
            obs['socket_pc'] = dict()
            obs['peg_pc'] = dict()
            for cam_id in ['angle', 'back']:

                obs['depth'] = dict()
                obs['depth'][cam_id] = physics.render(height=480, width=640, camera_id=cam_id, depth=True)
                obs['seg'] = dict()
                raw_seg= physics.render(height=480, width=640, camera_id=cam_id, segmentation=True)
                processed_seg = raw_seg[:, :, 0].astype(np.uint8) 
                obs['seg'][cam_id] = processed_seg+ 1  # for visualization


                peg_mask = self.get_mask('peg', physics, processed_seg)
                socket_mask = self.get_mask('socket', physics, processed_seg)

                cam_intrinsics = self.get_cam_intrinsics(physics, cam_id)
                cam_extrinsics = self.get_cam_extrinsics(physics, cam_id)

                socket_pc = self.get_pc_with_mask(socket_mask, cam_intrinsics, cam_extrinsics, \
                    obs['depth'][cam_id], obs['images'][cam_id])
                peg_pc = self.get_pc_with_mask(peg_mask, cam_intrinsics, cam_extrinsics, \
                    obs['depth'][cam_id], obs['images'][cam_id])

                obs['socket_pc'][cam_id] = socket_pc
                obs['peg_pc'][cam_id] = peg_pc

                ## merge front and back to top
                self.merge_to_top(obs['socket_pc'], 'top', obs['socket_pc'][cam_id])
                self.merge_to_top(obs['peg_pc'], 'top', obs['peg_pc'][cam_id])

            self.recorded_pc = True

            # ### debug using o3d
            # self.vis_frame_pc(obs['peg_pc']['top'])
        return obs

    def get_mask(self, obj_name, physics, raw_seg):
        obj_ids = get_geom_ids(physics, obj_name)
        obj_mask = np.isin(raw_seg, obj_ids)
        return obj_mask

    #https://github.com/google-deepmind/dm_control/issues/85
    ## https://github.com/openai/mujoco-py/issues/271
    def get_cam_intrinsics(self, physics, cam_name):
        camera = mujoco.Camera(physics, camera_id=cam_name)
        camera.update()

        height, width = 480, 640
        fovy = physics.model.cam_fovy[physics.model.name2id("top", "camera")]
        # f =  0.5 * height / math.tan(0.5*math.radians(fovy))
        # intrinsic_matrix = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        
        theta = math.radians(fovy)
        # fx = width / 2 / np.tan(theta / 2)
        fy = height / 2 / np.tan(theta / 2)
        fx = fy
        cx = (width-1) / 2.0
        cy = (height-1) / 2.0
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return intrinsic_matrix

    # https://simulately.wiki/docs/snippets/mujoco/camera/
    def get_cam_extrinsics(self, physics, cam_name):
        camera_id = physics.model.name2id(cam_name, "camera")
        rot = physics.data.cam_xmat[camera_id].reshape(3, 3)

        ## in mujoco, z points front and x points right
        R_camera_to_world = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        rot = np.dot(R_camera_to_world, rot)
        pos = physics.data.cam_xpos[camera_id]
        translation = np.eye(4)
        translation[:3, :3] = rot
        translation[:3, 3] = pos
        return translation

    def get_pc_with_mask(self, seg_mask, cam_intrinsics, cam_extrinsics,  \
        depth, color, save_ply = False, target_size = 256):
        # mask_pixels=  np.argwhere(seg_mask)
        # depth_values = depth[mask_pixels[:, 0], mask_pixels[:, 1]]
        

        # fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        # x_cam = (mask_pixels[:, 1] - cx) * depth_values / fx
        # y_cam = (mask_pixels[:, 0] - cy) * depth_values / fy
        # z_cam = depth_values
        # points_camera = np.vstack([x_cam, y_cam, z_cam, np.ones_like(z_cam)]) ## should be 4*N
        # assert points_camera.shape[0] == 4

        xyzrgb, xyz_cam = rgbd_to_pointcloud(color, depth, \
            cam_intrinsics, cam_extrinsics, seg_mask=seg_mask)

        ## downsample to target_size
        xyz_selected = resize_point_cloud(xyzrgb[:, :3], target_size)

       ### save the point cloud
        if save_ply:
            # points_camera_n3 = points_camera[:3, :]
            points_camera_n3 = xyz_cam.T
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_camera_n3.T)
            o3d.io.write_point_cloud("mujoco_camera.ply", pcd)

            # points_world = np.dot(cam_extrinsics, points_camera)
            # points_world = points_world[:3, :]
            points_world = xyzrgb[:, :3].T

            ### save the point cloud
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world.T )
            o3d.io.write_point_cloud("mujoco_world.ply", pcd)

        return xyz_selected


        


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

