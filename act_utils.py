import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

import matplotlib.pyplot as plt
from mujoco_py.generated import const
# from dm_control.mujoco.wrapper.core import id2name
def get_geom_ids(env, obj = 'peg'):
    geom_ids = []
    if obj == 'peg':
        geom_ids = [env.model.name2id('red_peg', const.OBJ_GEOM)]
    elif obj == 'socket':
        for sock_geom_name in ['socket-1', 'socket-2', 'socket-3', 'socket-4', 'pin']:
            geom_ids.append(env.model.name2id(sock_geom_name, const.OBJ_GEOM))
    else:
        raise NotImplementedError(f'{obj=}')
    return geom_ids
            

class plt_render():
    def __init__(self, ts, dt = 0.02, img_type = 'rgb', cam_name = 'angle', env = None):
        # setup plotting
        self.img_type = img_type
        self.cam_name = cam_name
        self.dt = dt
        if env is not None:
            self.env = env
        ## set up the plot  
        ax = plt.subplot()
        self.plt_img = ax.imshow(self.get_img(ts, img_type, cam_name))
        plt.ion()



    def update(self, ts):
        self.plt_img.set_data(self.get_img(ts, self.img_type, self.cam_name))
        plt.pause(self.dt)

    def get_img(self, ts, img_type = 'rgb', cam_name = 'angle'):
        if img_type == 'rgb':
            return ts.observation['images'][cam_name]
        elif img_type == 'depth':
            depth= ts.observation['depth'][cam_name]  ## already in meter

            # Shift nearest values to the origin.
            depth -= depth.min()
            # Scale by 2 mean distances of near rays.
            depth /= 2*depth[depth <= 1].mean()
            # Scale to [0, 255]
            pixels = 255*np.clip(depth, 0, 1)
            depth_pixels = pixels.astype(np.uint8)
            return depth_pixels

        elif img_type == 'seg':
            seg = ts.observation['seg'][cam_name]

            # # Display the contents of the first channel, which contains object
            # # IDs. The second channel, seg[:, :, 1], contains object types.
            # geom_ids = seg[:, :, 0]
            # # geom_ids = seg[:, :, 1]
            # # Infinity is mapped to -1
            # geom_ids = geom_ids.astype(np.float64) +1
            ### 23 will filter all objects except the peg and socket
            # geom_ids = np.clip(geom_ids, 23, geom_ids.max())
            geom_ids = seg.astype(np.float64)
            # Scale to [0, 1]
            geom_ids = geom_ids / 30
            pixels = 255*geom_ids
            seg_pixels = pixels.astype(np.uint8)
            return seg_pixels

            # ## follow https://github.com/openai/mujoco-py/issues/516
            # geoms_ids = np.unique(seg)
            # body_names = self.env.physics.model.names   
            # for i in geoms_ids:
            #     name = self.env.physics.model.id2name(i, const.OBJ_GEOM)
            #     print(f'{i=}, {name=}')

            # peg_id =  self.env.physics.model.name2id('red_peg', const.OBJ_GEOM)
            # socket_id =  self.env.physics.model.name2id('socket', const.OBJ_BODY)

def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    seg_mask = None,
    depth_trunc: float = 20.0,
    width: int = 640,
    height: int = 480,
):
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
    valid = (depth > 0) & (depth < depth_trunc)
    if seg_mask is not None:
        valid = valid & seg_mask
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
    y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
    xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
    color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
    mask = np.isnan(xyz[:, 2])
    xyz = xyz[~mask]
    color = color[~mask]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_t = (extr @ xyz_h.T).T
    xyzrgb = np.hstack([xyz_t[:, :3], color])
    return xyzrgb, xyz
    
def resize_point_cloud(point_cloud, target_size):
    num_points = point_cloud.shape[0]

    if num_points ==0: # fully blocked
        return np.zeros((target_size, 3))
    elif num_points > target_size:
        # Downsample
        idx = np.random.choice(num_points, target_size, replace=False)
        return point_cloud[idx]
    
    elif num_points < target_size:
        # Upsample: Repeat points if necessary
        while target_size - num_points > num_points:
            # Concatenate the point cloud with itself until we get close to the target size
            point_cloud = np.concatenate([point_cloud, point_cloud], axis=0)
            num_points = point_cloud.shape[0]
        
        # Add the remaining points via random sampling
        additional_idx = np.random.choice(num_points, target_size - num_points, replace=True)
        additional_points = point_cloud[additional_idx]
        return np.concatenate([point_cloud, additional_points], axis=0)
    
    return point_cloud  # Return as is if the size matches