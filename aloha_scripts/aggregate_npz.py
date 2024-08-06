import numpy as np
import os
import open3d as o3d

# TODO: save scene.npz and obj.npz when collecting demonstrations. For each demo, save 20 frames.
# Remember to add obj.observed_pose as the input of riemann.
# record more tape pc using record_episode_grasp.py for testing.


def process_point_cloud(point_cloud, target_size):
    # Downsample the point cloud to the target size
    pc_size = len(point_cloud.points)
    if pc_size > target_size:
        indices = np.random.choice(len(point_cloud.points), target_size, replace=False)
        point_cloud = point_cloud.select_by_index(indices)
    else:
        raise ValueError('Point cloud size is less than the target size')

    return point_cloud

def old_process(save_dir):
    # Define the number of demonstrations and target point cloud size
    num_demos = 10
    target_size = 1024  # Adjust this value as needed

    # Initialize lists to hold data for each demonstration
    xyz_out = np.zeros((num_demos, target_size, 3))
    rgb_out = np.zeros((num_demos, target_size, 3))
    seg_center_out = np.zeros((num_demos, 3))
    axes_out = np.zeros((num_demos, 9))    
    # Iterate over all files in the save_dir
    for file in os.listdir(save_dir):
        # if file.endswith(".ply") and file.startswith("scene"):
        if file.endswith(".ply") and file.startswith("graspobj"):
            episode_idx = int(file.split('_')[1].split('.')[0])
            full_file = os.path.join(save_dir, file)
            loaded_pc = o3d.io.read_point_cloud(full_file)
            
            normalized_pc = process_point_cloud(loaded_pc, target_size)
            xyz = np.asarray(normalized_pc.points)
            rgb = np.asarray(normalized_pc.colors)
            print('xyz shape:', xyz.shape)
            
            xyz_out[episode_idx] = xyz
            rgb_out[episode_idx] = rgb

        if file.endswith('.npz') and file.startswith('graspPose'):
            episode_idx = int(file.split('_')[1].split('.')[0])
            npz_file = os.path.join(save_dir, file)
            npz_dict = np.load(npz_file)
            
            seg_center_out[episode_idx] = npz_dict['seg_center']
            axes_out[episode_idx] = npz_dict['axes']


    # Save data to a .npz file
    npz_dict = {
        'seg_center': seg_center_out,
        'axes': axes_out,
        'xyz': xyz_out,
        'rgb': rgb_out
    }
    # npz_file = os.path.join(save_dir, 'riemann_demo.npz')
    npz_file = os.path.join(save_dir, 'riemann_focus_demo.npz')
    np.savez(npz_file, **npz_dict)


def process(save_dir):
    num_demos = 9
    target_size = 1024
    id_offset = 101
    # xyz_out = np.zeros((num_demos, target_size, 3))
    # rgb_out = np.zeros((num_demos, target_size, 3))
    # seg_center_out = np.zeros((num_demos, 3))
    # axes_out = np.zeros((num_demos, 9))  
    # obj_point_out = np.zeros((num_demos, 3)) 
    xyz_ls = []
    rgb_ls = []
    seg_center_ls = []
    axes_ls = []
    obj_point_ls = []   
    for episode_idx in range(101, 110):
        npz_file = os.path.join(save_dir, f'graspDemo_{episode_idx}.npz')
        npz_dict = np.load(npz_file)

        xyz_points = npz_dict['xyz']
        rgb_points = npz_dict['rgb']
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_points)
        normalized_pc = process_point_cloud(pcd, target_size)
        xyz = np.asarray(normalized_pc.points)
        rgb = np.asarray(normalized_pc.colors)

        traj_len = npz_dict['axes'].shape[0]
        for i in range(traj_len):

            seg_center = npz_dict['seg_center'][i]
            axes = npz_dict['axes'][i]

            xyz_ls.append(xyz)
            rgb_ls.append(rgb)
            seg_center_ls.append(seg_center)
            axes_ls.append(axes)
            obj_point_ls.append(npz_dict['obj_point'][i])

            # xyz_out[episode_idx-id_offset] = xyz
            # rgb_out[episode_idx-id_offset] = rgb
            # seg_center_out[episode_idx-id_offset] = seg_center
            # axes_out[episode_idx-id_offset] = axes
            # obj_point_out[episode_idx-id_offset] = npz_dict['obj_point'][i]

    # Save data to a .npz file
    # npz_dict = {
    #     'seg_center': seg_center_out,
    #     'axes': axes_out,
    #     'xyz': xyz_out,
    #     'rgb': rgb_out,
    #     'obj_point': obj_point_out
    # }
    npz_dict = {
        'seg_center': np.asarray(seg_center_ls),
        'axes': np.asarray(axes_ls),
        'xyz': np.asarray(xyz_ls),
        'rgb': np.asarray(rgb_ls),
        'obj_point': np.asarray(obj_point_ls)
    }
    npz_file = os.path.join(save_dir, f'riemann_center.npz')
    np.savez(npz_file, **npz_dict)
        
if __name__ == '__main__':
    save_dir = '/home/xuhang/interbotix_ws/src/ACT/aloha/depth_data/aloha_transfer_tape'
    process(save_dir)




    

