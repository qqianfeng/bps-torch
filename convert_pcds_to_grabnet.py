import os
import open3d as o3d
import numpy as np
from bps_torch.bps import bps_torch
from bps_torch.utils import to_np


def show_pcd_and_bps(pcd, bps):
    color_pcd_np = np.tile(np.array([1, 0, 0]), (pcd.shape[0], 1))
    color_bps_np = np.tile(np.array([0, 1, 0]), (bps.shape[0], 1))

    pcd_pcd = o3d.geometry.PointCloud()
    bps_pcd = o3d.geometry.PointCloud()

    pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
    bps_pcd.points = o3d.utility.Vector3dVector(bps)

    pcd_pcd.colors = o3d.utility.Vector3dVector(color_pcd_np)
    bps_pcd.colors = o3d.utility.Vector3dVector(color_bps_np)

    o3d.visualization.draw_geometries([pcd_pcd, bps_pcd])


def main(data_path):
    # Generate random bps
    bps = bps_torch(bps_type='random_uniform', n_bps_points=4096, radius=0.15, n_dims=3)
    # Save the "ground_truth" bps
    base = os.path.split(data_path)[0]
    np.save(os.path.join(data_path, 'basis_point_set.npy'), to_np(bps.bps.squeeze()))
    # Append point_clouds to path
    data_path = os.path.join(data_path, 'point_clouds')
    objs = [obj for obj in os.listdir(data_path) if '.' not in obj]
    for obj_full in objs:
        print("Processing object: ", obj_full)
        obj_path = os.path.join(data_path, obj_full)
        pcds = [dir for dir in os.listdir(obj_path) if 'pcd' in dir]
        for pcd_name in pcds:
            pcd_path = os.path.join(obj_path, pcd_name)
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            pcd_enc = bps.encode(points)['dists']
            pcd_enc_np = np.squeeze(to_np(pcd_enc))

            #show_pcd_and_bps(points, bps_np)

            num_str = pcd_path.split('pcd')[-2][:-1]
            save_path = os.path.join(
                os.path.split(pcd_path)[0], obj_full + '_bps' + num_str + '.npy')
            np.save(save_path, pcd_enc_np)


if __name__ == '__main__':

    data_path = '/home/vm/data/vae-grasp/train'
    main(data_path)