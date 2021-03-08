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
    bps = bps_torch(bps_type='random_uniform', n_bps_points=4096, radius=0.15, n_dims=3)
    # Save the "ground_truth" bps
    np.save(os.path.join(data_path, 'basis_point_set.npy'), to_np(bps.bps.squeeze()))
    for obj_full in os.listdir(data_path):
        obj_path = os.path.join(data_path, obj_full)
        dirs = os.listdir(obj_path)
        for pcd_name in dirs:
            pcd_path = os.path.join(obj_path, pcd_name)
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            pcd_enc = bps.encode(points)
            bps_np = np.squeeze(to_np(bps.bps))

            #show_pcd_and_bps(points, bps_np)

            num_str = pcd_path.split('pcd')[-2][:-1]
            save_path = os.path.join(os.path.split(pcd_path)[0], obj_full + '_' + num_str + '.npy')
            np.save(save_path, bps_np)


if __name__ == '__main__':
    bps = np.load(
        '/home/vm/data/vae-grasp/point_clouds/kit_CoffeeCookies/kit_CoffeeCookies003.npy')
    data_path = '/home/vm/data/vae-grasp/point_clouds'
    main(data_path)