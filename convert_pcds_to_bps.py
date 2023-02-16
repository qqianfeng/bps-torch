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


def main(base_path, data_set_names, use_existing_bps, bps_path=None):
    # Generate random bps
    if use_existing_bps:
        assert bps_path is not None
        bps = np.load(bps_path)
    else:
        bps = bps_torch(bps_type='random_uniform', n_bps_points=4096, radius=0.4, n_dims=3)
        # Save the "ground_truth" bps
        np.save(os.path.join(base_path, 'basis_point_set.npy'), to_np(bps.bps.squeeze()))

    bps_np = bps.bps.detach().cpu().numpy()
    bps_np = bps_np.reshape(-1, 3)
    for dset in data_set_names:
        pcd_folder = os.path.join(base_path, dset, 'point_clouds')
        objs = [obj for obj in os.listdir(pcd_folder) if '.' not in obj]
        for obj_full in objs:
            print("Processing object: ", obj_full)
            pcd_obj_path = os.path.join(pcd_folder, obj_full)
            bps_obj_folder = os.path.join(base_path, dset, 'bps', obj_full)
            if not os.path.exists(bps_obj_folder):
                os.makedirs(bps_obj_folder)
            pcds = [f for f in os.listdir(pcd_obj_path) if 'pcd' in f]
            for pcd_name in pcds:
                pcd_path = os.path.join(pcd_obj_path, pcd_name)
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points)

                pcd_enc = bps.encode(points)['dists']
                pcd_enc_np = np.squeeze(to_np(pcd_enc))

                # show_pcd_and_bps(points, bps_np)

                num_str = pcd_path.split('pcd')[-2][:-1]

                save_path = os.path.join(bps_obj_folder, obj_full + '_bps' + num_str + '.npy')
                print(save_path)
                np.save(save_path, pcd_enc_np)


if __name__ == '__main__':

    base_path = '/home/vm/new_data_full'
    # bps_path = '/home/vm/new_data_full/basis_point_set.npy'
    main(base_path,
         data_set_names=['train', 'test', 'eval'],
         use_existing_bps=False,
         bps_path=None)
