from tqdm import tqdm
import h5py
from utils.functions import quaternion_to_rotation_matrix, clouds3d_from_kpt, is_invertible
from scipy.spatial.transform import Rotation as R
import re
from pathlib import Path
from os.path import join, exists, isfile
import open3d as o3d
import numpy as np

def icp(dataset_root, path_image_retrieval, path_loc_features_matches, output_dir, topk=1):
    trans_init = np.asarray([[1,0,0,0],   
                            [0,1,0,0],   
                            [0,0,1,0], 
                            [0,0,0,1]])
    results = {
        "(5m, 20°)": 0,
        "(1m, 10°)": 0,
        "(0.5m, 5°)": 0,
        "(0.25m, 2°)": 0,
        "(5m)": 0,
        "(1m)": 0,
        "(0.5m)": 0,
        "(0.25m)": 0
    }
    icp_numbers = 0
    query_numbers = 0
    threshold = 1.4
    root_datasets = Path(dataset_root).parent
    dataset_path =  join(root_datasets, 'HPointLoc_dataset')

    with open(path_image_retrieval, 'r') as f:
        for pair in tqdm(f.readlines()[2:]):
            query_numbers += 1
            q, m, score = pair.split(', ')
            q = q.split('records_data/')[1]
            m = m.split('records_data/')[1]

            q_fold, q_cloud, q_name = q.split('/')
            m_fold, m_cloud, m_name = m.split('/')
            
            q_cloud = q_fold + '_point' + q_cloud + '.hdf5'
            m_cloud = m_fold + '_point' + m_cloud + '.hdf5'

            hdf5_q_path = join(dataset_path, q_fold, q_cloud)
            hdf5_m_path = join(dataset_path, m_fold, m_cloud)

            q_file = h5py.File(hdf5_q_path, 'r')
            m_file = h5py.File(hdf5_m_path, 'r')
            
            rgb_base = m_file['rgb_base']
            depth_base = m_file['depth_base']
            gps_base = m_file['gps_base']
            quat_base = m_file['quat_base']
            
            rgb = q_file['rgb']
            depth = q_file['depth']
            gps = q_file['gps']
            quat = q_file['quat']

            q_name = int(re.findall(r'\d+', q_name)[0])
            m_name = int(re.findall(r'\d+', m_name)[0])

            gt_transl = gps[q_name]
            gt_quat_wxyz = quat[q_name]
            gt_quat_xyzw = [gt_quat_wxyz[1], gt_quat_wxyz[2], gt_quat_wxyz[3], gt_quat_wxyz[0]]

            estimated_transl = gps_base[m_name]
            estimated_quat_wxyz = quat_base[m_name]
            estimated_quat_xyzw = [estimated_quat_wxyz[1], estimated_quat_wxyz[2], estimated_quat_wxyz[3], estimated_quat_wxyz[0]]
        
            pairpath = q.replace('/','_') + '_' + m.replace('/','_') +'.json'
            fullpath = join(path_loc_features_matches, pairpath)

            fullpath = re.sub('.png', '', fullpath)

            points_3d_query, points_3d_mapping = clouds3d_from_kpt(fullpath) 

            if points_3d_mapping.shape[1] > 1:
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(points_3d_query.transpose())  
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(points_3d_mapping.transpose())  

                reg_p2p = o3d.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.registration.TransformationEstimationPointToPoint())

                Rotation = reg_p2p.transformation[:3,:3]
                translation = reg_p2p.transformation[:3,3]

                gt_mapping_4x4 = np.zeros((4,4))
                gt_mapping_4x4[:3,:3] = quaternion_to_rotation_matrix(estimated_quat_wxyz)
                gt_mapping_4x4[:3,3] = estimated_transl
                gt_mapping_4x4[-1,-1] = 1


                translation_4x4 = list(translation)
                translation_4x4.append(1.0)
                transformation_4x4 = np.zeros((4,4))
                transformation_4x4[:3,:3] = Rotation
                transformation_4x4[:,3] = translation_4x4

                predict = gt_mapping_4x4 @ np.linalg.inv(transformation_4x4)
                quat_predict = R.from_matrix(predict[:3,:3]).as_quat()
                qw_qx_qy_qz_predict = [quat_predict[3], quat_predict[0], quat_predict[1], quat_predict[2]]
                xyz_predict = predict[:3,3]

                if not np.isnan(qw_qx_qy_qz_predict).any():
                    icp_numbers += 1
                    estimated_transl = xyz_predict
                    estimated_quat_wxyz = qw_qx_qy_qz_predict
                    estimated_quat_xyzw = [estimated_quat_wxyz[1], estimated_quat_wxyz[2], estimated_quat_wxyz[3], estimated_quat_wxyz[0]]

            pose_estimated = np.eye(4)
            pose_gt = np.eye(4)

            r = R.from_quat(estimated_quat_xyzw)
            pose_estimated[:3, :3] = r.as_matrix()
            pose_estimated[:3, 3] = estimated_transl

            r = R.from_quat(gt_quat_xyzw)
            pose_gt[:3, :3] = r.as_matrix()
            pose_gt[:3, 3] = gt_transl

            error_pose = np.linalg.inv(pose_estimated) @ pose_gt

            dist_error = np.sum(error_pose[:3, 3]**2) ** 0.5

            r = R.from_matrix(error_pose[:3, :3])
            rotvec = r.as_rotvec()
            angle_error = (np.sum(rotvec**2)**0.5) * 180 / 3.14159265353
            angle_error = abs(90 - abs(angle_error-90))


            if  dist_error < 0.25:
                results["(0.25m)"] += 1
            if  dist_error < 0.5:
                results["(0.5m)"] += 1
            if  dist_error < 1:
                results["(1m)"] += 1
            if  dist_error < 5:
                results["(5m)"] += 1    

            if angle_error < 2 and dist_error < 0.25:
                results["(0.25m, 2°)"] += 1
            if angle_error < 5 and dist_error < 0.5:
                results["(0.5m, 5°)"] += 1
            if angle_error < 10 and dist_error < 1:
                results["(1m, 10°)"] += 1
            if angle_error < 20 and dist_error < 5:
                results["(5m, 20°)"] += 1

    for key in results.keys():
        results[key] = results[key] / query_numbers

    print('>>>> \n', results, '\n>>>>',)        
    print('Proportion of optimized:', icp_numbers/query_numbers)

    # outpath = join(output_dir, q + '_' + m + '.json')    
    # with open(outpath, 'w') as outfile:
    # json.dump(str(dictionary_kpt), outfile)