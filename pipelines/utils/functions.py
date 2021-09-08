import json
from scipy.spatial.transform import Rotation as R
import ast
import math
import numpy as np

def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c)

def quaternion_to_rotation_matrix(quaternion_wxyz):
    r = R.from_quat([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    matrix = r.as_matrix()
    matrix[:3,2] = -matrix[:3,2]
    matrix[:3,1] = -matrix[:3,1]
    return matrix

def get_point_3d(x, y, depth, fx, fy, cx, cy, cam_center_world, R_world_to_cam, w_in_quat_first = True):
    if depth <= 0:
        return 0
    new_x = (x - cx)*depth/fx
    new_y = (y - cy)*depth/fy
    new_z = depth
    coord_3D_world_to_cam = np.array([new_x, new_y, new_z], float)
    if len(R_world_to_cam) == 4:
        if w_in_quat_first:
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
        else:
            R_world_to_cam = [R_world_to_cam[3], R_world_to_cam[0], R_world_to_cam[1], R_world_to_cam[2]]
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
    coord_3D_cam_to_world = np.matmul(matrix, coord_3D_world_to_cam) + cam_center_world
    return coord_3D_cam_to_world


def clouds3d_from_kpt(path):
    filedict = open(path, 'r')
    dict_string = json.load(filedict)
    ast.literal_eval(dict_string)  
    kpt_coord = ast.literal_eval(dict_string)
    
    points_db = []
    points_query = []
    for mode in kpt_coord.keys():
        for triple in kpt_coord[mode]:
            x, y, depth_point = triple
            if depth_point > 0 and not (math.isnan(depth_point) or (math.isinf(depth_point))): 
                point_3d_xyz = cloud_3d_cam(x, y, depth_point)
                
            else:
                continue
            if mode == list(kpt_coord.keys())[0]:
                points_query.append(list(point_3d_xyz))
            else:
                points_db.append(list(point_3d_xyz))

    size = min(len(points_query), len(points_db))
    points_3d_query = np.empty((3, size), float)
    points_3d_mapping = np.empty((3, size), float)
    for i in range(size):
        points_3d_query[:,i] = points_query[i]
        points_3d_mapping[:,i] = points_db[i]
    
    return points_3d_query, points_3d_mapping

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

def cloud_3d_cam(x, y, depth, fx = 128, fy = 128, cx = 128, cy = 128):
    if depth <= 0:
        return 0
    new_x = (x - cx)*depth/fx
    new_y = (y - cy)*depth/fy
    new_z = depth
    coord_3D_world_to_cam = np.array([new_x, new_y, new_z], float)
    return coord_3D_world_to_cam

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]