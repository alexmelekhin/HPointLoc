from tqdm import tqdm
import h5py
from os.path import join
import os
import numpy as np
from pathlib import Path
import json
import numpy as np

MAXDEPTH = 10

def conv_to_json(dataset_root, path_to_npz_folder, output_dir):
    root_datasets = Path(dataset_root).parent
    dataset_path =  join(root_datasets, 'HPointLoc_dataset')
    pairs_npz = os.listdir(path_to_npz_folder) 
    os.makedirs(output_dir, exist_ok = True)
    for pair_npz in tqdm(pairs_npz):
        npz = np.load(join(path_to_npz_folder, pair_npz))
        q_fold, q_cloud, query, q_name = pair_npz.split('_')[:4] 
        m_fold, m_cloud, mapping, m_name = pair_npz.split('_')[4:8] 
        q = '_'.join(pair_npz.split('_')[:4])
        m = '_'.join(pair_npz.split('_')[4:8]) 

        q_cloud = q_fold + '_point' + q_cloud + '.hdf5'
        m_cloud = m_fold + '_point' + m_cloud + '.hdf5'

        hdf5_q_path = join(dataset_path, q_fold, q_cloud)
        hdf5_m_path = join(dataset_path, m_fold, m_cloud)

        q_file = h5py.File(hdf5_q_path, 'r')
        m_file = h5py.File(hdf5_m_path, 'r')
        
        depth_base = m_file['depth_base']
        depth = q_file['depth']

        q_depth = np.squeeze(depth[int(q_name)])*MAXDEPTH
        m_depth = np.squeeze(depth_base[int(m_name)])*MAXDEPTH

        q_coord_frame = []
        m_coord_frame = []

        for kpt in range(min(npz['keypoints1'].shape[0], npz['matches'].shape[0])): 
            if npz['matches'][kpt] != -1:
                x_q, y_q = map(int, npz['keypoints0'][kpt])
                x_m, y_m = map(int, npz['keypoints1'][npz['matches'][kpt]])
                
                q_coord_frame.append((x_q, y_q, float(q_depth[y_q, x_q])))
                m_coord_frame.append((x_m, y_m, float(m_depth[y_m, x_m])))
    
        dictionary_kpt = {q: q_coord_frame, m:m_coord_frame}
        outpath = join(output_dir, q + '_' + m + '.json')
        with open(outpath, 'w') as outfile:
            json.dump(str(dictionary_kpt), outfile)
