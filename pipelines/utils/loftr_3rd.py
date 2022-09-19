import os
import torch
import numpy as np
import re
import json  
from os.path import join
from tqdm import tqdm
import h5py
from utils.subprocces import run_python_command
import sys

MAXDEPTH = 10

def loftr(dataset_path, input_pairs, output_dir, root_dir):
    sys.path.append(join(root_dir, 'PNTR/LoFTR/src/loftr/loftr.py'))
    for p in sys.path:
        print(p)
    from loftr import LoFTR, default_cfg

    os.makedirs('PNTR/LoFTR/weights', exist_ok = True)
    repo_dir = os.getcwd()
    os.chdir('PNTR/LoFTR/weights')
    exctraction_stage_args = ['--id', '1w1Qhea3WLRMS81Vod_k5rxS_GNRgIi-O']
    run_python_command('gdown', exctraction_stage_args, notbash = False)
    os.chdir(repo_dir)
    assert 1==0
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("./indoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    os.makedirs(output_dir, exist_ok = True)
    outpath = join(output_dir, q + '_' + m + '.json')
    with open(input_pairs, 'r') as f:
        for pair in f.readlines()[0:]:
            q, m, score = pair.split(', ')
            q = q.split('records_data/')[1]
            m = m.split('records_data/')[1]

            q_fold, q_cloud, q_name = q.split('/')
            m_fold, m_cloud, m_name = m.split('/')
            
            q_cloud = q_fold + '_point' + q_cloud + '.hdf5'
            m_cloud = m_fold + '_point' + m_cloud + '.hdf5'

            hdf5_q_path = os.path.join(dataset_path, q_fold)
            hdf5_m_path = os.path.join(dataset_path, m_fold)

            q_file = h5py.File(hdf5_q_path, 'r')
            m_file = h5py.File(hdf5_m_path, 'r')
            q_name = int(re.findall(r'\d+', q_name)[0])
            m_name = int(re.findall(r'\d+', m_name)[0])

            q_file = h5py.File(hdf5_q_path, 'r')
            m_file = h5py.File(hdf5_m_path, 'r')
            
            rgb_base = m_file['rgb_base']
            depth_base = m_file['depth_base']
            rgb = q_file['rgb']
            depth = q_file['depth']

            img0_raw = np.squeeze(rgb[q_name])[:,:,0]
            img1_raw = np.squeeze(rgb_base[m_name])[:,:,0]
            q_depth = np.squeeze(depth[q_name])*MAXDEPTH
            m_depth = np.squeeze(depth_base[m_name])*MAXDEPTH

            img0 = torch.from_numpy(img0_raw)[None][None].cuda()/255.
            img1 = torch.from_numpy(img1_raw)[None][None].cuda()/255.
            batch = {'image0': img0, 'image1': img1}

            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

            q_kpt = mkpts0
            m_kpt = mkpts1
            q_coord_frame = []
            m_coord_frame = []
            for j in range(m_kpt.shape[0]): # собираем все заматченные кипоинты
                x_m, y_m = map(int, m_kpt[j])
                x_q, y_q = map(int, q_kpt[j])

                q_coord_frame.append((x_q, y_q, q_depth[y_q, x_q]))
                m_coord_frame.append((x_m, y_m, m_depth[y_m, x_m]))

            dictionary_kpt = {q: q_coord_frame, m:m_coord_frame}
            outpath = os.path.join(output_dir, q.replace('/','_') + '_' + m.replace('/','_') + '.json')
            with open(outpath, 'w') as outfile:
                json.dump(str(dictionary_kpt), outfile)
