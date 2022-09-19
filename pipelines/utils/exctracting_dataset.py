import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import argparse
from os.path import join, exists, isfile
from pathlib import Path
import re

def exctracting_hdf5(hdf5_dataset_path, force):
    input_dir = hdf5_dataset_path
    # root_datasets = Path(hdf5_dataset_path)
    dataset_path =  join(input_dir, 'extracted_HPointLoc')
    print(dataset_path)
    print(exists(dataset_path))
    if not exists(dataset_path) or force:
        shutil.rmtree(dataset_path, ignore_errors=True) 
        query_path = os.path.join(dataset_path, 'query')
        os.makedirs(query_path, exist_ok=True)
        database_path = os.path.join(dataset_path, 'mapping')
        os.makedirs(database_path, exist_ok=True)

        global_counter_query = 0
        global_counter_db = 0
        w, h, fx, fy, cx, cy = 256, 256, 128, 128, 128, 128
        #count = 0
        for map_name in tqdm(os.listdir(input_dir)):
            if (map_name == "extracted_HPointLoc"):
                continue
            if (map_name.find(".") != -1):
                continue
            if (map_name == "_info"):
                continue
            if map_name.find('.') != -1:
                continue
            print("map_name: {}".format(map_name))
            for hdf5_dataset in sorted(os.listdir(os.path.join(input_dir, map_name))):
                # print("hdf5_dataset: {}".format(hdf5_dataset))
                if hdf5_dataset.find('.hdf5') == -1:
                    continue
            
                hdf5_dataset_path = os.path.join(input_dir, map_name, hdf5_dataset)
                file = h5py.File(hdf5_dataset_path, 'r')
                rgb_base = file['rgb_base']
                depth_base = file['depth_base']
                gps_base = file['gps_base']
                quat_base = file['quat_base']
                rgb = file['rgb']
                depth = file['depth']
                gps = file['gps']
                quat = file['quat']
                num_cloud = re.findall(r'\d+', hdf5_dataset)[-2]
                semantic_base = file['semantic_base']
                
                output_query_path_images = os.path.join(query_path, 'sensors', 'records_data', map_name, str(num_cloud))
                os.makedirs(output_query_path_images, exist_ok=True)
                output_database_path_images = os.path.join(database_path, 'sensors', 'records_data',  map_name, str(num_cloud))
                os.makedirs(output_database_path_images, exist_ok=True)
                
                for num_query in range(len(rgb)):
                    with open(query_path + '/sensors.txt', 'a') as fin:
                        fin.write(f'sensor{global_counter_query}, ,camera, PINHOLE, {w}, {h}, {fx}, {fy}, {cx}, {cy}\n')
                    
                    with open(query_path + '/trajectories.txt', 'a') as fin:
                        qw, qx, qy, qz = quat[num_query]
                        tx, ty, tz =  gps[num_query]
                        fin.write(f'{global_counter_query}, sensor{global_counter_query}, {qw}, {qx}, {qy}, {qz}, {tx}, {ty}, {tz}\n')
                    
                    with open(query_path + '/records_camera.txt', 'a') as fin:
                        fin.write(f'{global_counter_query}, sensor{global_counter_query}, {map_name}/{num_cloud}/query_{str(num_query).zfill(4)}.png\n')
                        
                    
                    with open(query_path + '/records_depth.txt', 'a') as fin:
                        fin.write(f'{global_counter_query}, sensor{global_counter_query}, {map_name}/{num_cloud}/query_{str(num_query).zfill(4)}.npy\n')
                        
                    cv2.imwrite(os.path.join(output_query_path_images, 'query_' + str(num_query).zfill(4) + '.png'), cv2.cvtColor(rgb[num_query], cv2.COLOR_RGB2BGR))
                    np.save(os.path.join(output_query_path_images, 'query_' + str(num_query).zfill(4) + '.png'), depth[num_query])

                    global_counter_query += 1

                for num_base in range(len(rgb_base)):
                    with open(database_path + '/sensors.txt', 'a') as fin:
                        fin.write(f'sensor{global_counter_db}, ,camera, PINHOLE, {w}, {h}, {fx}, {fy}, {cx}, {cy}\n')
                    
                    with open(database_path + '/trajectories.txt', 'a') as fin:
                        qw, qx, qy, qz = quat_base[num_base]
                        tx, ty, tz =  gps_base[num_base]
                        fin.write(f'{global_counter_db}, sensor{global_counter_db}, {qw}, {qx}, {qy}, {qz}, {tx}, {ty}, {tz}\n')
                    
                    with open(database_path + '/records_camera.txt', 'a') as fin:
                        fin.write(f'{global_counter_db}, sensor{global_counter_db}, {map_name}/{num_cloud}/mapping_{str(num_base).zfill(4)}.png\n')
                        
                    
                    with open(database_path + '/records_depth.txt', 'a') as fin:
                        fin.write(f'{global_counter_db}, sensor{global_counter_db}, {map_name}/{num_cloud}/mapping_{str(num_base).zfill(4)}.npy\n')
                        
                    cv2.imwrite(os.path.join(output_database_path_images, 'mapping_' + str(num_base).zfill(4) + '.png'), cv2.cvtColor(rgb_base[num_base], cv2.COLOR_RGB2BGR))
                    np.save(os.path.join(output_database_path_images, 'mapping_' + str(num_base).zfill(4) + '.png'), depth_base[num_base])
                    global_counter_db += 1

                    cv2.imwrite(os.path.join('assets', map_name + '_point' + num_cloud, str(num_base).zfill(4) + '.png'), cv2.cvtColor(rgb_base[num_base], cv2.COLOR_RGB2BGR)) 
    else: print('Dataset already exctracted, skipped extraction process. Use flag --force to overwrite dataset')


if __name__ == '__main__':
    print('>>>> HPointLoc.hdf5 exctracting >>>>\n')
    parser = argparse.ArgumentParser(description=('evaluate place recognition pipeline on Habitat dataset'))
    parser.add_argument('--dataset_path', required=True, help='path to HPointLoc dataset')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='overwrite dataset')
    args = parser.parse_args()
    exctracting_hdf5(args.dataset_path, args.force)