import argparse
import os
from os.path import join, exists, isfile
from os import makedirs
from utils.subprocces import run_python_command
from utils.preproccesing_dataset import preprocces_metadata
import configparser
from optimizers.teaser import teaser

def image_retrieval_stage(method, dataset_root, query_path, db_path, image_retrieval_path, topk=1):
    exct_stage = {'apgem': 'PATH.py', 'patchnetvlad':'3rd/Patch-NetVLAD/feature_extract.py', 'netvlad':'PATH.py', 'hfnet':'PATH.py'}
    matching_stage = {'patchnetvlad':'3rd/Patch-NetVLAD/feature_match.py',}
    command = exct_stage[method]
    if method == 'patchnetvlad':
        configfile = '3rd/Patch-NetVLAD/patchnetvlad/configs/{}.ini'.format(input('choose config: speed/performance \n'))
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        patch_top = int(config['feature_match']['n_values_all'].split(",")[-1])
        query_descriptor = '3rd/Patch-NetVLAD/query_descriptor'
        exctraction_stage_args = ['--config_path', configfile,
                                '--dataset_file_path', query_path,
                                '--dataset_root_dir', dataset_root, 
                                '--output_features_dir', query_descriptor]
        run_python_command(command, exctraction_stage_args, None)

        db_descriptor = '3rd/Patch-NetVLAD/db_descriptor'
        exctraction_stage_args = ['--config_path', configfile,
                                '--dataset_file_path', db_path,
                                '--dataset_root_dir', dataset_root,
                                '--output_features_dir', db_descriptor]
        run_python_command(command, exctraction_stage_args, None)
        
        command = matching_stage[method]
        matching_stage_args = ['--config_path', configfile,
                                '--dataset_root_dir', dataset_root, 
                                '--query_file_path', query_path,
                                '--index_file_path', db_path,
                                '--query_input_features_dir', query_descriptor,
                                '--index_input_features_dir', db_descriptor,
                                '--result_save_folder', image_retrieval_path ]
        run_python_command(command, matching_stage_args, None)

        #get topk reranked
        pairsfile_path = f'{method}_top{topk}.txt'
        pairsfile_path_full = join(image_retrieval_path, pairsfile_path)
        with open(join(image_retrieval_path , 'PatchNetVLAD_predictions.txt'), 'r') as origfile, \
            open(pairsfile_path_full, 'w') as topkfile:  
                for line in origfile.readlines()[2::patch_top]:
                    query_string, mapping_string = map(lambda x: x.split('\n')[0], line.split(', ')) 
                    string = query_string + ', ' + mapping_string + ', 1\n'
                    topkfile.write(string)  
    else:
        pass
    

def keypoints_matching_stage(method, dataset_root, input_pairs, output_dir,):
    methods = {'r2d2': ['PATH.py', 'PATH_k.py'], 'loftr':'PATH.py', 'superpoint_superglue':'3rd/SuperGluePretrainedNetwork/match_pairs.py',}
    command = methods[method]
    if method == 'superpoint_superglue':
        compute_image_pairs_args = ['--input_pairs', input_pairs,
                                    '--input_dir', dataset_root,  
                                    '--resize', '-1',
                                    '--output_dir', output_dir]
        run_python_command(command, compute_image_pairs_args, None)

def pose_optimization(method):
    pass


def pipeline_eval(dataset_root, image_retrieval, keypoints_matching, optimizer_cloud, topk, result_path, dataset, force):

    """
    Evaluate place recognition pipeline. Pipeline consist of 3 stages:
    - image retrieval
    - keypoints matching
    - 3d pose optimization (registration)
    """
    root_dir = os.getcwd()
    query_image_path, db_image_path = preprocces_metadata(dataset_root)

    #######image retrieval
    image_retrieval_path = join(root_dir, result_path, dataset, 'image_retrieval')
    if not exists(image_retrieval_path):
        os.makedirs(image_retrieval_path)
    #print('root directory:', root_dir)
    pairsfile_path = f'{image_retrieval}_top{topk}.txt'
    pairsfile_path_full = join(image_retrieval_path, pairsfile_path)

    if not exists(pairsfile_path_full) or force:
        image_retrieval_stage(image_retrieval, dataset_root, query_image_path, db_image_path, image_retrieval_path, topk)
    
    else:
        print('image retrieval results already computed:......')
 
    ######local features
    local_featue_path = join(root_dir, result_path, dataset, 'keypoints')
    if not exists(local_featue_path):
        os.makedirs(local_featue_path)

    keypoints_path = f'{image_retrieval}_{keypoints_matching}'
    local_features_path_full =  join(local_featue_path, keypoints_path)
    if not exists(local_features_path_full) or force: 
        keypoints_matching_stage(keypoints_matching, dataset_root, pairsfile_path_full, local_features_path_full)

    else:
        print('local feature matching already computed:......')


    ######optimization
    root_dir = os.getcwd()
    os.chdir('3rd/TEASER-plusplus/build/python')
    teaser(dataset_root, 'patchnetvlad', 'superpoint_superglue', pairsfile_path_full)
    os.chdir(root_dir)


def pipeline_command_line():
    """
    Parse the command line arguments to start place recognition .
    """
    parser = argparse.ArgumentParser(description=('evaluate place recognition pipeline on Habitat dataset'))
    parser.add_argument('-dst', '--dataset_root', required=True,
                        help='name of image retrieval')
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently delete data if already exists.')
    parser.add_argument('-imgrtv', '--image-retrieval', required=True,
                        help='name of image retrieval')
    parser.add_argument('-kpt', '--keypoints-matching', required=True,
                        help='name of keypoints-exctraction/matching method')
    parser.add_argument('-opt', '--optimizer-cloud', required=True,
                        help='name of 3d point-cloud optimizer')    
    parser.add_argument('--topk',  default=1, help='top k image-retrievals')
    parser.add_argument('--result-path',  default='result', help='path to result of evaluation')
    parser.add_argument('--dataset',   default='val', type=str, choices=['val', 'full'], help='1 map of dataset')

    args = parser.parse_args()
    pipeline_eval(args.dataset_root , args.image_retrieval, args.keypoints_matching, args.optimizer_cloud, args.topk, args.result_path, args.dataset, args.force)


if __name__ == '__main__':
    print('>>>> PNTR and other frameworks\n')
    #print('choose approach for:\nimage retrieval (pathcnetvlad, netvlad, apgem, hfnet)\nlocal features (r2d2_k, loftr, superpoint_superglue)\npoint cloud registration (teaser, icp, g2o)')
    pipeline_command_line()