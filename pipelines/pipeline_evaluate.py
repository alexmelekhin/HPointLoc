import argparse
import os
from os.path import join, exists
from utils.subprocces import run_python_command
from utils.preproccesing_dataset import preprocces_metadata
from utils.conv_to_json import conv_to_json
from utils.loftr_3rd import loftr
import configparser
import subprocess
import shutil
from optimizers.icp import icp

def image_retrieval_stage(method, dataset_root, query_path, db_path, 
                            image_retrieval_path, topk = 1, force = False):
    """
    @:param method: name of image retrieval method
    @:param dataset_root: path to dataset
    @:param query_path: path to .txt file with paths to the query image
    @:param db_path: path to .txt file with paths to the db image
    @:param image_retrieval_path: path to output folder
    @:param topk: top-k ranked images from db
    """

    exct_stage = {'apgem': 'PATH.py', 'patchnetvlad':'PNTR/Patch-NetVLAD/feature_extract.py', 
                                        'netvlad':'PATH.py', 'hfnet':'PATH.py'}
    matching_stage = {'patchnetvlad':'PNTR/Patch-NetVLAD/feature_match.py',}
    command = exct_stage[method]

    if method == 'patchnetvlad':
        option = input('choose config: speed/performance \n')
        configfile = 'PNTR/Patch-NetVLAD/patchnetvlad/configs/{}.ini'.format(option)
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        patch_top = int(config['feature_match']['n_values_all'].split(",")[-1])
        query_descriptor = join(dataset_root, 'query_descriptor')
        if not exists(query_descriptor) or force:  
            exctraction_stage_args = ['--config_path', configfile,
                                    '--dataset_file_path', query_path,
                                    '--dataset_root_dir', dataset_root, 
                                    '--output_features_dir', query_descriptor]
            run_python_command(command, exctraction_stage_args, None)

        db_descriptor = join(dataset_root, 'db_descriptor')
        if not exists(db_descriptor) or force:
            exctraction_stage_args = ['--config_path', configfile,
                                    '--dataset_file_path', db_path,
                                    '--dataset_root_dir', dataset_root,
                                    '--output_features_dir', db_descriptor]
            run_python_command(command, exctraction_stage_args, None)
            

        if not exists(join(image_retrieval_path,'PatchNetVLAD_predictions.txt')) or force:
            command = matching_stage[method]
            matching_stage_args = ['--config_path', configfile,
                                    '--dataset_root_dir', dataset_root, 
                                    '--query_file_path', query_path,
                                    '--index_file_path', db_path,
                                    '--query_input_features_dir', query_descriptor,
                                    '--index_input_features_dir', db_descriptor,
                                    '--result_save_folder', image_retrieval_path ]
            run_python_command(command, matching_stage_args, None)
        
        
        pairsfile_path = f'{method}_{option}_top{topk}.txt'
        pairsfile_path_full = join(image_retrieval_path, pairsfile_path)
        if not exists(pairsfile_path_full) or force:
            with open(join(image_retrieval_path , 'PatchNetVLAD_predictions.txt'), 'r') as origfile, \
                open(pairsfile_path_full, 'w') as topkfile:  
                    for line in origfile.readlines()[2::patch_top]:
                        query_string, mapping_string, score = map(lambda x: x.split('\n')[0], line.split(', ')) 
                        string = query_string + ', ' + mapping_string + f', {score}\n'
                        topkfile.write(string)  
        else:
            print('image retrieval results already computed:......')
    else:
        raise Exception("Wrong name of image retrieval method")

    return pairsfile_path_full

def keypoints_matching_stage(method, dataset_root, input_pairs, 
                                        output_dir, force = False, root_dir = None):
    """
    @:param method: name of keypoints matching method
    @:param dataset_root: path to dataset
    @:param input_pairs: path to pairs (from image retrieval stage)
    @:param output_dir: path to output folder
    @:param topk: top-k ranked images from db
    """
    if method == 'superpoint_superglue':
        command = 'PNTR/SuperGluePretrainedNetwork/match_pairs.py'
        if not exists('./PNTR/SuperGluePretrainedNetwork/dump_match_pairs') or force:    
            compute_image_pairs_args = ['--input_pairs', input_pairs,
                                        '--input_dir', dataset_root,  
                                        '--resize', '-1', 
                                        '--output_dir', './PNTR/SuperGluePretrainedNetwork/dump_match_pairs']
            run_python_command(command, compute_image_pairs_args, None)
        else:
            print('local feature matching already computed:......')

        if not exists(output_dir) or force:   
            print('>>>> converting keypoints to json format file')  
            conv_to_json(dataset_root, './PNTR/SuperGluePretrainedNetwork/dump_match_pairs', output_dir) 
    elif method == 'loftr':
        loftr(dataset_root, input_pairs, output_dir, root_dir)
    else:
        raise Exception("Wrong name of keypoints-matching method")

def pose_optimization(dataset_root, image_retrieval, kpt_matching, 
                                    pose_optimization, force, output_dir, topk = 1):
    """
    @:param dataset_root: path to dataset
    @:param image_retrieval: name of image retrieval method
    @:param kpt_matching: name of keypoints matching method
    @:param pose_optimization: name of point cloud optimization method
    """
    if pose_optimization == 'teaser':
        if not exists('./PNTR/TEASER-plusplus/') or force:
            shutil.rmtree('./PNTR/TEASER-plusplus/', ignore_errors=True)
            completed = subprocess.run(['bash', './PNTR/teaser.sh'])
        from optimizers.teaser import teaser
        print('>>>> TEASER++ Point cloud registration')
        teaser(dataset_root, image_retrieval, kpt_matching, output_dir)

    elif pose_optimization == 'icp':
        print('>>>> ICP Point cloud registration')
        icp(dataset_root, image_retrieval, kpt_matching, output_dir)
    else:
        raise Exception("Wrong name of pose_optimization method")

def pipeline_eval(dataset_root, image_retrieval, keypoints_matching, 
                                optimizer_cloud, topk, result_path, dataset, force):

    """
    Evaluate Place recognition pipeline. Pipeline consists of 3 stages:
    - image retrieval
    - keypoints matching
    - 3d pose optimization (registration)
    """
    root_dir = os.getcwd()
    query_image_path, db_image_path = preprocces_metadata(dataset_root)

    ###image retrieval
    image_retrieval_path = join(root_dir, result_path, dataset, 'image_retrieval')
    if not exists(image_retrieval_path):
        os.makedirs(image_retrieval_path)
    
    pairsfile_path_full = image_retrieval_stage(image_retrieval, dataset_root, query_image_path, 
                                                    db_image_path, image_retrieval_path, topk, force)
        
    ###local features
    local_featue_path = join(root_dir, result_path, dataset, 'keypoints')
    if not exists(local_featue_path):
        os.makedirs(local_featue_path)

    keypoints_path = f'{image_retrieval}_{keypoints_matching}'
    local_features_path_full =  join(local_featue_path, keypoints_path)
    keypoints_matching_stage(keypoints_matching, dataset_root, pairsfile_path_full, 
                                    local_features_path_full, force, root_dir)

    ###point cloud optimization
    output_dir = join(root_dir, result_path, dataset, 'pose_optimization')
    pose_optimization(dataset_root, pairsfile_path_full, local_features_path_full, 
                                    optimizer_cloud, force, output_dir, topk)

def pipeline_command_line():
    """
    Parse the command line arguments to start place recognition .
    """
    parser = argparse.ArgumentParser(description=('evaluate place recognition pipeline on Habitat dataset'))
    parser.add_argument('-dst', '--dataset_root', required=True,
                        help='name of image retrieval')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='silently delete data if already exists.')
    parser.add_argument('-imgrtv', '--image-retrieval', required=True,
                        help='name of image retrieval')
    parser.add_argument('-kpt', '--keypoints-matching', required=True,
                        help='name of keypoints-exctraction/matching method')
    parser.add_argument('-opt', '--optimizer-cloud', required=True,
                        help='name of 3d point-cloud optimizer')    
    parser.add_argument('--topk',  default=1, help='top k image-retrievals')
    parser.add_argument('--result-path',  default='result', help='path to result of evaluation')
    parser.add_argument('--dataset',   default='val', type=str, choices=['val', 'full'], help='maps of dataset')

    args = parser.parse_args()
    pipeline_eval(args.dataset_root , args.image_retrieval, args.keypoints_matching, 
                args.optimizer_cloud, args.topk, args.result_path, args.dataset, args.force)

if __name__ == '__main__':    
    print('>>>> PNTR framework\n')
    pipeline_command_line()
