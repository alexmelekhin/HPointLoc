import os
import tqdm
import torch
import torch.nn.functional as F
from typing import Optional

os.environ['DB_ROOT'] = ''

from dirtorch.utils import common  # noqa: E402
from dirtorch.utils.common import tonumpy, pool  # noqa: E402
from dirtorch.datasets.generic import ImageList  # noqa: E402
from dirtorch.test_dir import extract_image_features  # noqa: E402
from dirtorch.extract_features import load_model  # noqa: E402

import kapture  # noqa: E402
from kapture.io.csv import kapture_from_dir, get_all_tar_handlers  # noqa: E402
from kapture.io.csv import get_feature_csv_fullpath, global_features_to_file  # noqa: E402
from kapture.io.records import get_image_fullpath  # noqa: E402
from kapture.io.features import get_global_features_fullpath, image_global_features_to_file  # noqa: E402
from kapture.io.features import global_features_check_dir  # noqa: E402


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu>=0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True # speed-up cudnn
        torch.backends.cudnn.fastest = True # even more speed-up?
        print( 'Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'] )

    else:
        print( 'Launching on CPU' )

    return cuda


def extract_kapture_global_features(kapture_root_path: str, net, global_features_type: str,
                                    trfs, pooling='mean', gemp=3, whiten=None,
                                    threads=8, batch_size=16):
    """ Extract features from trained model (network) on a given dataset.
    """
    print(f'loading {kapture_root_path}')

    time_ap_gem = []
    iscuda = torch_set_gpu(0)
    with get_all_tar_handlers(kapture_root_path,
                              mode={kapture.Keypoints: 'r',
                                    kapture.Descriptors: 'r',
                                    kapture.GlobalFeatures: 'a',
                                    kapture.Matches: 'r'}) as tar_handlers:
        kdata = kapture_from_dir(kapture_root_path, None,
                                 skip_list=[kapture.Keypoints,
                                            kapture.Descriptors,
                                            kapture.Matches,
                                            kapture.Points3d,
                                            kapture.Observations],
                                 tar_handlers=tar_handlers)
        root = get_image_fullpath(kapture_root_path, image_filename=None)
        assert kdata.records_camera is not None
        imgs = [image_name for _, _, image_name in kapture.flatten(kdata.records_camera)]
        if kdata.global_features is None:
            kdata.global_features = {}

        if global_features_type in kdata.global_features:
            imgs = [image_name
                    for image_name in imgs
                    if image_name not in kdata.global_features[global_features_type]]
        if len(imgs) == 0:
            print('All global features are already extracted')
            return

        dataset = ImageList(img_list_path=None, root=root, imgs=imgs)

        print(f'\nEvaluation on {dataset}')
        # extract DB feats
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bdescs = []
        trfs_list = [trfs] if isinstance(trfs, str) else trfs

        for trfs in trfs_list:
            kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size,
                      same_size='Pad' in trfs or 'Crop' in trfs)
            bdescs.append(extract_image_features(dataset, trfs, net, desc="DB", **kw))

        # pool from multiple transforms (scales)
        bdescs = tonumpy(F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1))

        if whiten is not None:
            bdescs = common.whiten_features(bdescs, net.pca, **whiten)  
        end.record()
        torch.cuda.synchronize()
        time_ap_gem.append((start.elapsed_time(end)/1000))

        print('writing extracted global features')
        os.umask(0o002)
        gfeat_dtype = bdescs.dtype
        gfeat_dsize = bdescs.shape[1]
        if global_features_type not in kdata.global_features:
            kdata.global_features[global_features_type] = kapture.GlobalFeatures('dirtorch', gfeat_dtype,
                                                                                 gfeat_dsize, 'L2')
            global_features_config_absolute_path = get_feature_csv_fullpath(kapture.GlobalFeatures,
                                                                            global_features_type,
                                                                            kapture_root_path)
            global_features_to_file(global_features_config_absolute_path, kdata.global_features[global_features_type])
        else:
            assert kdata.global_features[global_features_type].dtype == gfeat_dtype
            assert kdata.global_features[global_features_type].dsize == gfeat_dsize
            assert kdata.global_features[global_features_type].metric_type == 'L2'
        for i in tqdm.tqdm(range(dataset.nimg)):
            image_name = dataset.get_key(i)
            global_feature_fullpath = get_global_features_fullpath(global_features_type, kapture_root_path, image_name,
                                                                   tar_handlers)
            gfeat_i = bdescs[i, :]
            assert gfeat_i.shape == (gfeat_dsize,)
            image_global_features_to_file(global_feature_fullpath, gfeat_i)
            kdata.global_features[global_features_type].add(image_name)
            del gfeat_i

        print('average time for ap-gem :', sum(time_ap_gem)/(bdescs.shape[0]*len(time_ap_gem)))
        del bdescs

        if not global_features_check_dir(kdata.global_features[global_features_type], global_features_type,
                                         kapture_root_path, tar_handlers):
            print('global feature extraction ended successfully but not all files were saved')
        else:
            print('Features extracted.')
   
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--kapture-root', type=str, required=True, help='path to kapture root directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')
    parser.add_argument('--global-features-type', default=None,
                        help='global features type_name, default is basename of checkpoint')

    parser.add_argument('--trfs', type=str, required=False, default='',
                        nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')

    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    parser.add_argument('--whitenp', type=float, default=0.5, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None,
                        help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0,
                        help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)

    if args.global_features_type is None:
        args.global_features_type = os.path.splitext(os.path.basename(args.checkpoint))[0]
        print(f'global_features_type set to {args.global_features_type}')

    net = load_model(args.checkpoint, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = extract_kapture_global_features(args.kapture_root, net, args.global_features_type,
                                          args.trfs, pooling=args.pooling, gemp=args.gemp,
                                          threads=args.threads, whiten=args.whiten)
