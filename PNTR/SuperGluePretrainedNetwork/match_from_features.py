import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py

from models.superglue import SuperGlue
# from models.nearest_neighbor_matcher import NearestNeighborMatcher
from models.utils import read_image

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=50,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--pair_list', type=str, required=False,
        help='Path to the list of image pairs')
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--image_glob', type=str, default='jpg')
    parser.add_argument(
        '--feature_dir', type=str, required=True,
        help='Path to the directory that contains the extracted features')
    parser.add_argument(
        '--results_dir', type=str, required=True,
        help='Path to the directory in which the .npz results are written')
    parser.add_argument(
        '--hdf5', type=str)

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device {}'.format(device))
    config = {
        'weights': opt.superglue,
        'sinkhorn_iterations': opt.sinkhorn_iterations,
        'match_threshold': opt.match_threshold,
    }
    matcher = SuperGlue(config).eval().to(device)
    # matcher = NearestNeighborMatcher({'mutual_check': True}).eval()

    image_dir = Path(opt.image_dir)
    results_dir = Path(opt.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    if opt.hdf5:
        hfile = h5py.File(str(results_dir / opt.hdf5), 'w')
        feat_file = h5py.File(opt.feature_dir, 'r')

    if opt.pair_list:
        print('Matching from pair list')
        with open(opt.pair_list, 'r') as f:
            pairs = [l.split() for l in f.readlines()]
    else:
        print('Exhaustive matching of all images')
        images = image_dir.rglob(f'*.{opt.image_glob}')
        images = sorted([str(p.relative_to(image_dir)) for p in images])
        pairs = []
        # Ordered by (largest key, smallest key)
        for i in range(len(images)):
            for j in range(i):
                pairs.append((images[i], images[j]))

    for pair in tqdm(pairs):
        name0, name1 = pair[:2]
        image0, inp0, scales0 = read_image(
            image_dir / name0, device, [-1], 0, True, False)
        image1, inp1, scales1 = read_image(
            image_dir / name1, device, [-1], 0, True, False)

        feats = {}
        if opt.hdf5:
            feats['keypoints0'] = feat_file[name0]['keypoints'].__array__()
            feats['descriptors0'] = feat_file[name0]['descriptors'].__array__()
            feats['scores0'] = feat_file[name0]['scores'].__array__()
            feats['keypoints1'] = feat_file[name1]['keypoints'].__array__()
            feats['descriptors1'] = feat_file[name1]['descriptors'].__array__()
            feats['scores1'] = feat_file[name1]['scores'].__array__()
        else:
            with np.load(str(Path(opt.feature_dir, name0+'.npz'))) as npz:
                feats['keypoints0'] = npz['keypoints']
                feats['descriptors0'] = npz['descriptors']
                feats['scores0'] = npz['scores']
            with np.load(str(Path(opt.feature_dir, name1+'.npz'))) as npz:
                feats['keypoints1'] = npz['keypoints']
                feats['descriptors1'] = npz['descriptors']
                feats['scores1'] = npz['scores']
        feats = {k: torch.from_numpy(v)[None].float().to(device=device)
                 for k, v in feats.items()}

        # Perform the matching
        pred = matcher({'image0': inp0, 'image1': inp1, **feats})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        out = {'matches': pred['matches0'],
               'match_confidence': pred['matching_scores0']}

        pair_id = '{}-{}'.format(
            name0.replace('/', '_'), name1.replace('/', '_'))
        if opt.hdf5:
            grp = hfile.create_group(pair_id)
            for k, v in out.items():
                grp.create_dataset(k, data=v)
        else:
            np.savez(str(results_dir / (pair_id + '.npz')), **out)

    if opt.hdf5:
        hfile.close()
