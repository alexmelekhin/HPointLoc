import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import cv2

from models.superpoint import SuperPoint
from models.utils import read_image

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--quadratic_refinement', action='store_true')
    parser.add_argument(
        '--refinement_radius', type=int, default=0)
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers,'
             ' resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--no_resize_force', action='store_true')

    parser.add_argument(
        '--image_dir', type=str, required=True)
    parser.add_argument(
        '--image_glob', type=str, default='jpg')
    parser.add_argument(
        '--results_dir', type=str, required=True)
    parser.add_argument(
        '--hdf5', type=str)

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0])
              + (' (only if larger)' if opt.no_resize_force else ''))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    image_list = Path(opt.image_dir).rglob(f'*.{opt.image_glob}')
    image_list = [p.relative_to(opt.image_dir) for p in image_list]
    assert len(image_list) > 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device {}'.format(device))
    config = {
        'max_keypoints': opt.max_keypoints,
        'keypoint_threshold': opt.keypoint_threshold,
        'nms_radius': opt.nms_radius,
        'refinement_radius': opt.refinement_radius,
        'do_quadratic_refinement': opt.quadratic_refinement,
    }
    frontend = SuperPoint(config).eval().to(device)

    results_dir = Path(opt.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    if opt.hdf5:
        hfile = h5py.File(str(results_dir / opt.hdf5), 'w')

    for name in tqdm(image_list):
        image, inp, scales = read_image(
            opt.image_dir / name, device, opt.resize, 0, True,
            resize_force=not opt.no_resize_force, interp=cv2.INTER_CUBIC)

        pred = frontend({'image': inp})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts = (pred['keypoints'] + .5) * np.array([scales]) - .5

        out = {'keypoints': kpts, 'descriptors': pred['descriptors'],
               'scores': pred['scores']}

        if opt.hdf5:
            grp = hfile.create_group(str(name))
            for k, v in out.items():
                grp.create_dataset(k, data=v)
        else:
            path = results_dir / '{}.npz'.format(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(path), **out)

    if opt.hdf5:
        hfile.close()
