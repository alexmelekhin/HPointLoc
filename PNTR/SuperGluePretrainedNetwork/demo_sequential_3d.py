#! /usr/bin/env python3

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
# import pyquaternion
from scipy.spatial.transform import Rotation as sRot
import pycolmap
import json

from models.matching import Matching
from models.utils import (AverageTimer, estimate_pose, angle_error_mat,
                          make_matching_plot, to_homogeneous, read_image)

torch.set_grad_enabled(False)


def write_ply(path, pts, colors):
    assert len(pts) == len(colors)
    invalid = np.isnan(pts).any(1)
    print(f'Writing PLY file to {path}')
    print(f'#invalid: {invalid.sum()}, {invalid.mean()}%')
    pts = pts[~invalid]
    colors = colors[~invalid]
    txt = 'ply\nformat ascii 1.0\n'
    txt += f'element vertex {len(pts)}\n'
    txt += 'property float x\nproperty float y\nproperty float z\n'
    txt += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
    txt += 'end_header'
    for p, c in zip(pts, colors):
        c = np.round(c[:3] * 255).astype(np.uint8)
        txt += '\n' + ' '.join(map(str, p.tolist() + c.tolist()))
    with open(path, 'w') as f:
        f.write(txt)


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def estimate_absolute_pose(kpts_2d, kpts_3d, K, thresh):
    if len(kpts_2d) < 4:
        return None
    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': 640,
        'height': 480,
        'params': [K[0, 0], K[0, 2], K[1, 2]]
    }
    ret = pycolmap.absolute_pose_estimation(
        kpts_2d, kpts_3d, cfg, thresh)
    qw, qx, qy, qz = ret['qvec']
    R = sRot.from_quat([qx, qy, qz, qw]).as_matrix()
    t = ret['tvec']
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    T = np.linalg.inv(T)
    R, t = T[:3, :3], T[:3, 3]
    ret = (R, t, np.array(ret['inliers'])) if ret['success'] else None
    return ret

    kpts_2d = kpts_2d.astype(np.float32).reshape((-1, 1, 2))
    kpts_3d = kpts_3d.astype(np.float32).reshape((-1, 1, 3))
    success, R_vec, t, inlier_idx = cv2.solvePnPRansac(
        kpts_3d, kpts_2d, K, np.array([0., 0, 0, 0]),
        iterationsCount=5000, reprojectionError=thresh,
        flags=cv2.SOLVEPNP_P3P)
    if success:
        inliers = np.zeros(len(kpts_2d), np.bool)
        inliers[inlier_idx[:, 0]] = True

        ret, R_vec, t = cv2.solvePnP(
                kpts_3d[inliers], kpts_2d[inliers], K,
                np.array([0., 0, 0, 0]), rvec=R_vec, tvec=t,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret

        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        w_T_query = np.linalg.inv(query_T_w)
        ret = (w_T_query[:3, :3], w_T_query[:3, 3], inliers)
    else:
        ret = None

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--focal', type=float, required=True)

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
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
        '--sinkhorn_iterations', type=int, default=50,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    # parser.add_argument(
        # '--no_display', action='store_true',
        # help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    with open(Path(opt.input, 'rgb.txt'), 'r') as f:
        frames = [i.split() for i in f.read().split('\n') if i and (i[0] != '#')]
    frames = frames[::opt.skip][:opt.max_length]

    with open(Path(opt.input, 'depth.txt'), 'r') as f:
        depths = {}
        for i in f.read().split('\n'):
            if i and (i[0] != '#'):
                ts, path = i.split(' ')
                depths[float(ts)] = path
    with open(Path(opt.input, 'groundtruth.txt'), 'r') as f:
        poses = {}
        for i in f.read().split('\n'):
            if i and (i[0] != '#'):
                tstq = i.split()
                ts, t, q = tstq[0], tstq[1:4], tstq[4:]
                # q = np.array(list(map(float, q)))
                # T = np.eye(4)
                # T[:3, 3] = np.array(list(map(float, t)))
                # T[:3, :3] = pyquaternion.Quaternion(q).rotation_matrix
                # poses[float(ts)] = T
                T44 = transform44(i.split())
                poses[float(ts)] = T44
                # if (j % 500) == 0:
                    # print(T, T44, T-T44)
                # j += 1

    def associate_ts(ts, dic):
        all_ts = np.array(list(dic.keys()))
        diff = np.abs(float(ts) - all_ts)
        return all_ts[np.argmin(diff)]

    def read_frame(i):
        ts, path = frames[i]
        frame, inp, scales = read_image(
            str(Path(opt.input, path)), device, opt.resize, 0, False)
        ts_depth = associate_ts(ts, depths)
        depth = cv2.imread(
            str(Path(opt.input, depths[ts_depth])), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float) / 5000
        assert depth.shape[:2] == frame.shape[:2]
        return frame, inp, depth, scales

    frame, frame_tensor, depth, scales = read_frame(0)

    h, w = frame.shape[:2]
    K = np.eye(3)
    K[0, 0] = opt.focal / scales[0]
    K[1, 1] = opt.focal / scales[1]
    K[0, 2] = w / 2
    K[1, 2] = h / 2

    def backproject(depth, pts_2d, T):
        d = depth[tuple(np.round(pts_2d).astype(int).T)[::-1]]
        valid = d > 0
        pts_3d = (to_homogeneous(pts_2d) @ np.linalg.inv(K).T)*d[:, None]
        pts_3d = (pts_3d @ T[:3, :3].T) + T[:3, 3][None]
        pts_3d[~valid] = None
        return pts_3d

    all_pts_3d = []
    all_pts_color = []
    viz_poses = []
    is_ref_frame = []

    pred = matching.superpoint({'image': frame_tensor})
    pred = {k+'1': v for k, v in pred.items()}

    def update_reference(i):
        last_data = {k+'0': pred[k+'1'] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_id = i
        num = len(last_data['keypoints0'][0])
        colors = cm.hsv(np.random.rand(num))
        T = poses[associate_ts(frames[i][0], poses)]
        pts_3d = backproject(
            depth, last_data['keypoints0'][0].cpu().numpy(), T)
        write_ply(
            Path(opt.output_dir, f'model_{i}.ply'), pts_3d, colors)
        return last_data, last_frame, last_id, colors, pts_3d, T

    last_data, last_frame, last_id, colors, pts_3d, last_T = update_reference(0)
    all_pts_3d.append(pts_3d)
    all_pts_color.append(colors)
    viz_poses.append(last_T)
    is_ref_frame.append(True)

    # frame_tensor = frame2tensor(frame, device)
    # last_data = matching.superpoint({'image': frame_tensor})
    # last_data = {k+'0': last_data[k] for k in keys}
    # last_data['image0'] = frame_tensor
    # last_frame = frame
    # last_image_id = 0
    # kp_colors = cm.hsv(np.random.rand(len(last_data['keypoints0'][0])))

            # last_data = {k+'0': pred[k+'1'] for k in keys}
            # last_data['image0'] = frame_tensor
            # last_frame = frame
            # last_image_id = (vs.i - 1)
            # num = len(last_data['keypoints0'][0])
            # kp_colors = cm.hsv(np.random.rand(len(last_data['keypoints0'][0])))

    timer = AverageTimer()

    for i in range(1, len(frames)):
        frame, frame_tensor, depth, scales = read_frame(i)

    # while True:
        # frame, ret = vs.next_frame()
        # if not ret:
            # print('Finished demo_sequential.py')
            # break
        timer.update('data')
        # stem0, stem1 = last_image_id, vs.i - 1
        stem0, stem1 = last_id, i

        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = colors[valid]

        mkpts_3d = pts_3d[valid]
        has_depth = ~np.any(np.isnan(mkpts_3d), 1)
        mkpts_3d = mkpts_3d[has_depth]
        mkpts_2d = mkpts1[has_depth]
        ret = estimate_absolute_pose(mkpts_2d, mkpts_3d, K, 4)
        # ret = estimate_pose(mkpts0, mkpts1, K, K, 1.)

        text = [
            'SuperGlue',
            'Matches: {}'.format(len(mkpts0))
        ]
        small_text = [
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]

        if opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
        else:
            out_file = None

        if (stem1 - stem0) == 1:
            line_alpha = 0.8
            line_width = 1.5
        else:
            line_alpha = 0.2
            line_width = 0
        out = make_matching_plot(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=out_file, show_keypoints=opt.show_keypoints,
            small_text=small_text, lw=line_width, psm=15, lalpha=line_alpha)

        fail = True
        if ret is not None:
            R, t, inliers = ret
            num_inliers = np.sum(inliers.astype(int))
            if num_inliers > 10:
                fail = False
            T_gt = poses[associate_ts(frames[i][0], poses)]
            dt = np.linalg.norm(t - T_gt[:3, 3])
            dr = angle_error_mat(R, T_gt[:3, :3])
            print(f'dt {dt:.3f}, dr {dr:.3f}, # {num_inliers}')
            if dt > 0.22:
                fail = True

        if fail:
            print('Pose estimation failed, restart')
            # last_data = {k+'0': pred[k+'1'] for k in keys}
            # last_data['image0'] = frame_tensor
            # last_frame = frame
            # last_image_id = (vs.i - 1)
            # num = len(last_data['keypoints0'][0])
            # kp_colors = cm.hsv(np.random.rand(len(last_data['keypoints0'][0])))
            last_data, last_frame, last_id, colors, pts_3d, last_T = update_reference(i)
            all_pts_3d.append(pts_3d)
            all_pts_color.append(colors)
            viz_poses.append(last_T)
            is_ref_frame.append(True)
        else:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            viz_poses.append(T)
            is_ref_frame.append(False)

        # if not opt.no_display:
            # cv2.imshow('SuperGlue matches', out)
            # key = chr(cv2.waitKey(1) & 0xFF)
            # if key == 'q':
                # vs.cleanup()
                # print('Exiting (via q) demo_superglue.py')
                # break
            # elif key == 'n':  # set the current frame as anchor
                # last_data = {k+'0': pred[k+'1'] for k in keys}
                # last_data['image0'] = frame_tensor
                # last_frame = frame
                # last_image_id = (vs.i - 1)
            # elif key in ['e', 'r']:
                # # Increase/decrease keypoint threshold by 10% each keypress.
                # d = 0.1 * (-1 if key == 'e' else 1)
                # matching.superpoint.config['keypoint_threshold'] = min(max(
                    # 0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                # print('\nChanged the keypoint threshold to {:.4f}'.format(
                    # matching.superpoint.config['keypoint_threshold']))
            # elif key in ['d', 'f']:
                # # Increase/decrease match threshold by 0.05 each keypress.
                # d = 0.05 * (-1 if key == 'd' else 1)
                # matching.superglue.config['match_threshold'] = min(max(
                    # 0.05, matching.superglue.config['match_threshold']+d), .95)
                # print('\nChanged the match threshold to {:.2f}'.format(
                    # matching.superglue.config['match_threshold']))
            # elif key == 'k':
                # opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        # if opt.output_dir is not None:
            # #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            # stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            # out_file = str(Path(opt.output_dir, stem + '.png'))
            # print('\nWriting image to {}'.format(out_file))
            # cv2.imwrite(out_file, out)

    # cv2.destroyAllWindows()
    # vs.cleanup()

    if opt.output_dir is not None:
        viz_cam_centers = np.stack([T[:3, 3] for T in viz_poses], 0)
        viz_cam_colors = np.stack([(1, 0, 0, 1) if i else (0, 0, 0, 1) for i in is_ref_frame], 0)
        all_pts_3d += [viz_cam_centers]
        all_pts_color += [viz_cam_colors]
        write_ply(
            Path(opt.output_dir, 'model_all.ply'),
            np.concatenate(all_pts_3d, 0),
            np.concatenate(all_pts_color, 0))

        with open(Path(opt.output_dir, 'trajectory.json'), 'w') as f:
            data = {
                'poses': [T.tolist() for T in viz_poses],
                'is_ref': is_ref_frame,
            }
            json.dump(data, f)
