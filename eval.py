import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import numpy as np 
from PIL import Image
import yaml
import argparse
import datetime
import glob 
from tqdm import tqdm 
import cv2

from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.helpers.model_helper import build_model
from lib.datasets.kitti.kitti_utils import Calibration

from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', type=str, default="configs/monodetr.yaml", help='settings of detection in yaml format')
parser.add_argument('--checkpoint', type=str, default="checkpoint_best_2.pth", help='the checkpoint path')
parser.add_argument('--calib', type=str, default="/home/spurs/dataset/2011_10_03/calib_cam_to_cam.txt", help='the path of camera calibration')
parser.add_argument('--datadir', type=str, default="/home/spurs/dataset/2011_10_03/2011_10_03_drive_0047_sync/image_02/data", help='the path of the dataset')
args = parser.parse_args()


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def preprocess(img_path):
    # basic configuration
    num_classes = 3
    max_objs = 50
    class_name = ['Pedestrian', 'Car', 'Cyclist']
    cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
    resolution = np.array([1280, 384])  # W * H
    # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
    downsample = 32

    # statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                    [1.52563191462 ,1.62856739989, 3.88311640418],
                                    [1.73698127    ,0.59706367   , 1.76282397   ]])
    

    img = Image.open(img_path)    # (H, W, 3) RGB mode
    img_ori = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_size = np.array(img.size)
    features_size = resolution // downsample    # W * H

    # data augmentation for image
    center = np.array(img_size) / 2
    crop_size, crop_scale = img_size, 1

    # add affine transformation for 2d images.
    trans, trans_inv = get_affine_transform(center, crop_size, 0, resolution, inv=1)
    img = img.transform(tuple(resolution.tolist()),
                        method=Image.AFFINE,
                        data=tuple(trans_inv.reshape(-1).tolist()),
                        resample=Image.BILINEAR)

    # image encoding
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # C * H * W
    img = torch.tensor(img)
    img = torch.unsqueeze(img, 0) # 1 * C * H * W

    img_size = torch.tensor(img_size)
    img_size = torch.unsqueeze(img_size, 0)

    info = {'img': img,
            'img_size': img_size,
            'img_id': torch.tensor([0]),
            'bbox_downsample_ratio': img_size / features_size
    }
    
    return info, img_ori, class_name


def main():
    print(args.config)

    #assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint
    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, device)
    epoch = checkpoint.get('epoch', -1)
    best_result = checkpoint.get('best_result', 0.0)
    best_epoch = checkpoint.get('best_epoch', 0.0)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    calib_file = args.calib
    calib = read_calib_file(calib_file)['P_rect_02'].reshape(3, 4)
    #print(f"calib={calib}")
    calibs = Calibration(calib)

    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=None,
                    logger=None,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    
    img_paths = list(glob.glob(os.path.join(args.datadir, '*png')))
    img_paths.sort()


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = 'output.mp4'
    fps = 10
    videoWriter = None 
    color = (0, 255, 0)
    thickness = 1
    for img_path in tqdm(img_paths):
        info, img, class_name = preprocess(img_path)
        inputs = info['img']
        # [cls_id, alpha, bbox, dimensions, locations, ry, score]
        dets = tester.inferenceV2(inputs, calibs, info)[0]
        for i in range(len(dets)):
            cls_id = dets[i][0]
            bbox = [int(it+0.5) for it in dets[i][2:6]]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, class_name[cls_id], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, thickness, cv2.LINE_AA)

            continue 
            #l, h, w = dets[i][6:9]
            # l, w, h = dets[i][6:9]
            # [1.52563191462 ,1.62856739989, 3.88311640418]
            h, w, l = dets[i][6:9]
            # objects[i].h, objects[i].w, objects[i].l
    
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

            pos = dets[i][9:12]
            ry = dets[i][12]

            #print(f"class={class_name[cls_id]}, h={h}, w={w}, l={l}, pos={pos}")
            R = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
            corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
            corners3d = np.dot(R, corners3d).T # 8 * 3
            corners3d = corners3d + pos # 8 * 3

            corners3d = np.dot(calib[:3, :3], corners3d.T)# 3 * 8

            corners3d[:, 0] = corners3d[:, 0] / corners3d[:, 2]
            corners3d[:, 1] = corners3d[:, 1] / corners3d[:, 2]

            corners3d += 0.5
            corners3d = corners3d.astype(np.int32).T 

            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                cv2.line(img, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(img, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
                i, j = k, k + 4
                cv2.line(img, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)


        if videoWriter is None:
            size = [img.shape[1], img.shape[0]]
            videoWriter = cv2.VideoWriter(output_path, fourcc, fps, size, True)

        videoWriter.write(img)
        cv2.imshow('MonoDETR', img)
        cv2.waitKey(1)

    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
