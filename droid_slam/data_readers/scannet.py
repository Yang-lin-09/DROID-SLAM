import numpy as np
import torch
import glob
import cv2
import os 
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
import re
from .rgbd_utils import *

class ScanNet(RGBDDataset):
    def __init__(self, mode = 'training', datapath = 'datasets/ScanNet', **kwargs):
        
        self.mode = mode
        self.n_frames = 2
        
        super(ScanNet, self).__init__(name = 'ScanNet', datapath = datapath, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        scanid = int(re.findall(r'scene(.+?)_', scene)[0])
        return scanid > 1000
    
    def _build_dataset(self):
        from tqdm import tqdm
        
        print("Building Scannet dataset")
        
        scene_info = {}
        scenes = os.listdir(self.root)[:200]
        
        for scene in tqdm(scenes):
            
            scene_path = osp.join(self.root, scene)
            depth_glob = osp.join(scene_path, 'depth', '*.png')
            depth_list = glob.glob(depth_glob)
            
            
            get_index = lambda x: int(osp.basename(x).split('.')[0])
            get_image = lambda x: osp.join(scene_path, 'color', '%d.jpg' %x)
            get_depth = lambda x: osp.join(scene_path, 'depth', '%d.png' % x)
            get_pose = lambda x: osp.join(scene_path, 'pose', '%d.txt' % x)
            
            indexs = sorted(map(get_index, depth_list))[::2]
            image_list = list(map(get_image, indexs))
            depth_list = list(map(get_depth, indexs))
            
            pose_list = map(get_pose, indexs)
            pose_list = list(map(ScanNet.pose_read, pose_list))
            
            # remove nan poses
            pvecs = np.stack(pose_list, 0)
            keep, = np.where(~np.any(np.isnan(pvecs) | np.isinf(pvecs), axis = 1))

            images = [image_list[i] for i in keep]
            depths = [depth_list[i] for i in keep]
            poses = np.array([pose_list[i] for i in keep])
            
            intri = ScanNet.calib_read(scene_path)
            intris = [intri] * len(images)
            
            graph = self.build_frame_graph(poses, depths, intris)
            
            scene_info[scene] = {
                'images': images, 'depths': depths,
                'poses': poses, 'intrinsics': intris, 'graph': graph
            }
            
        return scene_info
    
    @staticmethod
    def pose_read(pose_file):
        pose = np.loadtxt(pose_file, delimiter = ' ').astype(np.float64)
        return pose_matrix_to_quaternion(pose)
    
    @staticmethod
    def calib_read(scene_path):
        intri_file = osp.join(scene_path, 'intrinsic', 'intrinsic_depth.txt')
        K = np.loadtxt(intri_file, delimiter = ' ')
        return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    
    @staticmethod
    def image_read(image_file):
        image = cv2.imread(image_file)
        return cv2.resize(image, (640, 480))

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 1000.0