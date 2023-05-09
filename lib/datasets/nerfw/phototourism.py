import torch
import torch.utils.data as data
import glob
import numpy as np
import os
import pandas as pd
import pickle
import imageio
from lib.config import cfg
import cv2

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        test_num: number of test images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays
        self.input_ratio, self.test_num, self.use_cache = kwargs['input_ratio'], cfg.task_arg.test_num, cfg.task_arg.use_cache
        self.test_num = max(1, self.test_num) # at least 1
        self.white_bkgd = cfg.task_arg.white_bkgd

        self.read_meta()

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.data_root, '*.tsv'))[0]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.data_root, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.data_root, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            img_data = read_images_binary(os.path.join(self.data_root, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in img_data.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.data_root, f'cache/Ks2.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.data_root, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                H, W = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w, img_h = int(H * self.input_ratio), int(W * self.input_ratio)
                K[0, 0] = cam.params[0] * img_w / H # fx
                K[1, 1] = cam.params[1] * img_h / W # fy
                K[0, 2] = cam.params[2] * img_w / H # cx
                K[1, 2] = cam.params[3] * img_h / W # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.data_root, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = img_data[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (num_imgs, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (num_imgs, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.data_root, 'cache/xyz_world.npy'))
            with open(os.path.join(self.data_root, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.data_root, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.data_root, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)

            # compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far / 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_imgs_train = len(self.img_ids_train)
        self.N_imgs_test = len(self.img_ids_test)

        if self.split == 'train':
            if self.use_cache:
                all_rays = np.load(os.path.join(self.data_root,
                                                f'cache/rays2.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.data_root,
                                                f'cache/rgbs2.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                for id_ in self.img_ids_train:
                    c2w = torch.from_numpy(self.poses_dict[id_]).to(torch.float32)
                    img_path = os.path.join(self.data_root, 'dense/images', self.image_paths[id_])
                    img = imageio.imread(img_path)
                    W, H = img.shape[:2]
                    if self.input_ratio < 1:
                        H = int(H * self.input_ratio)
                        W = int(W * self.input_ratio)
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    img = (np.array(img) / 255.).astype(np.float32)
                    img = torch.from_numpy(img) # (h, w, 3)
                    img = img.reshape(-1, 3) # (h*w, 3)
                    self.all_rgbs += [img]

                    directions = get_ray_directions(H, W, self.Ks[id_])
                    ray_o, ray_d = get_rays(directions, c2w)
                    ray_at = id_ * torch.ones(len(ray_o), 1)

                    self.all_rays += [torch.cat([ray_o, ray_d,
                                                self.nears[id_]*torch.ones_like(ray_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(ray_o[:, :1]),
                                                ray_at],
                                                1)] # (h*w, 9)

                self.all_rays = torch.cat(self.all_rays, 0) # ((num_imgs-1) * h * w, 9)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((num_imgs-1) * h * w, 3)

        elif self.split in ['val', 'test']:
            self.test_id = self.img_ids_train[0] # use the first image to test

        else:
            pass

    def __getitem__(self, index):
        ret = {}

        if self.split == 'train':
            rays = self.all_rays[index, :8]
            ts = self.all_rays[index, 8].long()
            rgbs = self.all_rgbs[index]

        elif self.split in ['test', 'val']:
            if self.split == 'test':
                id_ = self.test_id
            else:
                id_ = self.img_ids_train[index]

            ret['c2w'] = c2w = torch.from_numpy(self.poses_dict[id_]).to(torch.float32)
            img_path = os.path.join(self.data_root, 'dense/images', self.image_paths[id_])
            img = imageio.imread(img_path)
            W, H = img.shape[:2]
            if self.input_ratio < 1:
                H = int(H * self.input_ratio)
                W = int(W * self.input_ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = (np.array(img) / 255.).astype(np.float32)
            img = torch.from_numpy(img).reshape(-1, 3)

            directions = get_ray_directions(H, W, self.Ks[id_])
            ray_o, ray_d = get_rays(directions, c2w)

            rays = torch.cat([ray_o, ray_d,
                                    self.nears[id_]*torch.ones_like(ray_o[:, :1]),
                                    self.fars[id_]*torch.ones_like(ray_o[:, :1]),],
                                    1) # (h*w, 8)
            ts = id_ * torch.ones(len(rays), dtype=torch.long) # (h*w, )
            rgbs = img # (h*w, 3)
            ret.update({'meta':
                {
                    'H': H,
                    'W': W,
                    'N_rays': self.batch_size,
                    'id': id_
                }
            })

        ret['rays'] = rays
        ret['ts'] = ts
        ret['rgbs'] = rgbs

        return ret

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test':
            return self.test_num
        if self.split == 'val':
            return self.N_imgs_train
        return len(self.poses_test)
