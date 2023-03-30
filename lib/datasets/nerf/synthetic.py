import torch.utils.data as data
import torch
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.data_root = os.path.join(data_root, scene)
        self.input_ratio = kwargs['input_ratio']
        self.split = split # train or test
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.num_iter_train = 0
        self.use_batching = not cfg.task_arg.no_batching
        # cams = kwargs['cams']
        self.precrop_iters = cfg.task_arg.precrop_iters
        self.precrop_frac = cfg.task_arg.precrop_frac
        self.batch_size = cfg.task_arg.N_rays
        self.use_single_view = cfg.train.single_view

        # read all images and poses
        imgs = []
        poses = []
        # json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            img_path = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            imgs.append(imageio.imread(img_path))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # (num_imgs, 800, 800, 4)
        poses = np.array(poses).astype(np.float32)  # (num_imgs, 4, 4)

        self.num_imgs = imgs.shape[0]

        # get inner arguments of camera
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(json_info['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        if self.input_ratio != 1.:
            H = int(H // 2)
            W = int(W // 2)
            focal = focal / 2.
            imgs_half = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half
        # whether use white background
        if self.white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:]) # (..., 4) -> (..., 3)
        else:
            imgs = imgs[..., :3]

        self.imgs = torch.from_numpy(imgs)
        self.poses = torch.from_numpy(poses)
        self.H = H
        self.W = W
        self.focal = focal
        # Simple Pinhole Camera Model
        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])

        # set args for center crop
        dH = int(self.H // 2 * self.precrop_frac)
        dW = int(self.W // 2 * self.precrop_frac)
        self.coords_center = torch.stack(
            torch.meshgrid(
                torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW), indexing='ij',
            ), -1
        )
        self.coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, self.H - 1, H),
                torch.linspace(0, self.W - 1, W), indexing='ij'
            ), -1
        )

        rays_o, rays_d = [], []

        for i in range(self.num_imgs):
            ray_o, ray_d = self.get_rays(self.H, self.W, self.K, self.poses[i, :3, :4])
            rays_d.append(ray_d)   # (H, W, 3)
            rays_o.append(ray_o)   # (H, W, 3)

        self.rays_o = torch.stack(rays_o)                    # (num_imgs, H, W, 3)
        self.rays_d = torch.stack(rays_d)                    # (num_imgs, H, W, 3)
        # self.imgs = self.imgs.reshape(self.num_imgs, -1, 3)  # (num_imgs, H * W, 3)


    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典(添加 'meta' 用于 evaluate)
        """
        index = 0 if self.use_single_view else index

        if self.split == 'train':
            self.num_iter_train += 1

            ray_os = self.rays_o[index]  # (H, W, 3)
            ray_ds = self.rays_d[index]  # (H, W, 3)
            rgbs = self.imgs[index]      # (H, W, 3)

            coords = self.coords_center if self.num_iter_train < self.precrop_iters else self.coords
            coords = torch.reshape(coords, [-1, 2])
            select_ids = np.random.choice(coords.shape[0], size=self.batch_size, replace=False)
            select_coords = coords[select_ids].long()

            ray_o = ray_os[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
            ray_d = ray_ds[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
            batch_rays = torch.stack([ray_o, ray_d], 0)               # (2, N_rays, 3)
            rgb = rgbs[select_coords[:, 0], select_coords[:, 1]]      # (N_rays, 3)

        else:
            ray_o = self.rays_o[index].reshape(-1, 3)  # (H * W, 3)
            ray_d = self.rays_d[index].reshape(-1, 3)  # (H * W, 3)
            rgb = self.imgs[index].reshape(-1, 3)      # (H * W, 3)

        ret = {'ray_o': ray_o, 'ray_d': ray_d, 'rgb': rgb}
        ret.update({'meta':
            {
                'H': self.H,
                'W': self.W,
                'ratio': self.input_ratio,
                'N_rays': self.batch_size
            }
        })
        return ret


    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        return self.num_imgs


    def get_rays(self, H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
        i, j = i.t(), j.t()
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d
