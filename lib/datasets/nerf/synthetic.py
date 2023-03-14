import torch.utils.data as data
import torch
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
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
        self.num_iter = 0
        self.use_batching = not cfg.task_arg.no_batching
        cams = kwargs['cams']
        self.batch_size = cfg.task_arg.N_rays
        self.precrop_iters = cfg.task_arg.precrop_iters
        self.precrop_frac = cfg.task_arg.precrop_frac

        # read all images and poses
        imgs = []
        poses = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(split))))
        # json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            img_path = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            imgs.append(imageio.imread(img_path))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        self.num_imgs = imgs.shape[0]

        # get inner arguments of camera
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(json_info['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        if self.input_ratio != 1.:
            H = H // 2
            W = W // 2
            focal = focal / 2.
            imgs_half = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half
        # adjust images according to the opacity
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:]) # (..., 4) -> (..., 3)

        self.imgs = torch.from_numpy(imgs)
        self.poses = torch.from_numpy(poses)
        self.H = H
        self.W = W
        self.focal = focal

        # set t_x, t_y
        X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        self.t_x = (X - self.W * 0.5) / self.focal
        self.t_y = (Y - self.H * 0.5) / self.focal

        # set args for center crop
        self.precrop_index = torch.arange(self.W * self.H).view(self.H, self.W)
        dH = int(self.H // 2 * self.precrop_frac)
        dW = int(self.W // 2 * self.precrop_frac)
        self.precrop_index = self.precrop_index[self.H // 2 - dH:self.H // 2 + dH, self.W // 2 - dW:self.W // 2 + dW].reshape(-1)

        rays_o, rays_d = [], []

        for i in range(self.num_imgs):
            ray_o, ray_d = self.get_rays(self.t_x, self.t_y, self.poses[i])
            rays_d.append(ray_d)
            rays_o.append(ray_o)

        self.rays_o = torch.stack(rays_o)
        self.rays_d = torch.stack(rays_d)

        self.imgs = self.imgs.view(self.num_imgs, -1, 3)


    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            None
        """
        if self.split == 'train':
            ray_ds = self.rays_d[index]
            ray_os = self.rays_o[index]
            img_rgbs = self.imgs[index]
            self.num_iter += 1
            if self.num_iter < self.precrop_iters:
                ray_ds = ray_ds[self.precrop_index]
                ray_os = ray_os[self.precrop_index]
                img_rgbs = img_rgbs[self.precrop_index]
            select_ids = np.random.choice(ray_ds.shape[0], self.batch_size, replace=False)
            ray_d = ray_ds[select_ids]
            ray_o = ray_os[select_ids]
            img_rgb = img_rgbs[select_ids]
        else:
            ray_d = self.rays_d
            ray_o = self.rays_o
            img_rgb = self.imgs

        ret = {'ray_o': ray_o, 'ray_d': ray_d, 'img_rgb': img_rgb}
        ret.update({'meta':{'H': self.H, 'W': self.W, 'focal': self.focal}})
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


    def get_rays(self, X, Y, pose):
        dirs = torch.stack([X, -Y, -torch.ones_like(X)])
        c2w = pose[:3, :3]

        ray_d = dirs.view(-1, 3) @ c2w.T
        ray_o = c2w[:3, -1].expand(ray_d.shape)
        return ray_o, ray_d
