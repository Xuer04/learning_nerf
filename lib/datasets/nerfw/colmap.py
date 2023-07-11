from lib.config import cfg
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import open3d as o3d
from kornia import create_meshgrid

from .ray_utils import *
from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

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
        self.vis_depth = False
        self.transform = T.ToTensor()
        self.read_meta()

    def get_colmap_depth(
        self,
        img_p3d_all,
        img_2d_all,
        img_err_all,
        pose,
        intrinsic,
        img_w,
        img_h,
        device=0,
    ):
        # return depth and weights for each image
        # calculate normalize factor
        grid = create_meshgrid(
            img_h, img_w, normalized_coordinates=False, device=device
        )[0]
        i, j = grid.unbind(-1)
        fx, fy, cx, cy = (
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        directions = torch.stack(
            [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1
        )  # (H, W, 3)

        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ pose[:, :3].T  # (H, W, 3)
        dir_norm = torch.norm(rays_d, dim=-1, keepdim=True).reshape(img_h, img_w)

        # depth from sfm key points
        depth_all = torch.zeros(img_h, img_w, device=device)
        weights_all = torch.zeros(img_h, img_w, device=device)

        img_2d_all = torch.round(img_2d_all).long()  # (width, height)
        valid_mask = (
            (img_2d_all[:, 0] >= 0)
            & (img_2d_all[:, 0] < img_w)
            & (img_2d_all[:, 1] >= 0)
            & (img_2d_all[:, 1] < img_h)
        )

        img_2d = img_2d_all[valid_mask]
        img_err = img_err_all[valid_mask].squeeze()

        img_p3d = img_p3d_all[valid_mask]
        pose = torch.cat((pose, torch.zeros(1, 4, device=device)), dim=0)
        pose[3, 3] = 1
        extrinsic = torch.linalg.inv(pose)

        Err_mean = torch.mean(img_err)
        projected = intrinsic @ extrinsic[:3] @ img_p3d.T

        depth = projected[2, :]
        weight = 2 * torch.exp(-((img_err / Err_mean) ** 2))

        depth_all[img_2d[:, 1], img_2d[:, 0]] = depth
        weights_all[img_2d[:, 1], img_2d[:, 0]] = weight
        return depth_all.cpu() * dir_norm.cpu(), weights_all.cpu()

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.data_root, '*.tsv'))[0]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # read the id from images.bin using image file name!
        if self.use_cache:
            # with open(os.path.join(self.data_root, f'cache/img_ids.pkl'), 'rb') as f:
            #     self.img_ids = pickle.load(f)
            # with open(os.path.join(self.data_root, f'cache/image_paths.pkl'), 'rb') as f:
            #     self.image_paths = pickle.load(f)
            pass
        else:
            print("Reading images.bin..")
            imgdata = read_images_binary(os.path.join(self.data_root, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imgdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                if filename not in img_path_to_id.keys():
                    print(f"image {filename} not found in sfm result!!")
                    continue
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # read and rescale camera intrinsics
        if self.use_cache:
            # with open(os.path.join(self.data_root, f'cache/Ks2.pkl'), 'rb') as f:
            #     self.Ks = pickle.load(f)
            pass
        else:
            self.Ks = {} # {id: K}
            print("Reading cameras.bin..")
            camdata = read_cameras_binary(os.path.join(self.data_root, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[imgdata[id_].camera_id]
                # cam = camdata[id_]
                if cam.model == "PINHOLE":
                    img_w, img_h = int(cam.params[2] * 2), int(cam.params[3] * 2)
                    img_w_, img_h_ = (
                        img_w * self.input_ratio,
                        img_h * self.input_ratio,
                    )
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                    K[1, 1] = cam.params[1] * img_h_ / img_h  # fy
                    K[0, 2] = cam.params[2] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[3] * img_h_ / img_h  # cy
                    K[2, 2] = 1
                elif cam.model == "SIMPLE_RADIAL":
                    img_w, img_h = int(cam.params[1] * 2), int(cam.params[2] * 2)
                    img_w_, img_h_ = (
                        img_w * self.input_ratio,
                        img_h * self.input_ratio,
                    )
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # f
                    K[1, 1] = cam.params[0] * img_h_ / img_h  # f
                    K[0, 2] = cam.params[1] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[2] * img_h_ / img_h  # cy
                    K[2, 2] = 1
                else:
                    raise NotImplementedError(f"Not supported camera model {cam.model}")
                self.Ks[id_] = K

        # read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            # self.poses = np.load(os.path.join(self.data_root, 'cache/poses.npy'))
            pass
        else:
            print("Compute c2w poses..")
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imgdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (num_imgs, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (num_imgs, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            # from OpenCV to OpenGL
            self.poses[..., 1:3] *= -1

        # correct scale
        if self.use_cache:
            # self.xyz_world = np.load(os.path.join(self.data_root, 'cache/xyz_world.npy'))
            # with open(os.path.join(self.data_root, f'cache/nears.pkl'), 'rb') as f:
            #     self.nears = pickle.load(f)
            # with open(os.path.join(self.data_root, f'cache/fars.pkl'), 'rb') as f:
            #     self.fars = pickle.load(f)
            pass
        else:
            pts3d = read_points3d_binary(os.path.join(self.data_root, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate(
                [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1
            )

            # compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()

            # TODO: set scale_factor
            scale_factor = max_far / 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor

        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [
            id_ for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, 'split']=='train'
        ]
        self.img_ids_test = [
            id_ for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, 'split']=='test'
        ]
        self.N_imgs_train = len(self.img_ids_train)
        self.N_imgs_test = len(self.img_ids_test)

        if self.split in ["train", "eval"]:
            self.all_rays = []
            self.all_rgbs = []
            if self.use_cache:
                print(f"Loading cached rays..")
                all_rays = np.load(os.path.join(
                    self.data_root,
                    f'cache/rays2.npy')
                )
                self.all_rays += [torch.from_numpy(all_rays["arr_0"])]
                all_rgbs = np.load(os.path.join(
                    self.data_root,
                    f'cache/rgbs2.npy')
                )
                self.all_rgbs += [torch.from_numpy(all_rgbs["arr_0"])]
                self.all_rays = torch.cat(self.all_rays, 0)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)
            else:
                print("Generating rays and rgbs..")
                if self.split == "eval":
                    self.img_wh = []
                    self.eval_images = []
                    self.extrinsics = []
                    self.intrinsics = []
                    self.test_id = self.img_ids_train[0] # use only one image to test
                pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
                error_array = torch.ones(max(pts3d.keys()) + 1, 1)
                for pts_id, pts in tqdm(pts3d.items()):
                    pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
                    error_array[pts_id, 0] = torch.from_numpy(pts.error)
                print("Mean Projection Error:", torch.mean(error_array))

                for id_ in tqdm(self.img_ids_train):
                    # c2w = torch.from_numpy(self.poses_dict[id_]).to(torch.float32)
                    # img_path = os.path.join(self.data_root, 'dense/images', self.image_paths[id_])
                    # img = imageio.imread(img_path)
                    # W, H = img.shape[:2]
                    # if self.input_ratio < 1:
                    #     H = int(H * self.input_ratio)
                    #     W = int(W * self.input_ratio)
                    #     img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    # img = (img / 255.).astype(np.float32)
                    # self.all_rgbs += [img]

                    # directions = get_ray_directions(H, W, self.Ks[id_])
                    # ray_o, ray_d = get_rays(directions, c2w)
                    # ray_at = id_ * torch.ones(len(ray_o), 1)

                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(
                        os.path.join(
                            self.data_root, "dense/images", self.image_paths[id_]
                        )
                    ).convert("RGB")
                    img_w, img_h = img.size
                    if self.input_ratio < 1:
                        img_w = int(img_w * self.input_ratio)
                        img_h = int(img_h * self.input_ratio)
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    if self.split == "eval":
                        self.img_wh += [torch.LongTensor([img_w, img_h])]
                        self.eval_images += [self.image_paths[id_]]
                        self.extrinsics += [c2w]
                        self.intrinsics += [self.Ks[id_]]
                    img = self.transform(img)  # (3, h, w)
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                    self.all_rgbs += [img]

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_at = id_ * torch.ones(len(rays_o), 1)

                    # fix pose
                    img_colmap = imgdata[id_]
                    pose = torch.FloatTensor(self.poses_dict[id_]).cuda()
                    pose[..., 1:3] *= -1
                    intrinsic = torch.FloatTensor(self.Ks[id_]).cuda()

                    valid_3d_mask = img_colmap.point3D_ids != -1
                    point3d_ids = img_colmap.point3D_ids[valid_3d_mask]
                    img_p3d = pts3d_array[point3d_ids].cuda()
                    img_err = error_array[point3d_ids].cuda()
                    img_2d = torch.from_numpy(img_colmap.xys)[valid_3d_mask] * self.input_ratio
                    depth_sfm, weight = self.get_colmap_depth(
                        img_p3d, img_2d, img_err, pose, intrinsic, img_w, img_h
                    )
                    depths = depth_sfm.reshape(-1, 1)
                    weights = weight.reshape(-1, 1)

                    image_name = self.image_paths[id_].split(".")[0]

                    if self.vis_depth:
                        print(f"saving... at results/depth/{image_name}.ply")
                        model_dir = os.path.join(cfg.result_dir, 'depth')
                        os.system('mkdir -p {}'.format(model_dir))
                        model_path = os.path.join(model_dir, f"{image_name}.ply")
                        pts = rays_o + rays_d * depths
                        gt_pcd = o3d.geometry.PointCloud()
                        gt_pcd.points = o3d.utility.Vector3dVector(pts.numpy())
                        o3d.io.write_point_cloud(
                            model_path, gt_pcd
                        )

                    self.all_rays += [torch.cat([
                        rays_o,
                        rays_d,
                        self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                        self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                        rays_at
                    ],1)] # (h*w, 9)

                self.all_rays = torch.cat(self.all_rays, 0) # ((num_imgs-1) * h * w, 9)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((num_imgs-1) * h * w, 3)


        elif self.split in ['val', 'test']:
            self.test_id = self.img_ids_train[0] # use only one image to test

        else:
            pass

    def __getitem__(self, index):
        ret = {}

        if self.split == 'train':
            # rays = self.all_rays[index][:, :8] # (h*w, 8)
            # ts = self.all_rays[index][:, 8].long().reshape(-1, 1) # (h*w, 1)
            # rgbs = self.all_rgbs[index]
            # rgbs = torch.from_numpy(rgbs).reshape(-1, 3) # (h*w, 3)
            # select_idx = np.random.choice(rays.shape[0], self.batch_size, replace=False)
            # rays = rays[select_idx] # (N_rays, 8)
            # ts = ts[select_idx] # (N_rays, 1)
            # rgbs = rgbs[select_idx] # (N_rays, 3)

            ret = {
                'rays': self.all_rays[index, :8],
                'ts': self.all_rays[index, 8].long(),
                'rgbs': self.all_rgbs[index],
            }

        elif self.split == 'val':
            id_ = self.img_ids_train[index]

            # ret['c2w'] = c2w = torch.from_numpy(self.poses_dict[id_]).to(torch.float32)
            # img_path = os.path.join(self.data_root, 'dense/images', self.image_paths[id_])
            # img = imageio.imread(img_path)
            # W, H = img.shape[:2]
            # if self.input_ratio < 1:
            #     H = int(H * self.input_ratio)
            #     W = int(W * self.input_ratio)
            #     img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            # img = (np.array(img) / 255.).astype(np.float32)
            # img = torch.from_numpy(img).reshape(-1, 3)

            # directions = get_ray_directions(H, W, self.Ks[id_])
            # ray_o, ray_d = get_rays(directions, c2w)

            # rays = torch.cat([ray_o, ray_d,
            #                         self.nears[id_]*torch.ones_like(ray_o[:, :1]),
            #                         self.fars[id_]*torch.ones_like(ray_o[:, :1]),],
            #                         1) # (h*w, 8)
            # ts = id_ * torch.ones(len(rays), dtype=torch.long) # (h*w, )
            # rgbs = img # (h*w, 3)

            w, h = self.img_wh[index]
            all_rays = self.all_rays[index].reshape(h, w, -1)
            all_rgbs = self.all_rgbs[index].reshape(h, w, -1)
            left_rays = all_rays[:, : w // 2].reshape(-1, 9)
            right_rays = all_rays[:, w // 2 :].reshape(-1, 9)
            left_rgbs = all_rgbs[:, : w // 2].reshape(-1, 3)
            right_rgbs = all_rgbs[:, w // 2 :].reshape(-1, 3)
            ret = {
                'rays': self.all_rays[index][:, :8],
                'ts': self.all_rays[index][:, 8].long(),
                'rgbs': self.all_rgbs[index],
                'rays_train': left_rays[:, :8],
                'ts_train': left_rays[:, 8].long(),
                'rgbs_train_gt': left_rgbs,
                'rays_eval': right_rays[:, :8],
                'ts_eval': right_rays[:, 8].long(),
                'rgbs_eval_gt': right_rgbs,
                'extrinsic': self.extrinsics[index],
                'intrinsic': self.intrinsics[index]
            }
            ret.update({
                'meta': {
                    'image_name': self.eval_images[index],
                    'id': id_,
                    'H': h,
                    'W': w
                }
            })
        elif self.split == "test":
            id_ = self.test_id
            ret["c2w"] = c2w = torch.FloatTensor(self.poses_dict[id_])
            img = Image.open(
                os.path.join(
                    self.data_root, "dense/images", self.image_paths[id_]
                )
            ).convert("RGB")
            img_w, img_h = img.size
            if self.input_ratio < 1:
                img_w = int(img_w * self.input_ratio)
                img_h = int(img_h * self.input_ratio)
                img = img.resize((img_w, img_h), Image.LANCZOS)
            w, h = img_w, img_h
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            ret["rgbs"] = img

            image_name = self.image_paths[id_].split(".")[0]
            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                    self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )
            ret['rays'] = rays
            ret['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            ret['K'] = self.Ks[id_]
            ret.update({
                'meta': {
                    'image_name': image_name,
                    'id': id_,
                    'H': h,
                    'W': w
                }
            })

        return ret

    def __len__(self):
        if self.split in ['train', 'eval']:
            return len(self.all_rays)
        if self.split == 'test':
            return self.test_num
        if self.split == 'val':
            return self.N_imgs_train
        return len(self.poses_test)
