import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.networks.nerf.nerf_network import NeRF
from lib.config import cfg
from lib.networks.nerf.render import *


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.batch_size = cfg.task_arg.N_rays
        self.chunk = cfg.task_arg.chunk_size
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Device: {self.device}")

        # coarse model
        self.model = NeRF().to(self.device)
        self.grad_vars = list(self.model.parameters())

        # fine model
        self.model_fine = NeRF().to(self.device)
        self.grad_vars.extend(list(self.model_fine.parameters()))

        # encoder
        self.xyz_encoder, self.input_ch = self.model.xyz_encoder, self.model.input_ch
        self.dir_encoder, self.input_ch_views = self.model.dir_encoder, self.model.input_ch_views


    def batchify_rays(self, ray_o, ray_d):
        all_ret = {}
        for i in range(0, self.batch_size, self.chunk):
            ret = render_rays(self.model, self.model_fine, ray_o[i:i + self.chunk], ray_d[i:i + self.chunk], self.N_samples, self.device, self.N_importance, self.white_bkgd)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):
        """
        batch['ray_o']: [1, N_rays, 3]
        batch['ray_d']: [1, N_rays, 3]
        batch['img_rgb']: [1, N_rays, 3]
        """
        B, N_rays, C = batch['ray_o'].shape
        ray_o, ray_d = batch['ray_o'].reshape(-1, C), batch['ray_d'].reshape(-1, C)
        ret = self.batchify_rays(ray_o, ray_d)
        return {k:ret[k].reshape(B, N_rays, -1) for k in ret}
