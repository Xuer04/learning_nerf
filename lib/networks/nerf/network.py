import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.D = net_cfg.nerf.D
        self.W = net_cfg.nerf.W
        self.skips = net_cfg.nerf.skips
        self.use_viewdirs = net_cfg.nerf.use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4
        self.xyz_encoder, self.input_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, self.input_ch_views = get_encoder(net_cfg.dir_encoder)

        self.pts_linears = nn.ModuleList(
        [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in
                                        range(self.D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W // 2)])

        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)


    def batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def render_rays(self, ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False,
                    perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0.,
                    verbose=False, pytest=False):
        pass


    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        pass


    def render(self):
        pass

    def forward(self, x):
        # x (N_rand * N_samples, 90)  90 = 60 + 3 + 24 + 3
        # input_pts (N_rand * N_samples, 63)
        # input_views (N_rand * N_samples, 27)
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.pts_linears):

            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha is related to pts
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # rgb is related to pts and views
            h = torch.cat([feature, input_views], -1)

            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
