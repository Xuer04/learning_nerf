import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.networks.encoding import get_encoder
from lib.config import cfg


class NeRF(nn.Module):
    def __init__(self,):
        super(NeRF, self).__init__()
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


    def forward(self, xyzs, view_dirs):
        """
        Input:
            @xyzs: [N_rays * N_samples, 63]
            @view_dirs: [N_rays * N_samples, 27]

        Output:
            @alpha:
            @rgb:
        """
        xyzs = torch.reshape(xyzs, [-1, xyzs.shape[-1]])
        xyzs = self.xyz_encoder(xyzs)
        view_dirs = torch.reshape(view_dirs, [-2, view_dirs.shape[-1]])
        view_dirs = self.dir_encoder(view_dirs)

        h = xyzs
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([xyzs, h], -1)

        if self.use_viewdirs:
            # alpha is related to pts
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # rgb is related to pts and views
            h = torch.cat([feature, view_dirs], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
