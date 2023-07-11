import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.networks.encoding import get_encoder
from lib.config import cfg
from collections import OrderedDict


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_a=48, skips=[4], encode_a=True, use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch_xyz = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_a = input_ch_a
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.encode_a = encode_a
        self.output_ch = 5 if self.use_viewdirs else 4

        self.pts_linears = nn.ModuleList(
        [nn.Linear(self.input_ch_xyz, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch_xyz, self.W) for i in
                                        range(self.D - 1)])

        if self.encode_a:
            apperence_encoding = OrderedDict([
                ('static_linear_0', nn.Linear(self.W + self.input_ch_views + self.input_ch_a, self.W // 2)),
                ('static_relu_0', nn.ReLU(True))
            ])
            for s_layer_i in range(1, self.D // 2):
                apperence_encoding[f'static_linear_{s_layer_i}'] = nn.Linear(self.W // 2, self.W // 2)
                apperence_encoding[f'static_relu_{s_layer_i}'] = nn.ReLU(True)
            self.apperence_encoding = nn.Sequential(apperence_encoding)

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W // 2)])

        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)
            # self.alpha_linear = nn.Sequential(nn.Linear(self.W, 1), nn.Softplus())
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            # self.rgb_linear = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)


    def forward(self, x):
        if self.encode_a:
            input_pts, input_views, embedding_a = torch.split(x, [self.input_ch_xyz, self.input_ch_views, self.input_ch_a], dim=-1)
        else:
            input_pts, input_views = torch.split(x, [self.input_ch_xyz, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            if self.encode_a:
                apperence_encoding_input = torch.cat([feature, input_views, embedding_a], -1)
                h = self.apperence_encoding(apperence_encoding_input)
            else:
                h = torch.cat([feature, input_views], -1)
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nerf_opts = cfg.network.nerf
        self.encode_a = nerf_opts.encode_a
        self.input_ch_a = nerf_opts.N_a if self.encode_a else 0

        # encoder
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        if self.encode_a:
            # embedding appearance
            self.embeda_fn = torch.nn.Embedding(nerf_opts.N_vocab, self.input_ch_a)


        # coarse model
        self.model = NeRF(D=cfg.network.nerf.D,
                          W=cfg.network.nerf.W,
                          input_ch=self.input_ch,
                          input_ch_views=self.input_ch_views,
                          input_ch_a=self.input_ch_a,
                          skips=cfg.network.nerf.skips,
                          encode_a=self.encode_a,
                          use_viewdirs=self.use_viewdirs)

        # fine model
        self.model_fine = NeRF(D=cfg.network.nerf.D,
                               W=cfg.network.nerf.W,
                               input_ch=self.input_ch,
                               input_ch_views=self.input_ch_views,
                               input_ch_a=self.input_ch_a,
                               skips=cfg.network.nerf.skips,
                               encode_a=self.encode_a,
                               use_viewdirs=self.use_viewdirs)


    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret


    def forward(self, inputs, viewdirs, ts, model=''):
        """Prepares inputs and applies network 'fn'."""
        if model == 'fine':
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        if self.encode_a:
            input_ts = ts[:, None, :].expand(inputs.shape)
            input_ts_flat = torch.reshape(input_ts, [-1, input_ts.shape[-1]])
            input_ts_flat = input_ts_flat[..., :1]
            embedded_ts = self.embeda_fn(input_ts_flat)
            embedded_ts = embedded_ts.reshape(-1, self.input_ch_a)
            embedded = torch.cat([embedded, embedded_ts], -1)

        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
