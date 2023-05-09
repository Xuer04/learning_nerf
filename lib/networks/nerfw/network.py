import torch
import torch.nn as nn
from lib.networks.encoding import get_encoder
from lib.config import cfg
from torch.nn import functional as F
from .rendering import *


class NeRF(nn.Module):
    def __init__(self, typ, D=8, W=256, input_ch_xyz=3, input_ch_dir=3, input_ch_a=48, input_ch_tau=16, skips=[4], use_viewdirs=False, encode_a=True, encode_t=True, beta_min=0.1):
        """
        Args:
            D (int, optional): depth of network. Defaults to 8.
            W (int, optional): width of network. Defaults to 256.
            input_ch_xyz (int, optional): input dimension of xyz. Defaults to 3.
            input_ch_dir (int, optional): input dimension of view. Defaults to 3.
            input_ch_a (int, optional): input embedding of appearance. Defaults to 48.
            input_ch_tau (int, optional): input embedding of transient. Defaults to 16.
            skips (list, optional): _description_. Defaults to [4].
            use_viewdirs (bool, optional): whether use view. Defaults to False.
            encode_t (bool, optional): whether encoding transient. Defaults to True.
            beta_min (float, optional): hyperparameter used in the transient loss calculation. Defaults to 0.1.
        """
        super(NeRF, self).__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.input_ch_xyz = input_ch_xyz
        self.input_ch_dir = input_ch_dir
        self.input_ch_a = input_ch_a
        self.input_ch_t = input_ch_tau
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4
        self.encode_appearance = encode_a
        self.encode_transient = encode_t
        self.beta_min = beta_min

        # xyz encoding layers
        self.xyz_linears = nn.ModuleList(
        [nn.Linear(self.input_ch_xyz, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch_xyz, self.W) for i in
                                        range(self.D - 1)])
        self.xyz_encoding_head = nn.Linear(W, W)

        # dir encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+input_ch_dir+self.input_ch_a, W//2), nn.ReLU(True))


        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+self.input_ch_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())


    def forward(self, x, output_sigma_only=False, output_transient=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).

        Inputs:
            `x`: the embedded vector of position (+ direction + appearance + transient)
            `sigma_only`: whether to infer sigma only.
            `output_transient`: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if output_sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.input_ch_xyz,
                                self.input_ch_dir+self.input_ch_a,
                                self.input_ch_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.input_ch_xyz,
                                self.input_ch_dir+self.input_ch_a], dim=-1)

        xyz_ = input_xyz
        for i, l in enumerate(self.xyz_linears):
            xyz_ = self.xyz_linears[i](xyz_)
            xyz_ = F.relu(xyz_)
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)

        static_sigma = self.static_sigma(xyz_)
        if output_sigma_only:
            return static_sigma # (B, 1)

        xyz_encoding = self.xyz_encoding_head(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding, input_dir_a], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], -1)

        if not output_transient:
            return static # (B, 4)

        transient_encoding_input = torch.cat([xyz_encoding, input_t], -1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], -1) # (B, 5)

        return torch.cat([static, transient], -1) # (B, 9)


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

        # parameters for nerf-w
        nerfw_opts = cfg.network.nerfw
        self.encode_appearance = nerfw_opts.encode_a
        self.input_ch_a = nerfw_opts.N_a if self.encode_appearance else 0
        self.encode_transient = nerfw_opts.encode_t
        self.input_ch_tau = nerfw_opts.N_tau if self.encode_transient else 0
        self.beta_min = nerfw_opts.beta_min

        # embedding xyz, dir
        self.embedding_xyz, self.input_ch_xyz = get_encoder(cfg.network.xyz_encoder)
        self.embedding_dir, self.input_ch_dir = get_encoder(cfg.network.dir_encoder)

        if self.encode_appearance:
            # embedding appearance
            self.embedding_a = torch.nn.Embedding(nerfw_opts.N_vocab, self.input_ch_a)
        if self.encode_transient:
            # embedding transient
            self.embedding_t = torch.nn.Embedding(nerfw_opts.N_vocab, self.input_ch_tau)

        # coarse model
        self.model = NeRF("coarse",
                          D=cfg.network.nerfw.D,
                          W=cfg.network.nerfw.W,
                          input_ch_xyz=self.input_ch_xyz,
                          input_ch_dir=self.input_ch_dir,
                          input_ch_a=self.input_ch_a,
                          input_ch_tau=self.input_ch_tau,
                          skips=cfg.network.nerfw.skips,
                          use_viewdirs=self.use_viewdirs,
                          encode_a=self.encode_appearance,
                          encode_t=self.encode_transient)

        if self.N_importance > 0:
            # fine model
            self.model_fine = NeRF("fine",
                                D=cfg.network.nerfw.D,
                                W=cfg.network.nerfw.W,
                                input_ch_xyz=self.input_ch_xyz,
                                input_ch_dir=self.input_ch_dir,
                                input_ch_a=self.input_ch_a,
                                input_ch_tau=self.input_ch_tau,
                                skips=cfg.network.nerfw.skips,
                                use_viewdirs=self.use_viewdirs,
                                encode_a=self.encode_appearance,
                                encode_t=self.encode_transient)

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret

    def forward(self, inputs, viewdirs, ts, model=''):
        """Do batched inference on rays using chunk."""
        if model == 'fine':
            fn = self.model_fine
        else:
            fn = self.model

        embedded = []
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded += [self.embedding_xyz(inputs_flat)]

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embedding_dir(input_dirs_flat)
            embedded += [embedded_dirs]
            # embedded = torch.cat([embedded, embedded_dirs], -1)

        if self.encode_appearance:
            ts = ts[:, None].expand([inputs.shape[0], inputs.shape[1],1])
            ts = torch.reshape(ts, [-1, ts.shape[-1]])
            embedded_ts = self.embedding_a(ts)
            embedded_ts = embedded_ts.reshape(-1, self.input_ch_a)
            embedded += [embedded_ts]
            # embedded = torch.cat([embedded, embedded_ts], -1)

        embedded = torch.cat(embedded, -1)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
