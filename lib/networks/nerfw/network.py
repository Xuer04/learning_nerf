import torch
import torch.nn as nn
from lib.networks.encoding import get_encoder
from collections import OrderedDict
from lib.config import cfg
from torch.nn import functional as F


class NeRF(nn.Module):
    def __init__(self,
                 typ,
                 D=8,
                 W=256,
                 input_ch_xyz=3,
                 input_ch_dir=3,
                 input_ch_a=48,
                 input_ch_tau=16,
                 skips=[4],
                 use_viewdirs=False,
                 encode_a=True,
                 encode_t=True,
                 beta_min=0.1):
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
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_xyz, self.W)] +
            [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch_xyz, self.W) for i in range(D - 1)]
        )

        if self.encode_appearance:
            apperence_encoding = OrderedDict([
                ('static_linear_0', nn.Linear(self.W + self.input_ch_dir + self.input_ch_a, self.W // 2)),
                ('static_relu_0', nn.ReLU(True))
            ])
            for s_layer_i in range(1, self.D // 2):
                apperence_encoding[f'static_linear_{s_layer_i}'] = nn.Linear(self.W // 2, self.W // 2)
                apperence_encoding[f'static_relu_{s_layer_i}'] = nn.ReLU(True)
            self.apperence_encoding = nn.Sequential(apperence_encoding)

        # view encoding layers
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_dir+ self.W, self.W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(self.W, self.W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(self.W, self.output_ch)


    def forward(self, x):
        input_pts, input_views, embedding_a = torch.split(x, [self.input_ch_xyz, self.input_ch_dir, self.input_ch_a], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            if self.encode_appearance:
                apperence_encoding_input = torch.cat([feature, input_views, embedding_a], -1)
                h = self.apperence_encoding(apperence_encoding_input)
            else:
                h = torch.cat([feature, input_views], -1)
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
            return outputs
        else:
            assert False


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

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embedding_xyz(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embedding_dir(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        if self.encode_appearance:
            input_ts = ts[:, None, :].expand(inputs.shape)
            input_ts_flat = torch.reshape(input_ts, [-1, input_ts.shape[-1]])
            input_ts_flat = input_ts_flat[..., :1]
            embedded_ts = self.embedding_a(input_ts_flat)
            embedded_ts = embedded_ts.reshape(-1, self.input_ch_a)
            embedded = torch.cat([embedded, embedded_ts], -1)

        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
