from lib.config import cfg
import numpy as np
import torch.nn.functional as F
import torch

def get_rays():
    pass


def sample_pdf():
    pass


def ndc_rays():
    pass


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
