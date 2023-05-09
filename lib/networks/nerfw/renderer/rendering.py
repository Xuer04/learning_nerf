import torch
from lib.config import cfg
import numpy as np
from .nerf_net_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net


    def render_rays(self, ray_batch, net_c=None, pytest=False):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., [6, 8]], [-1, 1, 2])
        # Fix some bug here, near and far with the shape of (N_rays, 2)
        # bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=cfg.task_arg.N_samples).to(near)
        if not cfg.task_arg.lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, cfg.task_arg.N_samples])

        if cfg.task_arg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]

        if net_c is None:
            raw = self.net(pts, viewdirs)
        else:
            raw = self.net(pts, viewdirs, net_c)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, cfg.task_arg.raw_noise_std, cfg.task_arg.white_bkgd)

        if cfg.task_arg.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid,
                                   weights[..., 1:-1],
                                   cfg.task_arg.N_importance,
                                   det=(cfg.task_arg.perturb == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
                ..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # raw = run_network(pts, fn=run_fn)
            if net_c is None:
                raw = self.net(pts, viewdirs, model='fine')
            else:
                raw = self.net(pts, viewdirs, net_c, model='fine')

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, cfg.task_arg.raw_noise_std, cfg.task_arg.white_bkgd)

        ret = {
            'rgb_map': rgb_map,
            'disp_map': disp_map,
            'acc_map': acc_map,
            'depth_map': depth_map
        }
        ret['raw'] = raw
        if cfg.task_arg.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            DEBUG = False
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret


    def batchify_rays(self, rays_flat, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def render(self, batch):
        """Do batched inference on rays using chunk."""
        rays, ts = batch['rays'], batch['ts']
        rays = rays.reshape(-1, 8)
        ts = ts.reshape(-1)
        sh = rays.shape[0]
        chunk_size = cfg.task_arg.chunk_size
        # results = defaultdict(list)
        # for i in range(0, sh, chunk_size):
        #     rendered_ray_chunks = \
        #         render_rays(self.models,
        #                     self.embeddings,
        #                     rays[i:i+chunk_size],
        #                     ts[i:i+chunk_size],
        #                     cfg.task_arg.N_samples,
        #                     cfg.task_arg.use_disp,
        #                     cfg.task_arg.perturb,
        #                     cfg.task_arg.raw_noise_std,
        #                     cfg.task_arg.N_importance,
        #                     chunk_size, # chunk size is effective in val mode
        #                     cfg.task_arg.white_bkgd)

        #     for k, v in rendered_ray_chunks.items():
        #         results[k] += [v]

        # for k, v in results.items():
        #     results[k] = torch.cat(v, 0)
        # return results
