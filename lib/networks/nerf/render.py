import torch


def sample_viewdirs(ray_d):
    viewdirs = ray_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    return viewdirs


def sample_rays(ray_o, ray_d, device, N_samples):
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    near, far = 2., 6.
    z_vals = near * (1. - t_vals) + far * (t_vals)
    num_rays = len(ray_o)
    z_vals = z_vals.expand([num_rays, N_samples])
    pts = ray_o[:, None, :] + ray_d[:, None, :] * z_vals[..., None]
    return pts, z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    # Get PDF
    device = weights.device
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # [batch, len(bins)]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples


def raw2rgb(sigma, rgb, z_vals, ray_d, white_bkgd=True):
    device = sigma.device
    delta_prefix = z_vals[..., 1:] - z_vals[..., :-1]
    delta_addition = torch.full((z_vals.size(0), 1), 1e10, device=device)
    delta = torch.cat([delta_prefix, delta_addition], dim=-1)
    delta = delta * torch.norm(ray_d[..., None, :], dim=-1)

    alpha = 1. - torch.exp(-sigma * delta)
    exp_term = 1. - alpha
    epsilon = 1e-10
    exp_addition = torch.ones(exp_term.size(0), 1, device=device)
    exp_term = torch.cat([exp_addition, exp_term + epsilon], dim=-1)
    transmittance = torch.cumprod(exp_term, axis=-1)[..., :-1]

    weights = alpha * transmittance
    rgb = torch.sigmoid(rgb[..., :3])
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb = rgb + (1. - acc_map[..., None])
    return rgb_map, depth_map, acc_map, weights


def render_rays(model, model_fine, ray_o, ray_d, device, N_samples, N_importance=0, white_bkgd=True):
    pts, z_vals = sample_rays(ray_o, ray_d, device, N_samples)
    view_dirs = sample_viewdirs(ray_d)

    sigma, rgb = model(pts, view_dirs)
    sigma = sigma.squeeze(dim=-1)
    rgb_map_0, depth_map_0, acc_map_0, weights_0 = raw2rgb(sigma, rgb, z_vals, ray_d, white_bkgd)

    ret = {'rgb0': rgb_map_0, 'depth0': depth_map_0, 'acc0': acc_map_0}

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples  = sample_pdf(z_vals_mid, weights_0[...,1:-1], N_importance, det=True)
    z_samples  = z_samples.detach()

    z_vals, _  = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples + N_importance, 3]
    sigma, rgb = model_fine(pts, view_dirs)
    sigma = sigma.squeeze(dim=-1)
    rgb_map_1, depth_map_1, acc_map_1, _ = raw2rgb(sigma, rgb, z_vals, ray_d, white_bkgd)

    ret.update({'rgb1': rgb_map_1, 'depth1': depth_map_1, 'acc1': acc_map_1, 'raw': rgb})

    for k in ret:
        if (torch.isnan(ret[k])).any() or torch.isinf(ret[k]).any():
            print(f"Numerical Error {k} contains nan or inf.")

    return ret
