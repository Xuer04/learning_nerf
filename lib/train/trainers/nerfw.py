import torch
import torch.nn as nn
from lib.networks.nerf.renderer import volume_renderer

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, results, targets):
        loss = self.loss(results['rgb_coarse'], targets)
        if 'rgb_fine' in results:
            loss += self.loss(results['rgb_fine'], targets)

        return self.coef * loss


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = 1
        self.lambda_u = 0.01
        self.net = net

    def forward(self, batch):
        """
        Equation 13 in the NeRF-W paper.
        Name abbreviations:
            c_l: coarse color loss
            f_l: fine color loss (1st term in equation 13)
            b_l: beta loss (2nd term in equation 13)
            s_l: sigma loss (3rd term in equation 13)
        """
        scalar_stats = {}
        loss = 0

        targets = batch['rgbs']
        results = self.net(batch)
        ret = {}
        ret['c_l'] = 0.5 * ((results['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in results:
            if 'beta' not in results: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((results['rgb_fine']-targets)**2).mean()
            else:
                ret['f_l'] = \
                    ((results['rgb_fine']-targets)**2/(2*results['beta'].unsqueeze(1)**2)).mean()
                ret['b_l'] = 3 + torch.log(results['beta']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * results['transient_sigmas'].mean()

        # sum the loss
        for k, v in ret.items():
            ret[k] = self.coef * v

        scalar_stats.update({'img_loss0': ret['c_l']})
        scalar_stats.update({'img_loss': ret['f_l']})
        scalar_stats.update({'beta_loss': ret['b_l']})
        scalar_stats.update({'sigma_loss': ret['s_l']})

        loss = sum(l for l in ret.values())
        scalar_stats.update({'all_loss': loss})

        image_stats = {}

        return results, loss, scalar_stats, image_stats
