import torch
import torch.nn as nn
from lib.networks.nerfw.renderer import rendering


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        """
        lambda_u: in equation 13
        """
        super(NetworkWrapper, self).__init__()
        self.coef = 1
        self.lambda_u = 0.01
        self.net = net
        self.renderer = rendering.Renderer(self.net)
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        """
        Equation 13 in the NeRF-W paper.
        Name abbreviations:
            c_l: coarse color loss
            f_l: fine color loss (1st term in equation 13)
            b_l: beta loss (2nd term in equation 13)
            s_l: sigma loss (3rd term in equation 13)
        """
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        img_loss = self.img2mse(ret['rgb_map'], batch['rgbs'])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgbs'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
