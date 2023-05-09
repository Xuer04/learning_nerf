import torch
from kornia.losses import ssim as dssim
import numpy as np
from lib.config import cfg
import warnings
import json
import os
import cv2
warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(self,):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.img = []
        self.coef = 1

    def mse_metric(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        value = (image_pred-image_gt)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return np.mean(value)
        return value

    def psnr_metric(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        return -10*np.log10(self.mse_metric(image_pred, image_gt, valid_mask, reduction))

    def ssim_metric(self, image_pred, image_gt, reduction='mean'):
        """
        image_pred and image_gt: (1, 3, H, W)
        """
        dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
        return 1-2*dssim_ # in [-1, 1]

    # def evaluate(self, results, batch):
    #     ret = {}
    #     rgbs = batch['rgbs']
    #     rgbs = rgbs.squeeze() # (H*W, 3)

    #     ret['c_l'] = 0.5 * ((results['rgb_coarse']-rgbs)**2).mean()
    #     if 'rgb_fine' in results:
    #         if 'beta' not in results: # no transient head, normal MSE loss
    #             ret['f_l'] = 0.5 * ((results['rgb_fine']-rgbs)**2).mean()
    #         else:
    #             ret['f_l'] = \
    #                 ((results['rgb_fine']-rgbs)**2/(2*results['beta'].unsqueeze(1)**2)).mean()

    #     # sum the loss
    #     for k, v in ret.items():
    #         ret[k] = self.coef * v

    #     mse = sum(l for l in ret.values())
    #     self.mses.append(mse)

    #     typ = 'fine' if 'rgb_fine' in results else 'coarse'
    #     psnr = self.psnr_metric(results[f'rgb_{typ}'], rgbs)
    #     self.psnrs.append(psnr)

    # def summarize(self):
    #     ret = {}
    #     ret.update({'mse': np.mean(self.mses)})
    #     ret.update({'psnr': np.mean(self.psnrs)})
    #     ret = {item: float(ret[item]) for item in ret}
    #     print(ret)
    #     self.mses = []
    #     self.psnrs = []
    #     # self.ssim = []
    #     print('Save visualization results to {}'.format(cfg.result_dir))
    #     json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
    #     return ret

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgbs'][0].detach().cpu().numpy()
        H, W, id = batch['meta']['H'].item(), batch['meta']['W'].item(), batch['meta']['id'].item()

        img_pred = np.reshape(rgb_pred, (H, W, 3))
        img_gt = np.reshape(rgb_gt, (H, W, 3))

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        result_dir = os.path.join(cfg.result_dir, 'images')
        os.system('mkdir -p {}'.format(result_dir))
        cv2.imwrite(
            '{}/view{:03d}_pred.png'.format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255)
        )
        cv2.imwrite(
            '{}/view{:03d}_gt.png'.format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255)
        )

    def summarize(self):
        ret = {}
        ret.update({'mse': np.mean(self.mse)})
        ret.update({'psnr': np.mean(self.psnr)})
        # ret.update({'ssim': np.mean(self.ssim)})
        ret = {item: float(ret[item]) for item in ret}
        print(ret)
        self.mse = []
        self.psnr = []
        # self.ssim = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret
