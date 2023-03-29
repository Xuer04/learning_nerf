import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import imageio
from lib.utils import img_utils
import cv2
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(self,):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.iter = 0


    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr


    def ssim_metric(self, img_pred, img_gt, batch):
        self.iter += 1
        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        cv2.imwrite(
            '{}/iterations{:04d}_view.png'.format(result_dir, self.iter),
            (img_pred[..., [2, 1, 0]] * 255)
        )
        cv2.imwrite(
            '{}/iterations{:04d}_view_gt.png'.format(result_dir, self.iter),
            (img_gt[..., [2, 1, 0]] * 255)
        )

        # compute the ssim
        # ssim = compare_ssim(img_pred, img_gt, win_size=101, full=True)
        ssim = 0
        return ssim


    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()

        # white_bkgd = int(cfg.task_arg.white_bkgd)
        # img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred = np.reshape(rgb_pred, (H, W, 3))
        # img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt = np.reshape(rgb_gt, (H, W, 3))

        # if cfg.eval.whole_img:
        #     rgb_pred = img_pred
        #     rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(img_pred, img_gt, batch)
        # self.ssim.append(ssim)


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
