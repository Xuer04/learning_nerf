import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(self,):
        self.mses = []
        self.psnrs = []


    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        H, W, ratio = batch['meta']['H'].item(), batch['meta']['W'].item(), batch['meta']['ratio'].item()
        H, W = int(H * ratio), int(W * ratio)

        white_bkgd = int(cfg.task_arg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt = rgb_gt

        if cfg.eval.whole_img:
            rgb_pred = img_pred
            rgb_gt = img_gt

        mses = np.mean((rgb_pred - rgb_gt)**2)
        self.mses.append(mses)

        psnrs = psnr(rgb_gt, rgb_pred, data_range=1.)
        self.psnrs.append(psnrs)


    def summarize(self):
        ret = {}
        ret.update({'mse': np.mean(self.mses)})
        ret.update({'psnr': np.mean(self.psnrs)})
        ret = {item: float(ret[item]) for item in ret}
        print(ret)
        self.mses = []
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret
