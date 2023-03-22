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
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis')


    def evaluate(self, output, batch):
        pred_rgb = output['rgb1'][0].detach().cpu().numpy()             # [H * W, 3]
        gt_rgb = batch['rgb'][0].detach().cpu().numpy()                 # [H * W, 3]

        mse_item = np.mean((pred_rgb - gt_rgb)**2)
        self.mses.append(mse_item)

        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)


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
