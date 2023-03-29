import matplotlib.pyplot as plt
import numpy as np
import os
from termcolor import colored
from lib.config import cfg
from lib.utils.vis_utils import to8b

class Visualizer:
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.result_dir = cfg.result_dir


    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        H, W, ratio = batch['meta']['H'].item(), batch['meta']['W'].item(), batch['meta']['ratio'].item()
        H, W = int(H * ratio), int(W * ratio)

        img_pred = to8b(np.cat(rgb_pred, dim=0).reshape(H, W, 3))
        img_gt = to8b(np.cat(rgb_gt, dim=0).reshape(H, W, 3))

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()
