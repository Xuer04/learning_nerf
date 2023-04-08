import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from termcolor import colored
from lib.config import cfg
from lib.utils.vis_utils import to8b

class Visualizer:
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.result_dir = cfg.result_dir
        self.write_video = True


    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()

        img_pred = to8b(np.reshape(np.concatenate(rgb_pred, axis=0), (H, W, 3)))
        img_gt = to8b(np.reshape(np.concatenate(rgb_gt, axis=0), (H, W, 3)))

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()


    def summarize(self, output, batch):
        pass
