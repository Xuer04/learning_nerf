import matplotlib.pyplot as plt
import numpy as np
import os
from termcolor import colored
from lib.config import cfg

class Visualizer:
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.result_dir = cfg.result_dir


    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        H, W, ratio = batch['meta']['H'].item(), batch['meta']['W'].item(), batch['meta']['ratio'].item()
        H, W = int(H * ratio), int(W * ratio)

        img_pred = np.zeros((H, W, 3))
        if cfg.task_arg.white_bkgd:
            img_pred = img_pred + 1
        img_pred = rgb_pred

        img_gt = np.zeros((H, W, 3))
        if cfg.task_arg.white_bkgd:
            img_gt = img_gt + 1
        img_gt = rgb_gt

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()
