import matplotlib.pyplot as plt
import numpy as np
import os
from termcolor import colored
from lib.config import cfg


class Visualizer:
    def __init__(self):
        self.write_video = None
        result_dir = cfg.result_dir
        self.result_dir = os.path.join(result_dir, 'vis', cfg.scene)
        print(colored('the results are saved at {}'.format(self.result_dir), 'yellow'))
        os.system('mkdir -p {}'.format(self.result_dir))

    def visualize(self, output, batch):
        white_bkgd = cfg.task_arg.white_bkgd
        rgb_pred = output['rgb1'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        H = int(cfg.train_dataset.H * cfg.train_dataset.input_ratio)
        W = int(cfg.train_dataset.W * cfg.train_dataset.input_ratio)

        pred_img = np.zeros((H, W, 3))
        if white_bkgd:
            pred_img = pred_img + 1
        pred_img = rgb_pred

        gt_img = np.zeros((H, W, 3))
        if white_bkgd:
            gt_img = gt_img + 1
        gt_img = rgb_gt

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(pred_img)
        # ax2.imshow(gt_img)
        # plt.show()

    def summarize(self):
        # TODO
        pass
