import matplotlib.pyplot as plt
import numpy as np
import os
from termcolor import colored
from lib.config import cfg


class Visualizer:
    def __init__(self, is_train):
        self.write_video = None
        result_dir = cfg.result_dir
        self.result_dir = os.path.join(result_dir, 'vis', cfg.scene)
        self.input_ratio = cfg.train_dataset.input_ratio if is_train else cfg.test_dataset.input_ratio
        self.H = cfg.train_dataset.H if is_train else cfg.test_dataset.H
        self.W = cfg.train_dataset.W if is_train else cfg.test_dataset.W
        print(colored('the results are saved at {}'.format(self.result_dir), 'yellow'))
        os.system('mkdir -p {}'.format(self.result_dir))


    def visualize(self, output, batch):
        white_bkgd = cfg.task_arg.white_bkgd
        rgb_pred = output['rgb1'][0].detach().cpu().numpy()     # [H * W, 3]
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()         # [H * W, 3]

        # print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        H = int(self.H * self.input_ratio)
        W = int(self.W * self.input_ratio)
        rgb_pred = rgb_pred.reshape(H, W, 3)                    # [H, W, 3]
        rgb_gt = rgb_gt.reshape(H, W, 3)                        # [H, W, 3]

        pred_img = np.zeros((H, W, 3))
        if white_bkgd:
            pred_img = pred_img + 1
        pred_img += rgb_pred

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
