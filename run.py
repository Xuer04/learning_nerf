from lib.config import cfg, args
import numpy as np
import os
### SCRIPTS BEGINING ###
def run_dataset():
    import tqdm
    from lib.datasets import make_data_loader

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    import tqdm
    import torch
    import time
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    import time
    import tqdm
    import torch
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.networks.nerf.renderer import make_renderer
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    renderer = make_renderer(cfg, network)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = renderer.render(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        # print(f"iter test: {batch['meta'['iter_test']]}")
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))


def run_visualize():
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.visualizers import make_visualizer
    from lib.networks.nerf.renderer import make_renderer
    from lib.utils.data_utils import to_cuda


    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
        visualizer.visualize(output, batch)

    if visualizer.write_video:
        visualizer.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()
