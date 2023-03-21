import imp
import os

def _visualizer_factory(cfg):
    path = cfg.visualizer_path
    module = path[:-3].replace('/', '.')
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer


def make_visualizer(cfg):
    return _visualizer_factory(cfg)
