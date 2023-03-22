import imp
import os

def _visualizer_factory(cfg, is_train):
    path = cfg.visualizer_path
    module = path[:-3].replace('/', '.')
    visualizer = imp.load_source(module, path).Visualizer(is_train)
    return visualizer


def make_visualizer(cfg, is_train):
    return _visualizer_factory(cfg, is_train)
