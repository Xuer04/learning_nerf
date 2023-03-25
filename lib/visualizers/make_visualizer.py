import imp
import os

def _visualizer_factory(cfg, is_train):
    module = cfg.visualizer_module
    path = cfg.visualizer_path
    visualizer = imp.load_source(module, path).Visualizer(is_train)
    return visualizer


def make_visualizer(cfg, is_train):
    return _visualizer_factory(cfg, is_train)
