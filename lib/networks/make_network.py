import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path  # network_path defined? default value?
    network = imp.load_source(module, path).Network()
    return network
