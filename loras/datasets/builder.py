from utils import Registry
import torch

DATASETS = Registry()

def build_data_module(identifier, **kwargs):
    dataset = DATASETS[identifier]
    return dataset(kwargs)