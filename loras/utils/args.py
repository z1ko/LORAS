import argparse
import yaml

from types import SimpleNamespace

def load_args_and_config():

    # Load cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/loras_assembly.yaml')
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config.update(vars(args))
    return SimpleNamespace(**config)

