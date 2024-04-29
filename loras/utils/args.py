import argparse
import yaml

from types import SimpleNamespace

def load_args_and_config(config_path='configs/loras_assembly.yaml'):

    # Load cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config.update(vars(args))
    return SimpleNamespace(**config)

