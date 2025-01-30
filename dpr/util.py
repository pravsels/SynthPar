from easydict import EasyDict as edict
import yaml
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Experiments for NetGAN"
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="config/0.yaml",
        required=True,
        help="Path of config file")
  
    args = parser.parse_args()

    return args


def get_config(config_file):
  config = edict(yaml.load(open(config_file, 'r'), 
                                Loader=yaml.FullLoader))

  return config

