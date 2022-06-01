import argparse
import subprocess
from collections import OrderedDict
from copy import deepcopy
import torch
import re


def convert(in_file, out_file):
    """Convert keys in checkpoints."""
    in_state_dict = torch.load(in_file, map_location='cpu')
    out_state_dict = deepcopy(in_state_dict)
    out_state_dict['model'] = dict()

    for key, val in in_state_dict['state_dict'].items():
        if key =='module.conv2.weight':
            break

        if 'module.' in key:
            key = key.replace('module.', '')

        if 'bneck.0' in key:
            new_key = key.replace('bneck.0', 'bneck1.0')
        elif 'bneck.1.' in key:
            new_key = key.replace('bneck.1.', 'bneck1.1.')
        elif 'bneck.2' in key:
            new_key = key.replace('bneck.2', 'bneck1.2')

        elif 'bneck.3' in key:
            new_key = key.replace('bneck.3', 'bneck2.0')
        elif 'bneck.4' in key:
            new_key = key.replace('bneck.4', 'bneck2.1')
        elif 'bneck.5' in key:
            new_key = key.replace('bneck.5', 'bneck2.2')
        elif 'bneck.6' in key:
            new_key = key.replace('bneck.6', 'bneck2.3')
        elif 'bneck.7' in key:
            new_key = key.replace('bneck.7', 'bneck2.4')

        elif 'bneck.8' in key:
            new_key = key.replace('bneck.8', 'bneck3.0')
        elif 'bneck.9' in key:
            new_key = key.replace('bneck.9', 'bneck3.1')
        elif 'bneck.10' in key:
            new_key = key.replace('bneck.10', 'bneck3.2')
        else:
            new_key = key
        new_key = 'backbone.backbone.' + new_key

        out_state_dict['model'][new_key] = val

    del out_state_dict['state_dict']
    torch.save(out_state_dict, out_file)
    print('end !!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default='/home/caoliwei/base_utils/mbv3_small.pth', help='input checkpoint file')
    parser.add_argument('--out_file', default='/home/caoliwei/base_utils/mbv3_small_for_detection.pth', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()

