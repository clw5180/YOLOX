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

    for key, val in in_state_dict['model'].items():
        if not 'backbone.backbone' in key:
            break
        out_state_dict['model'][key] = val

    torch.save(out_state_dict, out_file)


    # sha = subprocess.check_output(['sha256sum', out_file]).decode()
    # if out_file.endswith('.pth'):
    #     out_file_name = out_file[:-4]
    # else:
    #     out_file_name = out_file
    # final_file = out_file_name + f'-{sha[:8]}.pth'
    # subprocess.Popen(['mv', out_file, final_file])
    # print('final_file:', final_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default='/home/caoliwei/base_utils/yolox_s.pth', help='input checkpoint file')
    parser.add_argument('--out_file', default='/home/caoliwei/base_utils/yolox_s_backbone.pth', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()

