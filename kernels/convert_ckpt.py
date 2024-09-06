import torch
from collections import OrderedDict
import argparse
import os

def parse_args():

    parser = argparse.ArgumentParser(
        description="Script to convert the ckpt"
    )
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_path", type=str, default='./output')
    args = parser.parse_args()
    return args


args = parse_args()
checkpoint = torch.load(args.ckpt)

new_checkpoint = OrderedDict()

for key in checkpoint.keys():
    if not 'act_quantizer_k' in key and not 'act_quantizer_q' in key and not 'act_quantizer_v' in key:
        # new_key = key.replace('.act_quantizer', '').replace('.weight_quantizer', '')
        new_checkpoint[key] = {}
        if isinstance(checkpoint[key][0], OrderedDict):
            for sub_key in checkpoint[key][0].keys():
                if sub_key == 'delta_list' or sub_key == 'zero_point_list':
                    # 如果该项是一个Tensor，那么将其转换为torch.fp16
                    if isinstance(checkpoint[key][0][sub_key], torch.Tensor):
                            shape = checkpoint[key][0][sub_key].shape
                            print(shape)
                            if shape == torch.Size([0]):
                                print(key, sub_key)
                                print(checkpoint[key][0][sub_key])
                            new_checkpoint[key][sub_key] = checkpoint[key][0][sub_key].half()
                            if 'weight_quantizer' in key:
                                new_checkpoint[key][sub_key] = new_checkpoint[key][sub_key].reshape(shape[0], shape[1])
                            if 'act_quantizer' in key:
                                new_checkpoint[key][sub_key] = new_checkpoint[key][sub_key].reshape(shape[0])

print(len(new_checkpoint))

os.makedirs(args.save_path, exist_ok=True)
path_newckpt = os.path.join(args.save_path, 'new_ckpt.pth')
torch.save(new_checkpoint, path_newckpt)
