from .resnet18  import Resnet18

import torch
import torch.nn as nn


def get_model(args):
    if args.model_name  == 'resnet18':
        return  Resnet18(pretrained=args.pretrain,
                         drop_out = args.dropout)
        else:
        raise ValueError(f"Cannot find the model {args.model_name}")