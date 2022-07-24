import datetime
import logging
import os
import sys
import torch.nn as nn
from torch.nn import init
import yaml


def date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now()).replace('-', '') \
        .replace(' ', '').replace(':', '').replace('.', '')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    init.normal_(m.bias, 0.0001)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    init.normal_(m.bias, 0.0001)
            elif isinstance(m, nn.BatchNorm2d):
                try:
                    init.constant_(m.weight, 1)
                    init.normal_(m.bias, 0.0001)
                except:
                    print('This layer has no BN parameters:', m)