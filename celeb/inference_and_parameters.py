import torch
import numpy as np
import argparse
from Model.Networks.Generator import G_Net
from utils.Logging import Config
import os

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def print_inference_time(model, config):
    model_name = os.path.join(config.checkpoint)
    pretained_model = torch.load(model_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretained_model, strict=False)
    device = torch.device("cuda")
    model.to(device)
    dummy_l_input = torch.randn(1, 1, 256, 256, dtype=torch.float).to(device)
    dummy_ab_input = torch.randn(1, 2, 256, 256, dtype=torch.float).to(device)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float).to(device)
    dummy_mask = torch.randn(1, 1, 256, 256, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_l_input, dummy_ab_input,dummy_input, dummy_mask)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_l_input, dummy_ab_input,dummy_input, dummy_mask)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAB')
    parser.add_argument('--config', type=str,
                        default='/home/la_belva/PycharmProjects/Comparison_models/LAB_Inpainting/celeb/Configs/testing_config.yaml',
                        help='path to config file')
    config = parser.parse_args()
    config = Config(config.config)
    model = G_Net(config)
    print_inference_time(model, config)