import torch
import socket
import os


def checkpoint(config, model, epoch, step):
    hostname = str(socket.gethostname())
    model_out_path = config.checkpoint_dir + '/' + hostname + '_' + \
                     config.model_type + "_" + config.expname + "_" + "bs_%d_epoch_%03d_step_%06d.pth" % (
                         config.batch_size, epoch, step)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def resume(config, model):
    start_epoch = 0
    start_step = 0
    model_name = os.path.join(config.checkpoint)
    print('pretrained model: %s' % model_name)
    curr_steps = model_name[-10:-4]
    curr_epoch = model_name[-19:-16]
    if os.path.exists(model_name):
        pretained_model = torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretained_model)
        print('Pre-trained model is found!! Resuming training with the following hyper-parameters.......\n')
        print(' Current: G learning rate:', model.g_lr, ' | L1 loss weight:', model.l1_weight,
              ' | GAN loss weight:', model.gan_weight)
        start_epoch = start_epoch + int(curr_epoch)
        start_step = start_step + int(curr_steps)
        print("starting epoch from ", start_epoch)
        print("starting step from ", start_step)
        print("Successfully resumed!")
        return model, start_step, start_epoch
    else:
        print("No such pretrained model found!!! Starting new training!!")
        return model, start_step, start_epoch

