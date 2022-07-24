import torch
import torch.nn as nn
from torch.autograd import Variable


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


# class RelativisticLeastSquareGANLoss(nn.Module):
#     def __init__(self, mode='gen', target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(RelativisticLeastSquareGANLoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         self.mode = mode
#
#     def get_target_tensor(self, input, target_is_real):
#         target_tensor = None
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor
#
#     def __call__(self, y_pred_fake, y_pred, target_is_real):
#         target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
#
#         ### TODO ###
#         '''
#         Automatic device allocation
#         '''
#
#
#         target_tensor = target_tensor.cuda()
#         y_pred = y_pred.cuda()
#         y_pred_fake = y_pred_fake.cuda()
#
#         if self.mode == 'gen':
#             return (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) + torch.mean(
#                 (y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
#         elif self.mode == 'dis':
#             return (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) + torch.mean(
#                 (y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2

class RelativisticLeastSquareGANLoss(nn.Module):
    def __init__(self, mode='gen'):
        super(RelativisticLeastSquareGANLoss, self).__init__()
        self.real_label_var = None
        self.fake_label_var = None
        self.mode = mode

    def __call__(self, y_pred_fake, y_pred):
        ### TODO ###
        '''
        Automatic device allocation
        '''
        y_pred = y_pred.cuda()
        y_pred_fake = y_pred_fake.cuda()

        if self.mode == 'gen':
            return (torch.mean((y_pred - torch.mean(y_pred_fake) + 1.) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) - 1.) ** 2)) / 2
        elif self.mode == 'dis':
            return (torch.mean((y_pred - torch.mean(y_pred_fake) - 1.) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) + 1.) ** 2)) / 2
