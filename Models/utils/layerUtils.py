import torch.nn as nn

def chooseActivation(act='Relu'):
    if act == 'Tanh':
        return nn.Tanh()
    elif act == 'Sigmoid':
        return nn.Sigmoid()
    elif act == 'Softmax':
        return nn.Softmax()
    elif act == 'Relu':
        return nn.ReLU()
    elif act == 'LeakyReLU':
        return nn.LeakyReLU()