import torch.nn as nn

from utils.layerUtils import chooseActivation

class ConvBatActLayer2dLayer(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size,padding,act):
        super(ConvBatActLayer2dLayer, self).__init__()
        layer = []

        layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(chooseActivation(act))

        self.layer_ = nn.Sequential(*layer)

    def apply(self, fn):
        self.layer_.apply(fn)

    def forward(self,input):
        x = input
        for layer in self.layer_:
            x = layer(x)
        return x

