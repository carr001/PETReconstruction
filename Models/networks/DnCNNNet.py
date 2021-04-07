import torch
import torch.nn as nn

from layers.ConvBatActLayer2dLayer import ConvBatActLayer2dLayer

class DnCNNNet(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, padding, act, num_of_layers=17):
        super(DnCNNNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(ConvBatActLayer2dLayer(in_channels=64,out_channels=64, kernel_size=kernel_size,padding=padding,act=act))
        layers.append(nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn_ = nn.Sequential(*layers)

    def forward(self, input):
        # return self.dncnn_(input)

        x = input
        for layer in self.dncnn_:
            x = layer(x)
        return x+input

    def apply(self, fn):
        for module in list(self.children())[0]:
            module.apply(fn)
        fn(self)
        return self

if __name__ == "__main__":
    a = DnCNNNet(3,3,3,0,"Relu")
    print(a.device)