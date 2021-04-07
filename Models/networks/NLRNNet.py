import torch
import torch.nn as nn


class NLRNNet(nn.Module):
    def __init__(self,in_channels,out_channels,state_num=12):
        super().__init__()

        self.bn1        = nn.BatchNorm2d(1)
        self.pre_conv2  = nn.Conv2d(in_channels,128,kernel_size=3,padding=1)
        self.state_num  = state_num
        self.residual_block = ResidualBlock(128)

        self.bn2        = nn.BatchNorm2d(128)
        self.relu       = nn.ReLU()
        self.post_conv2 = nn.Conv2d(128,out_channels,kernel_size=3,padding=1)

    def forward(self,input):
        x = self.bn1(input)
        x = self.pre_conv2(x)
        y = x
        for i in range(self.state_num):
            x,corr = self.residual_block(x,y,None) if i==0 else self.residual_block(x,y,corr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.post_conv2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128,out_channels,3,padding=1)
        self.non_local_block = NonLocalBlock(64,128)

    def forward(self, input,y,corr=None):
        x = self.bn(input)
        x = self.relu(x)
        x,corr_new = self.non_local_block(x,y,corr)

        x = self.bn(x)
        x = self.conv2(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = torch.add(x,y)
        return x,corr_new


class NonLocalBlock(nn.Module):
    def __init__(self,filter_num,output_filter_num):
        super().__init__()
        self.theta_conv2 = nn.Conv2d(128,filter_num,3,padding=1)
        self.phi_conv2 = nn.Conv2d(128,filter_num,3,padding=1)
        self.g_conv2 = nn.Conv2d(128,output_filter_num,3,padding=1)

    def forward(self, x,y,corr=None):
        x_theta = self.theta_conv2(x)
        x_phi   = self.phi_conv2(x)
        x_g     = self.g_conv2(x)

        x_theta_reshaped = x_theta.reshape(x_theta.shape[0],x_theta.shape[1],-1)
        x_phi_reshaped   = x_phi.reshape(x_phi.shape[0],x_phi.shape[1],-1)
        x_theta_permuted   = x_theta_reshaped.permute([0,2,1])
        x_mul1 = x_theta_permuted@x_phi_reshaped

        if corr is not None:
            x_mul1 += corr

        x_mul_softmax  = nn.Softmax(dim=-1)(x_mul1)
        x_g_reshaped   = x_g.reshape(x_g.shape[0],x_g.shape[1],-1).permute([0,2,1])
        x_mul2         = x_mul_softmax @ x_g_reshaped
        x_mul2_reshaped = x_mul2.reshape(x_mul2.shape[0],x_phi.shape[2],x_phi.shape[3],x_g.shape[1]).permute([0,3,1,2])
        return torch.add(x,x_mul2_reshaped),x_mul1

