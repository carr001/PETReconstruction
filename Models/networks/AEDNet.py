import torch
import torch.nn as nn

class Unet_down(nn.Module):
    def __init__(self, inchan,outchan,kernel_size):
        super(Unet_down, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchan, outchan,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(outchan),
                                    torch.nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(outchan, outchan,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(outchan),
                                    torch.nn.ReLU())
        self.downsample = nn.Sequential(nn.Conv2d(outchan, outchan,kernel_size=kernel_size,padding=1,stride=2),
                                    nn.BatchNorm2d(outchan),
                                    torch.nn.ReLU())# kernel size is supporse to be odd
    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.downsample(x)
        return x

class Unet_up(nn.Module):
    def __init__(self, inchan,outchan,kernel_size):
        super(Unet_up,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchan, inchan//2,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(inchan//2),
                                    torch.nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(inchan//2, inchan//2,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(inchan//2),
                                    torch.nn.ReLU())
        # 对于upsample 需要修改kernel_size,得到2二倍输出
        self.upsample = nn.Sequential(nn.ConvTranspose2d(inchan//2, outchan,kernel_size=kernel_size,output_padding=1,padding=kernel_size//2,stride=2),
                                    nn.BatchNorm2d(outchan),
                                    torch.nn.ReLU())
        self.feature_map = []

    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        if not self.training:
            self.feature_map.append(x.detach().to('cpu').numpy())# its supposed to be from son class
        x = self.upsample(x)
        return x
class Unet_bottle(nn.Module):
    def __init__(self, inchan,kernel_size):
        super(Unet_bottle,self).__init__()
        self.conv1 =nn.Sequential(nn.Conv2d(inchan, inchan*2,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(inchan*2),
                                    torch.nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(inchan*2, inchan,kernel_size=kernel_size,padding=kernel_size//2),
                                    nn.BatchNorm2d(inchan),
                                    torch.nn.ReLU())
        # self.upsample = nn.Sequential(nn.ConvTranspose2d(2*inchan, inchan,kernel_size=kernel_size+1,padding=kernel_size//2,stride=2),
        #                             nn.BatchNorm2d(inchan),
        #                             torch.nn.ReLU())
    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        # x = self.upsample(x)
        return x

class AED(nn.Module):
    def __init__(self,kernel_size = 3):
        super(AED,self).__init__() # 初始化基类
        self.down1 = Unet_down(1,16,kernel_size)
        self.down2 = Unet_down(16, 32, kernel_size)
        self.down3 = Unet_down(32, 64, kernel_size)
        self.down4 = Unet_down(64, 128, kernel_size)

        self.bottle = Unet_bottle(128,kernel_size)
        # self.bottle = Unet_up(128,64,kernel_size)
        self.up4 = Unet_up(128, 64,kernel_size)
        self.up3 = Unet_up(64, 32, kernel_size)
        self.up2 = Unet_up(32, 16, kernel_size)
        self.up1 = Unet_up(16, 1, kernel_size)

    def forward(self,input):
        x = self.down1(input)# inchan 1,(160, 192)    , outchan 16,(80, 96)
        x = self.down2(x)# inchan 16,(80, 96)    , outchan 32,(40, 48)
        x = self.down3(x)# inchan 32,(40, 48)    , outchan 64,(20, 24)
        x = self.down4(x)# inchan 64,(20, 24)    , outchan 128,(10, 12)

        x = self.bottle(x)# inchan 128,(10, 12)    , outchan 128,(10, 12)

        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return x

if __name__ =='__main___':

    pass

























