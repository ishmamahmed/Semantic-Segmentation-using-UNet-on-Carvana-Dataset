'''
This is my UNet model architecture.
'''
import torch
import torch.nn as nn
from functools import reduce
from operator import __add__
from torchinfo import summary

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
    


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        # Encoder Layers
        # self.layer1 = self.doubleConv(1,64)
        self.layer1 = self.doubleConv(3,64)         # changed the input channels
        self.l2 = nn.MaxPool2d(2,2,0)
        self.l3 = self.doubleConv(64,128)
        self.l4 = nn.MaxPool2d(2,2,0)
        self.l5 = self.doubleConv(128,256)
        self.l6 = nn.MaxPool2d(2,2,0)
        self.l7 = self.doubleConv(256,512)
        self.l8 = nn.MaxPool2d(2,2,0)
        self.l9 = self.doubleConv(512,1024)
        # Decoder Layers
        self.l10 = nn.ConvTranspose2d(1024,512,2,2,0)
        self.l11 = self.doubleConv(1024,512)
        self.l12 = nn.ConvTranspose2d(512,256,2,2,0)
        self.l13 = self.doubleConv(512,256)
        self.l14 = nn.ConvTranspose2d(256,128,2,2,0)
        self.l15 = self.doubleConv(256,128)
        self.l16 = nn.ConvTranspose2d(128,64,2,2,0)
        self.l17 = self.doubleConv(128,64)
        # self.l18 = nn.ConvTranspose2d(64,2,1,1,0)
        self.l18 = nn.ConvTranspose2d(64,1,1,1,0)       # Changed the output channels
        

    def doubleConv(self, in_channel, out_channel):
        return nn.Sequential(
            # nn.Conv2d(in_channel,out_channel,3,1,0),        # VALID CONVOLUTION
            # nn.Conv2d(out_channel,out_channel,3,1,0),       # VALID CONVOLUTION
            Conv2dSamePadding(in_channel,out_channel,3,1,0),      # SAME CONVOLUTION
            Conv2dSamePadding(out_channel,out_channel,3,1,0),     # SAME CONVOLUTION
            nn.ReLU()
            )

    def concatenate(self, tensor, target_tensor):
        delta = int((tensor.shape[2] - target_tensor.shape[2])/2)
        tensor = tensor[:,:, delta:tensor.shape[2]-delta, delta:tensor.shape[2]-delta]
        return torch.cat((tensor,target_tensor),1)

    def forward(self, input):
        '''Need to add batch normalization'''
        # Encoder
        x1 = self.layer1(input)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)
        x6 = self.l6(x5)
        x7 = self.l7(x6)
        x8 = self.l8(x7)
        x9 = self.l9(x8)
        
        # Decoder
        x10 = self.l10(x9)
        x11 = self.l11(self.concatenate(x7,x10))
        x12 = self.l12(x11)
        x13 = self.l13(self.concatenate(x5,x12))
        x14 = self.l14(x13)
        x15 = self.l15(self.concatenate(x3,x14))
        x16 = self.l16(x15)
        x17 = self.l17(self.concatenate(x1,x16))
        x18 = self.l18(x17)

        return x18

def test():    
    model = UNet()
    x = torch.rand(1,3,512,512)
    print("shape of x: ", x.shape)
    y = model(x)
    print("shape of y: ", y.shape)


# test()

