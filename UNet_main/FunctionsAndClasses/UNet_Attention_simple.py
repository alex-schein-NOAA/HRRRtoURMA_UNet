# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# # from torchvision import datasets
# # from torchvision import transforms
# # from torchvision.transforms import ToTensor
# # from torchinfo import summary

# import torch.nn.functional as F

# import os
# from netCDF4 import Dataset as nc_Dataset
# import pandas as pd
# import numpy as np
# import xarray as xr

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
import torch.nn.functional as F

#######################################################################################################################

class UNet_Attention_simple(nn.Module): 
    def __init__(self, 
                 n_channels_in=1, 
                 n_channels_firstlayer=64, 
                 n_channels_out=1
                ): 
        super(UNet_Attention_simple, self).__init__() 

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        self.first_conv = FirstConv(n_channels_in, n_channels_firstlayer)
        self.second_conv = ConvBlock(n_channels_firstlayer, n_channels_firstlayer)

        self.enc_1 = Encoder(n_channels_firstlayer, 2*n_channels_firstlayer)
        self.enc_2 = Encoder(2*n_channels_firstlayer, 4*n_channels_firstlayer)
        self.enc_3 = Encoder(4*n_channels_firstlayer, 8*n_channels_firstlayer)
        self.enc_4 = Encoder(8*n_channels_firstlayer, 16*n_channels_firstlayer)

        self.dec_1 = Decoder(16*n_channels_firstlayer, 8*n_channels_firstlayer)
        self.dec_2 = Decoder(8*n_channels_firstlayer, 4*n_channels_firstlayer)
        self.dec_3 = Decoder(4*n_channels_firstlayer, 2*n_channels_firstlayer)
        self.dec_4 = Decoder(2*n_channels_firstlayer, n_channels_firstlayer)

        self.final_conv = FinalConv(n_channels_firstlayer, n_channels_out)

    def forward(self, x):
        x = self.first_conv(x)
        x1 = self.second_conv(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
    
        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.final_conv(x)
        return x


class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding="same", 
                      bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)  )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size=5,
                                  stride=1, 
                                  padding="same", 
                                  bias=True),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)  )
        
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, 
                                     stride=2, 
                                     padding=0),
                        ConvBlock(in_channels, out_channels)  )

    def forward(self, x):
        x = self.encoder(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(AttentionGate, self).__init__()
        self.C_g = nn.Sequential( 
                        nn.Conv2d(F_g, 
                                  F_int,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0, #may need to use "same"
                                  bias=True)  #no batchnorm here, may need it
                        )

        self.C_x = nn.Sequential( 
                        nn.Conv2d(F_x, 
                                  F_int,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0, #may need to use "same"
                                  bias=True)  #no batchnorm here, may need it
                        )

        self.psi = nn.Sequential( 
                        nn.Conv2d(F_int, 
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0, #may need to use "same"
                                  bias=True), #no batchnorm here, may need it
                        nn.Sigmoid()  
                        )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, g, x):
        g_prime = self.C_g(g)
        x_prime = self.C_x(x)
        sigma_1 = self.leaky_relu(g_prime + x_prime)
        activation = self.psi(sigma_1)

        return x*activation
                                  

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear"),
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size=1, 
                                  stride=1, 
                                  padding="same", 
                                  bias=True),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)  )

        self.attn_gate = AttentionGate(F_g=out_channels, F_x=out_channels, F_int=int(out_channels/2)) #channels of gate and skip layer should each equal the desired output size 
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape == skip.shape:
            skip = self.attn_gate(g=x, x=skip) #truly horrid notation due to previous work + attention gate conventions - should fix at some point
            x = torch.cat((x, skip), dim=1) 
        else:
            #print(f"x shape: {x.shape} | skip shape: {skip.shape}")
            # This may need to be changed from padding the outputted layer to instead cropping the skip layer
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

            skip = self.attn_gate(g=x, x=skip) #truly horrid notation due to previous work + attention gate conventions - should fix at some point
            x = torch.cat((x, skip), dim=1) 
            
        x = self.conv_block(x)
        return x


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding="same", 
                      bias=True),
            #some activation function here? Probably not, according to Ryan Lagerquist
                )
        
    def forward(self, x):
        return self.conv(x)