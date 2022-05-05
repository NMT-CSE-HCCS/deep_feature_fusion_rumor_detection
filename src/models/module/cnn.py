import torch.nn as nn
from einops import rearrange

class CNN_BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, padding, pool=False):
        super(CNN_BasicBlock,self).__init__()        
        layers = [
            nn.Conv1d(inchannel, outchannel, kernel_size, stride, padding),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(True)
        ]
        if pool:
            layers += [nn.MaxPool1d(kernel_size, stride, padding)]
        self.cnn_basic_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_basic_block(x)
        return x

class CNN(nn.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()
        channel = [64, 128, 256, 512]
        kernel = [9,7,5,3]
        stride = [2,2,2,2]
        padding = [4,3,2,1]

        self.cnn_blocks = nn.Sequential(
            CNN_BasicBlock(feature_dim, channel[0],kernel[0], stride[0], padding[0], True),
            CNN_BasicBlock(channel[0], channel[1], kernel[1], stride[1], padding[1], True),
            CNN_BasicBlock(channel[1], channel[2], kernel[2], stride[2], padding[2], True),
            CNN_BasicBlock(channel[2], channel[3], kernel[3], stride[3], padding[3], False),
        )
        
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1,-1)
        self.out_dim = 512

    def forward(self, x):
        # rearrage shape "batch time feature -> batch feature time"
        x = rearrange(x, 'b t f -> b f t ')
        x = self.cnn_blocks(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        return x