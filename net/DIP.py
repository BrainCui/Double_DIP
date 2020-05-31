import torch.nn as nn
from net.Concat import *

class DIP(nn.Module):
    def __init__(self,
                 down_channels=[8, 16, 32],
                 up_channels=[8, 16, 32],
                 skip_channels=[0, 0, 0],
                 in_channels=2,
                 out_channels=3
                 ):
        super(DIP, self).__init__()
        assert len(down_channels) == len(up_channels)
        self.model = nn.Sequential()
        self.add_module('model', self.model)
        model_temp = self.model
        for i in range(len(down_channels)):

            model_temp.add_module('down_con2=v1', nn.Conv2d(in_channels=in_channels, out_channels=down_channels[i],
                                 kernel_size=3, stride=2, padding=1))  # 使用stride下采样
            model_temp.add_module('down_bn1', nn.BatchNorm2d(num_features=down_channels[i]))
            model_temp.add_module('down_relu1',nn.LeakyReLU(0.2, inplace=True))
            model_temp.add_module('down_conv2', nn.Conv2d(in_channels=down_channels[i], out_channels=down_channels[i],
                                 kernel_size=3, stride=1, padding=1))
            model_temp.add_module('down_bn2', nn.BatchNorm2d(num_features=down_channels[i]))
            model_temp.add_module('down_relu2', nn.LeakyReLU(0.2, inplace=True))



            if i == len(down_channels) - 1:  # 最深一层
                deeper = nn.Sequential(
                    nn.Conv2d(in_channels=down_channels[i], out_channels=up_channels[i],
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=up_channels[i]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                deeper = nn.Sequential()   # deeper layers

            skip = nn.Sequential()  # skip connection
            if skip_channels[i] != 0:
                # 该层的skip
                skip.add_module('conv', nn.Conv2d(in_channels=down_channels[i], out_channels=skip_channels[i],
                                   kernel_size=3, stride=1, padding=1))
                skip.add_module('bn', nn.BatchNorm2d(num_features=skip_channels[i]))
                skip.add_module('relu', nn.LeakyReLU(0.2, inplace=True))
                model_temp.add_module('skip+deeper', Concat(skip, deeper))  # 把该层的deeper和skip连接
            else:
                model_temp.add_module('deeper', deeper)

            next_channel = up_channels[i - 1] if i != 0 else out_channels
            model_temp.add_module('up_bn1', nn.BatchNorm2d(num_features=up_channels[i] + skip_channels[i]))
            model_temp.add_module('up_conv1', nn.Conv2d(in_channels=up_channels[i] + skip_channels[i],
                                     out_channels=next_channel,
                                     kernel_size=3,
                                     padding=1))
            model_temp.add_module('up_bn2', nn.BatchNorm2d(num_features=next_channel))
            model_temp.add_module('up_relu1', nn.LeakyReLU(0.2, inplace=True))
            model_temp.add_module('up_conv2', nn.Conv2d(in_channels=next_channel,
                                     out_channels=next_channel,
                                     kernel_size=3,
                                     padding=1))
            model_temp.add_module('up_bn3', nn.BatchNorm2d(num_features=next_channel))
            model_temp.add_module('up_relu2', nn.LeakyReLU(0.2, inplace=True))
            model_temp.add_module('up_upsample', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            model_temp = deeper
            in_channels = down_channels[i]
        self.model.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        return self.model(input)


class Debug(nn.Module):
    def __init__(self):
        super(Debug, self).__init__()

    def forward(self, input):
        print(input.shape)
        return input
