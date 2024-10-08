import torch
import torch.nn as nn
# from torchinfo import summary
# from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict

'''
come from https://sahiltinky94.medium.com/know-about-mobilenet-v1-implementation-from-scratch-using-pytorch-7dd594f4d99b
'''

class Depthwise_Conv(nn.Module):
    def __init__(self, in_fts, stride=(1, 1)):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=in_fts),
            nn.BatchNorm2d(in_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Pointwise_Conv(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_fts, out_fts, stride=(1, 1)):
        super(Depthwise_Separable_Conv, self).__init__()
        self.dw = Depthwise_Conv(in_fts=in_fts, stride=stride)
        self.pw = Pointwise_Conv(in_fts=in_fts, out_fts=out_fts)

    def forward(self, input_image):
        x = self.pw(self.dw(input_image))
        return x


class MyMobileNet_v1(nn.Module):
    def __init__(self, in_fts=3, num_filter=32, num_classes=1000):
        super(MyMobileNet_v1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, num_filter, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

        self.in_fts = num_filter

        # if type of sublist is list --> means make stride=(2,2)
        # also check for length of sublist
        # if length = 1 --> means stride=(2,2)
        # if length = 2 --> means (num_times, num_filter)
        self.nlayer_filter = [
            num_filter * 2,  # no list() type --> default stride=(1,1)
            [num_filter * pow(2, 2)],  # list() type and length is 1 --> means put stride=(2,2)
            num_filter * pow(2, 2),
            [num_filter * pow(2, 3)],
            num_filter * pow(2, 3),
            [num_filter * pow(2, 4)],
            # list() type --> check length for this list = 2 --> means (n_times, num_filter)
            [5, num_filter * pow(2, 4)],
            [num_filter * pow(2, 5)],
            num_filter * pow(2, 5)
        ]

        self.DSC = self.layer_construct()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.conv(input_image)
        x = self.DSC(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        return x

    def layer_construct(self):
        block = OrderedDict()
        index = 1
        for l in self.nlayer_filter:
            if type(l) == list:
                if len(l) == 2:  # (num_times, out_channel)
                    for _ in range(l[0]):
                        block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l[1])
                        index += 1
                else:  # stride(2,2)
                    block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l[0], stride=(2, 2))
                    self.in_fts = l[0]
                    index += 1
            else:
                block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l)
                self.in_fts = l
                index += 1

        return nn.Sequential(block)


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    model = MyMobileNet_v1(num_classes=9)
    # summary(model, (1, 3, 224, 224))