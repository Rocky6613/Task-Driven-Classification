
import torch.nn as nn
import torch

from common import Conv
import torch.nn.functional as F
class ConvAttention(nn.Module):
    def __init__(self, in_channels, channels, reduction_factor=4):
        super(ConvAttention, self).__init__()
        inter_channels = max(in_channels//reduction_factor, 16)
        self.channels = channels
        self.fc1 = Conv(in_channels,inter_channels,1)
        self.fc2 = Conv(inter_channels,channels,1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        y = x
        y = F.adaptive_avg_pool1d(y, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.softmax(y)
        out = y.expand_as(x) * x
        return out

class inception(nn.Module):
    def __init__(self,c1,c2):
        super(inception, self).__init__()
        self.conv1 = Conv(c1,c1//2,5,1)
        self.conv2 = Conv(c1,c1//2,7,1)
        self.conv3 = Conv(c1,c1//2,9,1)


        self.conv_2 = Conv(c1 // 2 * 3,c2,5,2)

        self.se = ConvAttention(c1 // 2 * 3,c1  // 2 * 3)

        self.droupt = nn.Dropout(0.3)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_2 = torch.cat((x1,x2,x3),1)
        x_2 = self.se(x_2)
        x = self.conv_2(x_2)

        return x

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.cv1 = inception(c1,c2)
        self.cv = nn.Sequential(
            nn.Conv1d(c1, c2, 1, 2, 0),
            nn.BatchNorm1d(c2),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.cv1(x) + self.cv(x)
        return y



class resunit(nn.Module):
    def __init__(self,data_channel,classes,a):
        super(resunit, self).__init__()
        self.conv0 = Conv(data_channel,8 * a,5,2)
        self.conv_2 = Bottleneck(8 * a, 8 * a)
        self.conv_3 = Bottleneck(8 * a, 8 * a)
        self.conv_4 = Bottleneck(8 * a, 8 * a)
        self.conv_5 = Bottleneck(8 * a, 8 * a)
        self.conv_6 = Bottleneck(8 * a, 8 * a)
        self.conv_7 = Bottleneck(8 * a, 8 * a)
        self.fc = nn.Linear(176 * a, classes)
        # self.fc = nn.Linear(72 * a, classes)
        self.droupt =nn.Dropout(0.1)

    def forward(self,x):
        x = self.conv0(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = x.flatten(1)
        x = self.droupt(x)
        # print(x.shape)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    from read_data import read_data
    from data.compute.train import evaluate,run
    import os

    save_path = '../../../data\\compute2\\respeak_simple\\1'
    root_path = r'../../../datasets\compute2\train_x.npy'
    if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

    x_train, x1, x_t, x_val = read_data(root_path, 0)

    run(x_train, x_t, save_path, 'resunit(1,10,20)', 2, 1, 80)

    result = evaluate(x_t, save_path, 1)
    print(result)

