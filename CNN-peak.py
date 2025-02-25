
import torch.nn as nn
import torch

class Simple_SE(nn.Module):
    def __init__(self, channel):
        super(Simple_SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel, channel,bias=False)
        self.sigmoid = nn.Softmax(1)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y.squeeze()).squeeze().unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

#标准卷积块，Conv+BatchNorm+activation
class Conv(nn.Module):
    default_act = nn.ReLU()
    def __init__(self, c1, c2, k=5, s=2, p=0):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, p,bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class inception(nn.Module):
    def __init__(self,c1,c2):
        super(inception, self).__init__()
        self.conv1 = Conv(c1, c2, 5, 1, 0)
        self.conv2 = Conv(c1, c2, 7, 1, 1)
        self.conv3 = Conv(c1, c2, 9, 1, 2)
        self.conv11 = Conv(c1, c2, 7, 1, 1)
        self.conv12 = Conv(c1, c2, 9, 1, 2)
        self.conv21 = Conv(c1, c2, 5, 1, 0)
        self.conv22 = Conv(c1, c2, 9, 1, 2)
        self.conv31 = Conv(c1, c2, 5, 1, 0)
        self.conv32 = Conv(c1, c2, 7, 1, 1)
        self.se = Simple_SE(c2 * 6)
        self.conv = Conv(c2 * 6, c2, 5, 2, 0)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x11 = self.conv11(x1)
        x12 = self.conv12(x1)
        x21 = self.conv21(x2)
        x22 = self.conv22(x2)
        x31 = self.conv31(x3)
        x32 = self.conv32(x3)
        x = torch.cat((x11,x12,x21,x22,x31,x32),1)
        x = self.se(x)
        x = self.conv(x)
        return x

class peaknet(nn.Module):
    def __init__(self,data_channel,classes,a):
        super(peaknet, self).__init__()

        self.conv_1 = Conv(data_channel, 8 * a , 5, 1, 0)
        self.conv_2 = Conv(8 * a, 8 * a , 5, 1, 0)
        self.incept1 = inception(8 * a,8 * a)

        self.fc = nn.Linear(3744 // 2 * a, classes)
        self.droupt =nn.Dropout(0.3)


    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.incept1(x)

        x = x.flatten(start_dim=1)
        x = self.droupt(x)

        # print(x.shape)
        x = self.fc(x)
        return x






