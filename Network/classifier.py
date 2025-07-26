import torch
from torch import nn

from attention import AttnAugmentation2d


'利用Attention辅助分类（SENet）'
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)


'定义卷积块'
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.pool(y)
        return y


'定义残差卷积块'
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding='same')
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding='same')

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.conv(y)
        y = torch.add(x, y)
        return y


'定义注意力机制块'
class AttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttnBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')
        self.qkv = nn.Conv2d(in_channels=in_channels, out_channels=3*4, kernel_size=1, stride=1, padding='same')
        self.attn = AttnAugmentation2d(4, 4, 1, relative=True)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=in_channels+4, out_channels=out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        y = self.conv1(x)
        y1 = self.qkv(x)
        y1 = self.attn(y1)
        y1 = self.conv2(y1)
        y = torch.cat((y,y1), dim=1)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y
        return y


'定义用于提取CSI特征的模块'
class CSIBlock(nn.Module):
    def __init__(self, in_channels):
        super(CSIBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1)
        self.se = SELayer(channels=64)
        self.res1 = ResBlock(in_channels=64, out_channels=64, kernel=3, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.res2 = ResBlock(in_channels=64, out_channels=64, kernel=3, stride=1)
        self.attn1 = AttnBlock(in_channels=64, out_channels=64)
        self.attn2 = AttnBlock(in_channels=64, out_channels=64)

    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        y = self.res1(y)
        y = self.pool1(y)
        y = self.res2(y)
        y = self.attn1(y)
        y = self.attn2(y)
        return y


'定于用于提取空间谱特征的模块'
class SPBlock(nn.Module):
    def __init__(self, in_channels):
        super(SPBlock, self).__init__()
        self.layer1 = ConvBlock(in_channels=in_channels, out_channels=16, kernel_size=3)
        self.layer2 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3)
        self.layer3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        return y


'分别设计模型处理CSI和空间谱数据'
class CSISPClassifier(nn.Module):
    def __init__(self):
        super(CSISPClassifier, self).__init__()
        self.csi_block = CSIBlock(in_channels=2)
        self.sp_block = SPBlock(in_channels=1)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, y):
        m = self.csi_block(x)
        n = self.sp_block(y)

        n = nn.functional.adaptive_avg_pool2d(n, output_size=(3,3))
        z = m + n
        z = self.conv1(z)
        z = self.relu2(z)
        z = self.conv2(z)
        z = z.view(-1, 64 * 3 * 3)
        z = self.fc1(z)
        z = self.relu3(z)
        z = self.fc2(z)
        z = torch.sigmoid(z)
        return z