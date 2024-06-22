import torch
from torch import nn
from torch.nn import functional as F

class EEGNet(nn.Module):
        def __init__(self,
                     nb_classes, 
                     Chans = 16, 
                     Samples = 1024,
                     dropoutRate = 0.5,
                     kernLength = 512,
                     F1 = 8, 
                     D = 2, 
                     F2 = 16, 
                      ) -> None:
                super().__init__()

                self.conv = nn.Conv2d(1,F1,kernel_size=(1, kernLength),padding=(0, kernLength // 2), bias = False)
                self.bn1 = nn.BatchNorm2d(F1)
                self.depth_conv = nn.Conv2d(F1,F1 * D, kernel_size=(Chans,1), groups=F1,bias=False)
                self.bn2 = nn.BatchNorm2d(F1*D)
                self.drop1 = nn.Dropout2d(p=dropoutRate)

                self.point_conv = nn.Conv2d(F1 * D, F2, kernel_size=(1,16),padding=(0, 8), bias=False)
                self.bn3 = nn.BatchNorm2d(F2)
                self.drop2 = nn.Dropout2d(p=dropoutRate)
                self.fc = nn.Linear(F2*(Samples//32),nb_classes)

        def forward(self,x):
                h = self.conv(x)
                h = self.bn1(h)
                h = self.depth_conv(h)
                h = self.bn2(h)
                h = F.elu(h)
                h = F.avg_pool2d(h,(1, 4))
                h = self.drop1(h)
                h = self.point_conv(h)
                h = self.bn3(h)
                h = F.elu(h)
                h = F.avg_pool2d(h,(1, 8))
                h = self.drop2(h)
                h = torch.flatten(h,start_dim=1)
                h = self.fc(h)
                return h
