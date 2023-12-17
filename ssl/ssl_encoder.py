from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SimCLR(nn.Module):
    def __init__(self, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = Identity()
        self.encoder = resnet18
        self.projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)
        
        encoding = self.encoder(x)
        projection = self.projection(encoding) 
        return projection

class SSLEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, embDimension=128, poolSize=32):
        super(SSLEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']

        # self.ssl = torch.load('ssl_backbone2.pth')
        ssl = torch.load('ssl_backbone2.pth')
        
        self.encoder = ssl.encoder
            
        # self.ssl.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.num_ch_enc[1:] = 512
                    
        if self.embDimension>0:
            # self.ssl.encoder.fc =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
            self.encoder.fc =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
            

    def forward(self, input_image):
#         self.features = []
        
#         x = self.ssl.encoder.conv1(input_image)
#         x = self.ssl.encoder.bn1(x)
#         x = self.ssl.encoder.relu(x)
#         self.features.append(x)
        
#         x = self.ssl.encoder.layer1(x)
#         self.features.append(x)
        
#         x = self.ssl.encoder.layer2(x)
#         self.features.append(x)
        
#         x = self.ssl.encoder.layer3(x) 
#         self.features.append(x)
        
#         x = self.ssl.encoder.layer4(x)
#         self.features.append(x)
        
#         x = F.avg_pool2d(x, self.poolSize)
        
#         self.x = x.view(x.size(0), -1)
        
#         x = self.ssl.encoder.fc(self.x)
#         return x
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        x = self.encoder.layer1(x)
        self.features.append(x)
        
        x = self.encoder.layer2(x)
        self.features.append(x)
        
        x = self.encoder.layer3(x) 
        self.features.append(x)
        
        x = self.encoder.layer4(x)
        self.features.append(x)
        
        x = F.avg_pool2d(x, self.poolSize)
        
        self.x = x.view(x.size(0), -1)
        
        x = self.encoder.fc(self.x)
        return x