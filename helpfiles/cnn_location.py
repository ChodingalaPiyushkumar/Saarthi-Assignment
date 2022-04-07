import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_



class cnn_location(nn.Module):
    def __init__(self):
        super(cnn_location,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 5, stride = 1, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(16*24*2,200)
        self.relu6 = nn.ReLU()
        
        self.fc2 = nn.Linear(200,4)
        self.relu7 = nn.ReLU()
        
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                xavier_normal_(m.weight)
                m.bias.data.zero_()

        self.apply(_weights_init)   
            
    def forward(self,x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)        
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)        
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        out = self.maxpool3(out) 
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        #print('After forth layer:{}'.format(out.size()))
        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = self.relu6(out)
        out = self.fc2(out)
        return out