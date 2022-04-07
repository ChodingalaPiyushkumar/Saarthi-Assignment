import numpy as np
import pandas as pd
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import wave, os, glob
import torch.utils.data
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import shutil
import scipy.io as io
import os, sys
from scipy.io import savemat, wavfile
from hdf5storage import loadmat
from torch.autograd import Variable
from torch.utils import data
from torch.nn.init import xavier_normal_
from python_speech_features import mfcc, delta, logfbank
from sklearn.metrics import f1_score
from configparser import ConfigParser
import sys

if len(sys.argv) < 0:
    print ("missing configuration file path")

config_path = sys.argv[1]
config = ConfigParser()
config.read(config_path)    
train_csv_path=config['variables']['train_csv_path']
test_csv_path=config['variables']['test_csv_path']
device_path=config['variables']['device_path']


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# train_csv_path='/home/speechlab/Desktop/Piyush_try/task_data/train_data.csv'
# test_csv_path='/home/speechlab/Desktop/Piyush_try/task_data/valid_data.csv'
# device_path='/home/speechlab/Desktop/Piyush_try/task_data/'

def pad(S):
    if(S.shape[1]<400):    
        while(1):
            S = np.concatenate((S,S),axis=1)            
            if(S.shape[1]==400):
                break
            elif(S.shape[1]>400):
                S = np.array([list(x[0:400]) for x in S])
                break 
    elif(S.shape[1]>400):        
        S = np.array([list(x[0:400]) for x in S])
        
    return S

def generate_labels_fnames(file_description,device_path):
    df=pd.read_csv(file_description)
    path_initial=df["path"]
    path=[]
    for i in path_initial:
        path_line=device_path+i
        path.append(path_line)
    action=df["action"]
    object=df["object"]
    location=df["location"]
    #print(location[43]) 
    
    return path,action,object,location

class speechdata_action(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location = generate_labels_fnames(csv_path,device_path)
        self.targets_data = np.array(self.action)
        
    def __getitem__(self, index):
        if (self.targets_data[index])=='activate':
            l=0
        if (self.targets_data[index])=='increase':
            l=1    
        if (self.targets_data[index])=='change language':
            l=2    
        if (self.targets_data[index])=='decrease':
            l=3
        if (self.targets_data[index])=='deactivate':
            l=4
        if (self.targets_data[index])=='bring':
            l=5
        a = self.path[index]
        frequency_sampling, audio_signal = wavfile.read(a)
        feat=mfcc(audio_signal,frequency_sampling)
        feat=pad(feat.T)
        d_feat=delta(feat,3)
        dd_feat=delta(d_feat,3)
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data) 

class speechdata_object(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location = generate_labels_fnames(csv_path,device_path)
        self.targets_data = np.array(self.object)
        
    def __getitem__(self, index):
        if (self.targets_data[index])=='lights':
            l=0
        if (self.targets_data[index])=='heat':
            l=1    
        if (self.targets_data[index])=='Chinese':
            l=2    
        if (self.targets_data[index])=='none':
            l=3
        if (self.targets_data[index])=='volume':
            l=4
        if (self.targets_data[index])=='English':
            l=5
        if (self.targets_data[index])=='lamp':
            l=6
        if (self.targets_data[index])=='shoes':
            l=7    
        if (self.targets_data[index])=='newspaper':
            l=8    
        if (self.targets_data[index])=='socks':
            l=9
        if (self.targets_data[index])=='music':
            l=10
        if (self.targets_data[index])=='Korean':
            l=11
        if (self.targets_data[index])=='juice':
            l=12
        if (self.targets_data[index])=='German':
            l=13       
        a = self.path[index]
        frequency_sampling, audio_signal = wavfile.read(a)
        feat=mfcc(audio_signal,frequency_sampling)
        feat=pad(feat.T)
        d_feat=delta(feat,3)
        dd_feat=delta(d_feat,3)
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data)

class speechdata_location(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location = generate_labels_fnames(csv_path,device_path)
        self.targets_data = np.array(self.location)
        
    def __getitem__(self, index):
        if (self.targets_data[index])=='kitchen':
            l=0
        if (self.targets_data[index])=='none':
            l=1    
        if (self.targets_data[index])=='washroom':
            l=2    
        if (self.targets_data[index])=='bedroom':
            l=3
        a = self.path[index]
        frequency_sampling, audio_signal = wavfile.read(a)
        feat=mfcc(audio_signal,frequency_sampling)
        feat=pad(feat.T)
        d_feat=delta(feat,3)
        dd_feat=delta(d_feat,3)
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data)

class cnn_action(nn.Module):
    def __init__(self):
        super(cnn_action,self).__init__()
        
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
        
        self.fc2 = nn.Linear(200,6)
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
        return out.to(device)
 
class cnn_object(nn.Module):
    def __init__(self):
        super(cnn_object,self).__init__()
        
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
        
        self.fc2 = nn.Linear(200,14)
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
        return out.to(device)

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
        return out.to(device)

train_data_action = speechdata_action(train_csv_path,device_path)
trainloader_action = torch.utils.data.DataLoader(train_data_action, batch_size=32,
                                          shuffle=True)
train_data_object = speechdata_object(train_csv_path,device_path)
trainloader_object = torch.utils.data.DataLoader(train_data_object, batch_size=32,
                                          shuffle=True)
train_data_location = speechdata_location(train_csv_path,device_path)
trainloader_location = torch.utils.data.DataLoader(train_data_location, batch_size=32,
                                          shuffle=True)   
test_data_action = speechdata_action(test_csv_path,device_path)
testloader_action = torch.utils.data.DataLoader(test_data_action, batch_size=1,shuffle=False)

test_data_object = speechdata_object(test_csv_path,device_path)
testloader_object = torch.utils.data.DataLoader(test_data_object, batch_size=1,shuffle=False)

test_data_location = speechdata_location(test_csv_path,device_path)
testloader_location = torch.utils.data.DataLoader(test_data_location, batch_size=1,shuffle=False)

model_action = cnn_action().to(device)
model_object = cnn_object().to(device)
model_location = cnn_location().to(device)

def get_optimizer(model_name, lr):
    return optim.Adam(model_name.parameters(), lr=lr)
optimizer_action = get_optimizer(model_action, lr=0.001)
optimizer_object = get_optimizer(model_object, lr=0.001)
optimizer_location = get_optimizer(model_location, lr=0.001)
count = 1
########################################################################################################################


criterion = nn.CrossEntropyLoss()
curr_lr = 0.0001

def train_action(epoch):
    model_action.train()
    total_loss = 0.0
    global optimizer_action
    global prev_loss
    global count
# Model Training For Action to be Taken Class    
    for batch_idx, (data, target) in enumerate(trainloader_action):
        data = data.unsqueeze_(1)
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer_action.zero_grad()
        output = model_action(data.float()).to(device)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer_action.step()
    model_action.eval()
    pred_action = []
    lab_action =[]
    for data, devtarget in testloader_action:
        data = data.unsqueeze_(1)
        data, target = Variable(data).float().to(device), Variable(devtarget).to(device)
        output = model_action(data).to(device)
        _,predicted = torch.max(output.data,1)
        predicted = predicted.cpu().numpy() 
        target=target.cpu().data.numpy()
        pred_action.append(predicted)
        lab_action.append(target)
    return pred_action,lab_action

def train_object(epoch):
    model_object.train()
    total_loss = 0.0
    global optimizer_object
    global prev_loss
    global count
# Model Training For Which Object the Action to be Taken Class 
    for batch_idx, (data, target) in enumerate(trainloader_object):
        data = data.unsqueeze_(1)
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer_object.zero_grad()
        output = model_object(data.float()).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer_object.step()
    model_object.eval()
    pred_object = []
    lab_object =[]
    for data, devtarget in testloader_object:
        data = data.unsqueeze_(1)
        data, target = Variable(data).float().to(device), Variable(devtarget).to(device)
        output = model_object(data).to(device)
        _,predicted = torch.max(output.data,1)
        predicted = predicted.cpu().numpy()   
        target=target.cpu().data.numpy()
        pred_object.append(predicted)
        lab_object.append(target)
    return pred_object,lab_object

# Model Training For Which Object the Action to be Taken Class 
def train_location(epoch):
    model_location.train()
    total_loss = 0.0
    global optimizer_location
    global prev_loss
    global count
    for batch_idx, (data, target) in enumerate(trainloader_location):
        data = data.unsqueeze_(1)
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer_location.zero_grad()
        output = model_location(data.float()).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer_location.step()
    model_location.eval()
    pred_location = []
    lab_location =[]
    for data, devtarget in testloader_location:
        data = data.unsqueeze_(1)
        data, target = Variable(data).float().to(device), Variable(devtarget).to(device)
        output = model_location(data).to(device)
        _,predicted = torch.max(output.data,1)
        predicted = predicted.cpu().numpy()   
        target=target.cpu().data.numpy()
        pred_location.append(predicted)
        lab_location.append(target) 
    return pred_location,lab_location
model_save_path=config['variables']['model_save_path']
min_F1_action=0
min_F1_object=0
min_F1_location=0

for epoch in range(1,2):
    pred_action,lab_action=train_action(epoch)
    pred_object,lab_object=train_object(epoch)
    pred_location, lab_location= train_location(epoch)
    F1_score_action=f1_score(lab_action,pred_action,average='macro')
    if (min_F1_action<F1_score_action):
        min_F1_action=F1_score_action
        torch.save(model_action.state_dict(),model_save_path + str(epoch) + '_action.pth')

    F1_score_object=f1_score(lab_object,pred_object,average='macro')
    if (min_F1_object<F1_score_object):
        min_F1_object=F1_score_object
        torch.save(model_action.state_dict(),model_save_path + str(epoch) + '_object.pth')
    F1_score_location=f1_score(lab_location,pred_location,average='macro')     
    if (min_F1_location<F1_score_location):  
        min_F1_location=F1_score_location
        torch.save(model_action.state_dict(),model_save_path + str(epoch) + '_location.pth')    
    