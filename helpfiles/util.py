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

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# padding is done to make constant dimension of data
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
# labels generation
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

def get_optimizer(model_name, lr):
    return optim.Adam(model_name.parameters(), lr=lr)

# Training scripts
def train_action(epoch,model_action,trainloader_action,criterion, testloader_action,optimizer_action):
    model_action.train()
    total_loss = 0.0

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


def train_object(epoch,model_object,trainloader_object,criterion, testloader_object,optimizer_object):
    model_object.train()
    total_loss = 0.0
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
def train_location(epoch,model_location,trainloader_location,criterion, testloader_location,optimizer_location):
    model_location.train()
    total_loss = 0.0
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
    
