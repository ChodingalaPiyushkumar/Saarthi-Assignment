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
from helpfiles import util
from helpfiles import cnn_action as cnn_1
from helpfiles import cnn_object as cnn_2
from helpfiles import cnn_location as cnn_3
from helpfiles import speechdata_action as sp_1
from helpfiles import speechdata_object as sp_2
from helpfiles import speechdata_location as sp_3
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 0:
    print ("missing configuration file path")

config_path = sys.argv[1]
config = ConfigParser()
config.read(config_path)    
train_csv_path=config['variables']['train_csv_path']
test_csv_path=config['variables']['valid_csv_path']
device_path=config['variables']['device_path']



train_data_action = sp_1.speechdata_action(train_csv_path,device_path)
trainloader_action = torch.utils.data.DataLoader(train_data_action, batch_size=32,
                                          shuffle=True)
train_data_object = sp_2.speechdata_object(train_csv_path,device_path)
trainloader_object = torch.utils.data.DataLoader(train_data_object, batch_size=32,
                                          shuffle=True)
train_data_location = sp_3.speechdata_location(train_csv_path,device_path)
trainloader_location = torch.utils.data.DataLoader(train_data_location, batch_size=32,
                                          shuffle=True)   
test_data_action = sp_1.speechdata_action(test_csv_path,device_path)
testloader_action = torch.utils.data.DataLoader(test_data_action, batch_size=1,shuffle=False)

test_data_object = sp_2.speechdata_object(test_csv_path,device_path)
testloader_object = torch.utils.data.DataLoader(test_data_object, batch_size=1,shuffle=False)

test_data_location = sp_3.speechdata_location(test_csv_path,device_path)
testloader_location = torch.utils.data.DataLoader(test_data_location, batch_size=1,shuffle=False)

model_action = cnn_1.cnn_action().to(device)
model_object = cnn_2.cnn_object().to(device)
model_location = cnn_3.cnn_location().to(device)

optimizer_action = util.get_optimizer(model_action, lr=0.001)
optimizer_object = util.get_optimizer(model_object, lr=0.001)
optimizer_location = util.get_optimizer(model_location, lr=0.001)

criterion = nn.CrossEntropyLoss()
curr_lr = 0.0001
model_save_path=config['variables']['model_save_path']
min_F1_action=0
min_F1_object=0
min_F1_location=0

for epoch in range(1,2):
    pred_action,lab_action=util.train_action(epoch,model_action,trainloader_action,criterion, testloader_action,optimizer_action)
    pred_object,lab_object=util.train_object(epoch,model_object,trainloader_object,criterion, testloader_object,optimizer_object)
    pred_location, lab_location= util.train_location(epoch,model_location,trainloader_location,criterion, testloader_location,optimizer_location)
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
    
