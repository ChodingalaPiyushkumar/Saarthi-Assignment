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

if len(sys.argv) < 0:
    print ("missing configuration file path")

config_path = sys.argv[1]
config = ConfigParser()
config.read(config_path)    
test_csv_path=config['variables']['test_csv_path']
device_path=config['variables']['device_path']
model_load_path=config['variables']['model_save_path']


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

test_data_action = sp_1.speechdata_action(test_csv_path,device_path)
testloader_action = torch.utils.data.DataLoader(test_data_action, batch_size=1,shuffle=False)

test_data_object = sp_2.speechdata_object(test_csv_path,device_path)
testloader_object = torch.utils.data.DataLoader(test_data_object, batch_size=1,shuffle=False)

test_data_location = sp_3.speechdata_location(test_csv_path,device_path)
testloader_location = torch.utils.data.DataLoader(test_data_location, batch_size=1,shuffle=False)

action_model=config['models']['action_model']
object_model=config['models']['object_model']
location_model=config['models']['location_model']

path_action=model_load_path + action_model
path_object=model_load_path + object_model
path_location=model_load_path + location_model


model_action = cnn_1.cnn_action().to(device)
model_action.load_state_dict(torch.load(path_action))
model_action.eval()

model_object = cnn_1.cnn_action().to(device)
model_object.load_state_dict(torch.load(path_object))
model_object.eval()

model_location = cnn_1.cnn_action().to(device)
model_location.load_state_dict(torch.load(path_location))
model_location.eval()

def test_action():
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

def test_object():
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

def test_location():
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

pred_action,lab_action=test_action()
pred_object,lab_object=test_object()
pred_location, lab_location= test_location()
F1_score_action=f1_score(lab_action,pred_action,average='macro')
F1_score_object=f1_score(lab_object,pred_object,average='macro')
F1_score_location=f1_score(lab_location,pred_location,average='macro')
print('action_F1_score = {}, object_F1_score = {}, location_F1_score = {}'.format(F1_score_action,F1_score_object,F1_score_location))