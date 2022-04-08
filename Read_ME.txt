The packeges need to run this project is mention below.


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


To run the project you have to change the paths acording to your system in config files. Please make sure that path sould be given same as 
shown in config file.  

Using "train.py" file you can train the model and trained model will save at "model_save_path".
 
Then give "Test.csv" file path in config file and run "Test.py" file and it will automatically load the model and gives F1-score
as output for all three tasks.  

