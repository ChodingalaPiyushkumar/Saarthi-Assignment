import torch.utils.data
import numpy as np
from python_speech_features import mfcc, delta, logfbank
from torch.utils import data
from scipy.io import savemat, wavfile
from . import util


class speechdata_action(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location = util.generate_labels_fnames(csv_path,device_path)
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
        feat=mfcc(audio_signal,frequency_sampling)    # MFCC extraction 
        feat=util.pad(feat.T)    # padding is applied
        d_feat=delta(feat,3)      # Delta coefficients calculation
        dd_feat=delta(d_feat,3)   # Double Delta coefficients calculation
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)  # concatinate static, delta, and double-delta coefficients
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data) 
