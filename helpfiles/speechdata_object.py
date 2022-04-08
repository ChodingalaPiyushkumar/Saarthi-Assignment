import torch.utils.data
import numpy as np
from python_speech_features import mfcc, delta, logfbank
from torch.utils import data
from scipy.io import savemat, wavfile
from . import util


class speechdata_object(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location = util.generate_labels_fnames(csv_path,device_path)
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
        feat=mfcc(audio_signal,frequency_sampling)  # MFCC extraction
        feat=util.pad(feat.T)  # padding is done
        d_feat=delta(feat,3)   # Delta coefficients calculation
        dd_feat=delta(d_feat,3)   # Double-delta coefficients clculation
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data)
