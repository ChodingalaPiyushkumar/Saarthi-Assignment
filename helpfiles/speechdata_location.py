import torch.utils.data
import numpy as np
from python_speech_features import mfcc, delta, logfbank
from torch.utils import data
from scipy.io import savemat, wavfile
from . import util

class speechdata_location(data.Dataset):
    def __init__(self, csv_path,device_path):
        self.path, self.action, self.object, self.location =util.generate_labels_fnames(csv_path,device_path)
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
        feat=util.pad(feat.T)
        d_feat=delta(feat,3)
        dd_feat=delta(d_feat,3)
        x_s_d_dd = np.concatenate((feat,d_feat,dd_feat),axis=0)
        return x_s_d_dd, l
    
    def __len__(self):
        return len(self.targets_data)