
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
import os
default_path='/mnt/nvme/node02/pranav/AE24/data/split_data'




class EEG_Dataset(Dataset):
    def __init__(self, path='/mnt/nvme/node02/pranav/AE24/data/split_data', file_types=["train"],frame_length=64,hop_length=30):
        
        
        super(EEG_Dataset, self).__init__()
        self.frame_length=frame_length
        self.hop_length=hop_length

        file_types.append("eeg")
        self.input_paths = [os.path.join(path,file) for  file in os.listdir(path) if all(typ in file for typ in file_types)]
        assert len(self.input_paths) != 0, 'No data found'
        # print(f"loaded {len(self.input_paths)} files with file types {file_types}")


        self.index_map = {}
        global_index = 0
        for file_name in self.input_paths:
            data_shape = np.load(file_name, mmap_mode='r').shape
            num_samples = data_shape[0] 

            num_windows = 1 + (num_samples - self.frame_length) // self.hop_length

            for window_offset in range(num_windows):
                self.index_map[global_index] = (file_name, window_offset)
                global_index += 1
        


    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self,idx):
        file_name, window_offset=self.index_map[idx]
        start=window_offset*self.hop_length
        end=start+self.frame_length
        sample=np.load(file_name,mmap_mode='r')[start:end]

        return torch.from_numpy(sample.copy())
    

class maeeg_dataset(EEG_Dataset):
    def __init__(self, path='/mnt/nvme/node02/pranav/AE24/data/split_data', file_types=["train"],frame_length=64,hop_length=30):
        
        
        super(maeeg_dataset, self).__init__(path, file_types,frame_length,hop_length)
    
    def __getitem__(self,idx):
        file_name, window_offset=self.index_map[idx]
        start=window_offset*self.hop_length
        end=start+self.frame_length
        sample=np.load(file_name,mmap_mode='r')[start:end]

        return torch.from_numpy(sample.copy())
    

class meleeg_dataset(EEG_Dataset):
    def __init__(self, path='/mnt/nvme/node02/pranav/AE24/data/split_data', file_types=["train"],frame_length=120,hop_length=30):
        
        
        super(meleeg_dataset, self).__init__(path, file_types,frame_length,hop_length)
    
    def __getitem__(self,idx):
        file_name, window_offset=self.index_map[idx]
        start=window_offset*self.hop_length
        end=start+self.frame_length
        eeg_sample=np.load(file_name,mmap_mode='r')[start:end]
        mel_sample=np.load(file_name.replace("eeg.npy","mel.npy"),mmap_mode='r')[start:end]

        return torch.from_numpy(eeg_sample.copy()).type(torch.float32),torch.from_numpy(mel_sample.copy())#torch.from_numpy(env_sample.copy())