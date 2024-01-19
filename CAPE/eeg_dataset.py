import sys

sys.path.append("src")
import os
import pandas as pd
import yaml
import audioldm_train.utilities.audio as Audio
from audioldm_train.utilities.tools import load_json
from audioldm_train.dataset_plugin import *
from librosa.filters import mel as librosa_mel_fn

import random
from torch.utils.data import Dataset
import torch.nn.functional
import torch
import numpy as np
import torchaudio
import json
import pandas as pd

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class EEGDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",
        waveform_only=False,
        add_ons=[],
        dataset_json_path=None, 
        path='/mnt/nvme/node02/pranav/AE24/data/split_data', 
        overide=False
        # frame_length=64,
        # hop_length=30,
            
        ):

        super(EEGDataset,self).__init__()
        self.config = config
        # self.frame_length=frame_length
        # self.hop_length=hop_length
        self.stim_dir = "/storage/pranav/data/stimuli/eeg/"
        self.text_dir="/mnt/nvme/node02/pranav/AE24/data/transcripts/"
        file_types=[]
        file_types.append(split)
        file_types.append("eeg")
        self.input_paths = [os.path.join(path,file) for  file in os.listdir(path) if all(typ in file for typ in file_types)]
        assert len(self.input_paths) != 0, 'No data found'
        self.split=split

        self.build_from_config()

        self.target_length = int(self.duration * self.sampling_rate //self.hop_length )
        self.hop_len = int(self.hop_dur * self.sampling_rate)

        self.tar_len_eeg = int( self.duration * 64)
        self.hop_len_eeg = int( self.hop_dur  * 64)
        
        self.index_map = {}
        global_index = 0
        import pickle
        data_file = f"/mnt/nvme/node02/pranav/AE24/AudioLDM-training-finetuning/data/dataset/metadata_eeg/{self.split}_dataset_dict.pkl"
        if os.path.exists(data_file) or overide:
            with open(data_file,'rb') as f:
                self.index_map=pickle.load(f)
        else:
            for file_name in self.input_paths:
                data_shape = np.load(file_name, mmap_mode='r').shape
                num_samples = data_shape[0] 

                num_windows = 1 + (num_samples - self.tar_len_eeg) //  self.hop_len_eeg  
                audio_name = self.stim_dir + os.path.basename(file_name).split('_-_')[2] + ".npz"

                audio=np.load(audio_name,mmap_mode='r')["audio"]


                transcript_full = pd.read_csv(self.text_dir + os.path.basename(file_name).split('_-_')[2] + "_p.tsv", sep='\t')
                transcript_full.fillna('', inplace=True)

                if self.split =="train":
                    split_offset = int(audio.shape[0] * 0)
                elif self.split =="val":
                    split_offset = int(audio.shape[0] * 0.8)
                elif self.split =="test":
                    split_offset = int(audio.shape[0] * 0.9)

                for window_offset in range(num_windows):

                    start=window_offset*self.hop_len_eeg 
                    end=start+self.tar_len_eeg

                    start_idx =  split_offset  +  start * 48000 //64
                    end_idx = split_offset  +  end * 48000 //64

                    start_time = start_idx/48000
                    end_time = end_idx/48000

                    windowed_text = transcript_full[(transcript_full['start'] >= start_time) & (transcript_full['end'] <= end_time)]
                    try:
                        text = ' '.join(windowed_text['word'])
                    except:
                        print("error loading",windowed_text['word'],self.text_dir + os.path.basename(file_name).split('_-_')[2] + "_p.tsv")

                    self.index_map[global_index] = (file_name, window_offset,audio_name,text)
                    global_index += 1
            
            with open(data_file,'wb') as f:
                pickle.dump(self.index_map,f)





    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        file_name, window_offset,audio_name,text=self.index_map[index]
        start=window_offset*self.hop_len_eeg 
        end=start+self.tar_len_eeg
        eeg_sample=np.load(file_name,mmap_mode='r')[start:end]
        mel_sample=np.load(file_name.replace("eeg.npy","mel.npy"),mmap_mode='r')[start:end]


        audio=np.load(audio_name,mmap_mode='r')["audio"]
        sr = int(np.load(audio_name,mmap_mode='r')["fs"])

        if self.split =="train":
            split_offset = int(audio.shape[0] * 0)
        elif self.split =="val":
            split_offset = int(audio.shape[0] * 0.8)
        elif self.split =="test":
            split_offset = int(audio.shape[0] * 0.9)
        

        start_idx =  split_offset  +  (start * 48000) //64
        end_idx = split_offset  +  (end * 48000) //64



        waveform = torch.from_numpy(audio[start_idx:end_idx].copy()).type(torch.float32)
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        waveform = waveform.numpy()
        waveform = self.normalize_wav(waveform)

        waveform = waveform[None, ...]
        log_mel_spec, stft = self.wav_feature_extraction(waveform)
        waveform = torch.FloatTensor(waveform)


        
        eeg = torch.from_numpy(eeg_sample.copy()).type(torch.float32)
        mel = torch.from_numpy(mel_sample.copy()).type(torch.float32)

        data={
            "fname": file_name,
            "eeg":eeg,
            "mel":mel,
            "waveform":waveform,
            "stft": stft.float(),
            "log_mel_spec": log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "text": text,
            "label_vector": "",
            "random_start_sample_in_original_audio_file": start_idx

        }

        return data


    def build_from_config(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.dataset_name = self.config["data"][self.split]
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        # self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        # self.timem = self.config["preprocessing"]["mel"]["timem"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_dur = self.config["preprocessing"]["audio"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.mixup = self.config["augmentation"]["mixup"]

        self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]
        self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]

        self.STFT = Audio.stft.TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )




    def normalize_wav(self, waveform):
        
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5 
    
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft
    
    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec
    
    def mel_spectrogram_train(self, y):

        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel,
                self.mel_fmin,
                self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = dynamic_range_compression_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]




    def read_audio_file(self, filename, filename2=None):
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            print(
                'Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        mix_lambda = 0.0
        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, mix_lambda, waveform, random_start



    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        if ".npz" in filename:
            obj = np.load(filename)
            waveform,sr = obj["audio"],int(obj["fs"])
            waveform = waveform[None,:]
        else:
            waveform, sr = torchaudio.load(filename)

        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, random_start
    


if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(0)

    def write_json(my_dict, fname):
        # print("Save json file at "+fname)
        json_str = json.dumps(my_dict)
        with open(fname, "w") as json_file:
            json_file.write(json_str)

    def load_json(fname):
        with open(fname, "r") as f:
            data = json.load(f)
            return data

    config = yaml.load(
        open(
            "/mnt/nvme/node02/pranav/AE24/AudioLDM-training-finetuning/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_eeg.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )
    print("config loaded")
    add_ons = config["data"]["dataloader_add_ons"]

    # load_json(data)
    dataset = EEGDataset(
        config=config, split="train", waveform_only=False, add_ons=add_ons
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    for cnt, each in tqdm(enumerate(loader)):
        print(each["waveform"].size(), each["log_mel_spec"].size(),each["text"])
        break
        # print(each['freq_energy_percentile'])
        # import ipdb

        # ipdb.set_trace()
        # pass

    waveform = each["waveform"][0][0, ...]
    waveform = torch.FloatTensor(waveform)

    # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
    log_mel_spec, stft = dataset.mel_spectrogram_train(waveform.unsqueeze(0))
    print(log_mel_spec.shape)

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = dataset.pad_spec(log_mel_spec), dataset.pad_spec(stft)

    print(log_mel_spec.shape)
