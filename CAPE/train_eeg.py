import torch.nn as nn
from audioldm_train.modules.clap.open_clip import create_model
from audioldm_train.modules.clap.training.data import get_audio_features
import torchaudio
import torch.optim as optim
import torch
from audioldm_train.modules.eeg_encoder.config import Config_MBM_EEG
from audioldm_train.modules.eeg_encoder.mae_for_eeg import *
from audioldm_train.modules.eeg_encoder.eeg_dataset import EEGDataset
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class EEG2Latent(nn.Module):
    def __init__(self,device="cpu",freeze=True):
        super(EEG2Latent,self).__init__()


        ckpt= "/mnt/nvme/node02/pranav/AE24/DreamDiffusion/dreamdiffusion/results/eeg_pretrain/21-11-2023-05-58-07/checkpoints/checkpoint.pth"

        state_dict=torch.load(ckpt,map_location=device)
        config =Config_MBM_EEG()



        self.eeg_encoder = eeg_encoder(512,4,512,64,24,16,1).to(device)
        self.eeg_encoder.load_state_dict(state_dict,strict=False)

        if freeze:
            for param in self.eeg_encoder.parameters():
                param.requires_grad =False
        
        self.eeg_projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(128*512,512),
            nn.ReLU(),
            nn.Linear(512,512),
        ).to(device)

    def forward(self,x):
        x=self.eeg_encoder.forward(x.permute(0,2,1))
        x=self.eeg_projection(x)
        # x=self.linear(x)
        return x




class CLAPAudioEmbedding(nn.Module):
    def __init__(
        self,
        pretrained_path="/mnt/nvme/node02/pranav/AE24/AudioLDM-training-finetuning/data/checkpoints/clap_music_speech_audioset_epoch_15_esc_89.98.pt",
        sampling_rate=16000,
        embed_mode="audio",
        amodel="HTSAT-base",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=True,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute

        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )
        audio_cfg = self.model_cfg["audio_cfg"]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            win_length=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.unconditional_token = None
        self.model.eval()

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()


    def forward(self, batch):

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        with torch.no_grad():
            if self.sampling_rate != 48000:
                batch = torchaudio.functional.resample(
                    batch, orig_freq=self.sampling_rate, new_freq=48000
                )

            audio_data = batch.squeeze(1)
            mel = self.mel_transform(audio_data)
            audio_dict = get_audio_features(
                audio_data,
                mel,
                380000,
                data_truncating="fusion",
                data_filling="repeatpad",
                audio_cfg=self.model_cfg["audio_cfg"],
            )
            
            embed = self.model.get_audio_embedding(audio_dict)

        embed = embed.unsqueeze(1)
        return embed.detach()


def lossfn(Aemb,Bemb,temperature=1):

    # Calculating the Loss
    logits = (Aemb @ Bemb.T) / temperature
    Asimilarity = Aemb @ Aemb.T
    Bsimilarity = Bemb @ Bemb.T
    targets = F.softmax(
        (Asimilarity + Bsimilarity) / 2 * temperature, dim=-1
    )
    Aloss = cross_entropy(logits, targets, reduction='none')
    Bloss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (Aloss + Bloss) / 2.0 # shape: (batch_size)
    return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def main():

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    configs= yaml.load(open("/mnt/nvme/node02/pranav/AE24/AudioLDM-training-finetuning/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_eeg.yaml", "r"), Loader=yaml.FullLoader)
    dataset = EEGDataset(configs,split="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset,batch_size =480,num_workers=10,sampler=train_sampler)

    vdataset = EEGDataset(configs,split="val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(vdataset)
    vloader = DataLoader(vdataset, batch_size =480,num_workers=10,sampler=val_sampler)

    device = torch.device("cuda")

    audio_model = CLAPAudioEmbedding().cuda(local_rank)
    # audio_model.cuda(local_rank)
    # audio_model = DDP(audio_model, device_ids=[local_rank])

    ckpt = "/mnt/nvme/node02/pranav/AE24/starDuck/CAPE/models/1eegLatentmodel.pt"

    ckpt="/mnt/nvme/node02/pranav/AE24/starDuck/CAPE/models/eegLatentmodel_Linfonce.pth"

    if os.path.exists(ckpt):

        # state_dict=torch.load(ckpt,map_location=device)
        # eeg_model.load_state_dict(state_dict,strict=False)
        # eeg_model = torch.load(ckpt,map_location=device)
        # for param in eeg_model.parameters():
        #     param.requires_grad =False

        eeg_model= EEG2Latent()
        eegmodel= torch.load(ckpt,map_location=device)
        eeg_model.load_state_dict(eegmodel)
        print("******************************loaded**********************************")

    else:
        eeg_model= EEG2Latent().to(device)
    
    eeg_model.cuda(local_rank)
    eeg_model = DDP(eeg_model, device_ids=[local_rank],find_unused_parameters=True)


    optimizer = optim.Adam(eeg_model.parameters(), lr=5e-4)
    cosine = nn.CosineSimilarity()
    mse = nn.MSELoss()

    # num_epochs =10
    # for epoch in range(num_epochs):
    #     avg_loss=0
    #     for batch in tqdm(loader):
    #         optimizer.zero_grad()
    #         eeglatent = eeg_model(batch["eeg"].to(device))
    #         with torch.no_grad():
    #             audiolatent = torch.squeeze(audio_model(batch["waveform"]))

    #         loss = (1- cosine(eeglatent,audiolatent.to(device))).mean()
            
    #         optimizer.zero_grad()

    #         loss.backward()
    #         optimizer.step()
    #         avg_loss +=loss.item()
    #     avg_loss /= len(loader)
    #     print(f"Epoch:{epoch}/{num_epochs} ; Loss: {avg_loss}")
    # if local_rank==0:
    #     torch.save(eeg_model.module.state_dict(), "/mnt/nvme/node02/pranav/AE24/starDuck/CAPE/models/eegLatentmodel_test.pth")
    # return
    import time
    num_epochs = 20
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = 0
        eeg_model.train()
        for i, batch in enumerate(tqdm(loader)):

            # Data loading
            eeg_data = batch["eeg"].cuda(local_rank)
            waveform_data = batch["waveform"].cuda(local_rank)


            optimizer.zero_grad()


            eeglatent = eeg_model(eeg_data)

            audiolatent = torch.squeeze(audio_model(waveform_data))

            # Loss calculation and backward pass
            start_time = time.time()
            loss = lossfn(eeglatent, audiolatent)
            #mse(eeglatent, audiolatent.to(device))+(1 - cosine(eeglatent, audiolatent.to(device))).mean()

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            
            # Print times for each batch
            # print(f"Batch {i+1}/{len(loader)}: Data Loading Time: {data_loading_time:.6f} sec, Forward Pass Time: {forward_pass_time:.6f} sec, ffpt {ffpt :.6f} sec, Backward Pass Time: {backward_pass_time:.6f} sec")

        avg_loss /= len(loader)
        # print(f"Epoch: {epoch+1}/{num_epochs}; Loss: {avg_loss:.4f}")

        eeg_model.eval()
        val_loss=0
        with torch.no_grad():
            for i, batch in enumerate(vloader):

                # Data loading
                eeg_data = batch["eeg"].cuda(local_rank)
                waveform_data = batch["waveform"].cuda(local_rank)

                eeglatent = eeg_model(eeg_data)
                audiolatent = torch.squeeze(audio_model(waveform_data))

                loss = lossfn(eeglatent, audiolatent)

                val_loss += loss.item()
                
            val_avg_loss = val_loss/len(vloader)

        print(f"Epoch: {epoch+1}/{num_epochs}; Train Loss: {avg_loss:.4f}; Val Loss: {val_avg_loss:.4f}")

        if not epoch%5:
            if local_rank ==0:
                torch.save(eeg_model.module.state_dict(),"/mnt/nvme/node02/pranav/AE24/starDuck/CAPE/models/eegLatentmodel_Linfonce.pth")
            dist.barrier()
    
    if local_rank==0:
        torch.save(eeg_model.module.state_dict(),"/mnt/nvme/node02/pranav/AE24/starDuck/CAPE/models/eegLatentmodel_Linfonce.pth")
    dist.barrier()

if __name__ == "__main__":
    main()

    #srun -n 1 -p node03 --gres=gpu:2 --pty python -m torch.distributed.launch --nproc_per_node=2 train_eeg.py