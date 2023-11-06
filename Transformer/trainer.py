import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Transformer
from torch import Tensor
from torch.utils.data import DataLoader

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import model
from utils import *
from config import *

import math
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)#no 

print(f"Train Batch Size:{batch_size}, number of batches:{len(train_dataloader)} with {num_workers} workers")

# Model
model =model.model1
model =model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# checkpoint_path = "model1.pth"
ckpts_path = "/mnt/nvme/node02/pranav/AE24/starDuck/Transformer/model_ckpts/"

# Attempt to load the checkpoint if it exists
loaded_epoch = load_checkpoint("/mnt/nvme/node02/pranav/AE24/starDuck/Transformer/" + modelName +".pth" , model, optimizer)

# If a checkpoint was loaded, use its epoch and hyperparameters
if loaded_epoch is not None:
    start_epoch = loaded_epoch + 1
    # Update hyperparameters if needed
else:
    start_epoch = 0
max_batches =len(train_dataloader)
train_start_time=time.time()
MAE=torch.nn.L1Loss(reduction='none')
print(f"Training Started. with {start_epoch}")
for epoch in range(num_epochs):
    start_time = time.time()
    torch.cuda.empty_cache()
    # Training phase
    model.train()
    total_train_loss = 0

    for batch_idx, (eeg, mel) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        eeg = eeg.to(device).transpose(0,1)
        mel = mel.to(device).transpose(0,1)
        mel_input = mel[:-1]

        eeg_mask, mel_mask, eeg_padding_mask, mel_padding_mask = create_mask(eeg, mel_input,device)
        output = model(eeg, mel_input, eeg_mask, mel_mask, eeg_padding_mask, mel_padding_mask, eeg_padding_mask)

        # Calculate loss and metric\
        pr_loss = -pearson_non_mean(output, mel[1:],axis=0)
        mae_loss = MAE( mel[1:],output)
        loss = torch.mean(pr_loss+mae_loss)
        metric = -torch.mean(pr_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
        if batch_idx >= max_batches - 1:
            break

    num_batches_processed = min(max_batches, len(train_dataloader))
    avg_train_loss = total_train_loss / num_batches_processed
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - start_time:.2f} seconds, "
        f"Average Training Loss: {avg_train_loss:.4f},")
    if (epoch)%3==0:
        save_checkpoint(epoch, model, optimizer, ckpts_path + modelName +f"_{epoch+1}_-_{num_epochs}_.pth")
    save_checkpoint(epoch, model, optimizer, "/mnt/nvme/node02/pranav/AE24/starDuck/Transformer/" + modelName +".pth")

save_checkpoint(epoch, model, optimizer, ckpts_path + modelName +f"_{num_epochs}_-_{num_epochs}_.pth")

print(f"Training completed in {time.time() - train_start_time:.2f} seconds ")



test_files=[k  for k in os.listdir("/mnt/nvme/node02/pranav/AE24/data/split_data/") if "test" in k]
subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))

datasets_test = {}
# Create a generator for each subject
for sub in subjects:
    test_sub_datset = meleeg_dataset(file_types=["test",sub])
    datasets_test[sub] = DataLoader(test_sub_datset, batch_size=batch_size, shuffle=False,num_workers=num_workers)



evaluation,mea = evaluate_model(model, datasets_test,pearson_loss,device)
test_files=[k  for k in os.listdir("/mnt/nvme/node02/pranav/AE24/data/split_data/") if "test" in k]
subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))

datasets_test = {}
for sub in subjects:
    test_sub_datset = meleeg_dataset(file_types=["test",sub])
    datasets_test[sub] = DataLoader(test_sub_datset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

evaluation,mea = evaluate_model(model, datasets_test,pearson_loss,device)



    # # Validation phase
    # model.eval()
    # total_val_loss = 0
    # total_val_metric = 0
    # print("validations")
    # with torch.no_grad():
    #     for batch_idx,(eeg, mel) in enumerate(tqdm(valid_dataloader)):
    #         eeg = eeg.to(device).transpose(0,1)
    #         mel = mel.to(device).transpose(0,1)

    #         output = predict(model,eeg,device)

    #         loss = pearson_loss(output, mel,axis=0)
    #         metric = -loss
    #         total_val_loss += loss.item()
    #         total_val_metric += metric.item()

    #         # if batch_idx >= max_batches - 1:
    #         #     break

    # num_batches_processed_val =  len(valid_dataloader)
    # avg_val_loss = total_val_loss / num_batches_processed_val
    # avg_val_metric = total_val_metric / num_batches_processed_val

    # print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - start_time:.2f} seconds, "
    #        f"Average Training Loss: {avg_train_loss:.4f},")# Average Validation Loss: {avg_val_loss:.4f}, "
        #   f"Average Validation Metric: {avg_val_metric:.4f}")





