import time
import csv
from datetime import datetime
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import torch
from torch.utils.data import DataLoader


import model as md
from config import *
from utils import *

class CSVLogger:
    def __init__(self, filename, fieldnames=['Date', 'ModelName', 'Epoch', 'AvgLoss']):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exists = os.path.isfile(self.filename)
    
    def log(self, row=None,**kwargs):
        with open(self.filename, 'a', newline='') as csvfile:
            if not row:
                dictwriter = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if not self.file_exists:
                    dictwriter.writeheader()  # File doesn't exist yet, write a header
                    self.file_exists = True
                dictwriter.writerow(kwargs)
            else:
                writer = csv.writer(csvfile)
                if not self.file_exists:
                    writer.writerow(self.fieldnames)
                    self.file_exists = True

                writer.writerow(row)


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    else:
        return None

valid_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False,num_workers=num_workers)
print(f"Validation Batch Size:{batch_size}, number of batches:{len(valid_dataloader)}")


class NewModelHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(event)
        if event.is_directory:
            return None

        # Check for new model file
        if event.src_path.endswith('.pth'):
            print(f'New model file detected: {event.src_path}')
            time.sleep(10)
            # Here you can call a function to handle the validation
            validate_model(event.src_path)

logger = CSVLogger(f"/mnt/nvme/node02/pranav/AE24/starDuck/Transformer/model_ckpts/{modelName}_val.csv")
def validate_model(model_path):
 # Validation phase
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = md.model1
    ep=load_checkpoint(model_path,model)
    model.to(device)
    model.eval()
    total_val_loss = 0
    total_val_metric = 0
    print("validations")
    with torch.no_grad():
        for batch_idx,(eeg, mel) in enumerate(tqdm(valid_dataloader)):
            eeg = eeg.to(device).transpose(0,1)
            mel = mel.to(device).transpose(0,1)
            output = predict(model,eeg,device)
            loss = pearson_loss(output, mel,axis=0)
            metric = -loss
            total_val_loss += loss.item()
            total_val_metric += metric.item()

    num_batches_processed_val =  len(valid_dataloader)
    avg_val_loss = total_val_loss / num_batches_processed_val
    avg_val_metric = total_val_metric / num_batches_processed_val

    modelnames = os.path.basename(model_path).split("_")
    epoch=modelnames[-4]
    num_epochs =modelnames[-2]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [current_time, modelName, f"{epoch }/{num_epochs}", avg_val_loss]
    logger.log(row)
    print(f"Epoch [{epoch }/{num_epochs}] Average Validation Metric: {avg_val_metric:.4f}")




if __name__ == "__main__":
    path = '/mnt/nvme/node02/pranav/AE24/starDuck/Transformer/model_ckpts' 
    event_handler = NewModelHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    
    print(f'Starting observer on {path}')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
