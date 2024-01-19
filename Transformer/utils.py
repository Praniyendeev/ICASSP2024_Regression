import torch
import numpy as np
import os 
import time
from tqdm import tqdm

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz,device=device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt,device=device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src[:,:,0] == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt[:,:,0] == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def predict(model, src, device=device):
    model.eval()  
    seq_len = src.shape[0]
    src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)


    ys = torch.zeros(1, src.shape[1], 10).to(device)  
    output_sequence = []


    for i in range(seq_len):
        memory = memory.to(device)
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(device).type(torch.bool)

        out = model.decode(ys, memory, tgt_mask)
        

        next_val = model.generator(out)

        output_sequence.append(next_val)

        ys = torch.cat([ys, next_val[-1:]], dim=0)

    return ys[1:]

def pearson_non_mean(y_true, y_pred, axis=1):
    """
    Pearson correlation function implemented in PyTorch.
    """
    y_true_mean = y_true.mean(dim=axis, keepdim=True)
    y_pred_mean = y_pred.mean(dim=axis, keepdim=True)

    numerator = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).sum(dim=axis, keepdim=True)
    std_true = ((y_true - y_true_mean) ** 2).sum(dim=axis, keepdim=True)
    std_pred = ((y_pred - y_pred_mean) ** 2).sum(dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred +1e-8)

    c = torch.zeros_like(numerator)
    # zero mask
    mask = (denominator != 0)

    # finally perform division
    c[mask] = numerator[mask] / denominator[mask]
    # PyTorch equivalent
    return  c

def pearson_correlation(y_true, y_pred, axis=1):
    """
    Pearson correlation function implemented in PyTorch.
    """
    y_true_mean = y_true.mean(dim=axis, keepdim=True)
    y_pred_mean = y_pred.mean(dim=axis, keepdim=True)

    numerator = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).sum(dim=axis, keepdim=False)
    std_true = ((y_true - y_true_mean) ** 2).sum(dim=axis, keepdim=False)
    std_pred = ((y_pred - y_pred_mean) ** 2).sum(dim=axis, keepdim=False)
    denominator = torch.sqrt(std_true * std_pred + 1e-8)

    c = torch.zeros_like(numerator)
    # zero mask
    mask = (denominator != 0)

    # finally perform division
    c[mask] = numerator[mask] / denominator[mask]
    # PyTorch equivalent
    return  torch.mean(c)

def pearson_loss(y_true, y_pred, axis=1):
    """
    Pearson loss function in PyTorch, modified to return a scalar.
    """


    return -pearson_correlation(y_true, y_pred, axis=axis)

def pearson_metric(y_true, y_pred, axis=1):
    """
    Pearson metric function in PyTorch.
    """
    corr = pearson_correlation(y_true, y_pred, axis=axis)
    return corr.mean(dim=-1)

def evaluate_model(model, test_loaders, criterion, device):
    """Evaluate a model in PyTorch.

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model to evaluate.
    test_loaders: dict
        Mapping between a subject and a DataLoader containing the test set for the subject.
    criterion: loss function
        The loss function used for evaluation.
    device: torch.device
        Device to run the model on.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set.
    """

# pearson acroos time
# mean across bands/features
# mean across samples per subjects
# mean across subjects 
    start_time=time.time()
    model.eval()
    evaluation = {}
    pmean=[]
    for subi,(subject, loader) in enumerate(tqdm(test_loaders.items())):
        total_loss = 0
        all_labels = []
        all_predictions = []
        i=0
        with torch.no_grad():
            for eeg, mel_true in loader:
                eeg, mel_true = eeg.to(device).transpose(0,1), mel_true.to(device).transpose(0,1)
                mel_input = mel_true[:-1]


                output = predict(model,eeg,device)
                
                loss = criterion(output, mel_true,axis=0)
                total_loss += loss.item()
                all_labels.append(mel_true)
                all_predictions.append(output)
                i+=1

                
        # Concatenate all batches
        all_labels = torch.cat(all_labels, dim=1)
        all_predictions = torch.cat(all_predictions, dim=1)

        k=np.squeeze(pearson_non_mean(all_predictions,all_labels,axis=0))

        # Calculate Pearson correlation
        band_correlations = k.mean(dim=0)
        pmean.append(k.to("cpu").mean() )
        avg_loss = total_loss / len(loader)
        evaluation[subject] = {
            "loss": avg_loss,
            "pearson_correlation_per_band": band_correlations,
            "pMetric":pmean[-1]
        }
    print(f"completed evaluation in {time.time() - start_time:.2f} seconds")
    print(f"evaluation score: {np.mean(pmean)}")
    return evaluation, np.mean(pmean) 


def save_checkpoint(epoch, model, optimizer, path):
    print(f"saving chekpoint to {path}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer,device='cpu'):
    if os.path.isfile(path):
        checkpoint = torch.load(path,map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    else:
        return None
