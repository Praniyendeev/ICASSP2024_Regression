
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Transformer
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import meleeg_dataset

import math
import time
import os
import numpy as np


class EEGEncoder(nn.Module):
    def __init__(self, input_channels, feature_size, num_heads, num_layers, dropout=0.1):
        super(EEGEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = src.permute(0, 2, 1)  # (batch, channels, time)
        src = F.relu(self.conv1(src))
        src = F.relu(self.conv2(src))
        src = src.permute(2, 0, 1)  # (time, batch, feature) for transformer
        memory = self.transformer_encoder(src)
        return memory

class MelDecoder(nn.Module):
    def __init__(self, feature_size, mel_channels, num_heads, num_layers, dropout=0.1):
        super(MelDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(feature_size, mel_channels)

    def forward(self, tgt, memory):
        tgt = self.transformer_decoder(tgt, memory)
        output = self.out(tgt)
        return output


class EEG2Mel(nn.Module):
    def __init__(self, input_channels, feature_size, num_heads, num_layers, dropout, mel_channels):
        super(EEG2Mel, self).__init__()
        self.encoder = EEGEncoder(input_channels, feature_size, num_heads, num_layers, dropout)
        self.decoder = MelDecoder(feature_size, mel_channels, num_heads, num_layers, dropout)

    def forward(self, src):
        memory = self.encoder(src)
        tgt = torch.zeros_like(memory)  # Placeholder for tgt, replace with actual target during training
        output = self.decoder(tgt, memory)
        return output











class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class eeg2melTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_feature_size: int,
                 tgt_feature_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(eeg2melTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.src_fc = nn.Linear(src_feature_size, emb_size)
        self.tgt_fc = nn.Linear(tgt_feature_size, emb_size)
        self.generator = nn.Linear(emb_size, tgt_feature_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_fc(src))
        tgt_emb = self.positional_encoding(self.tgt_fc(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_fc(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_fc(tgt)), memory, tgt_mask)




model1= eeg2melTransformer(
    num_encoder_layers=3,
    num_decoder_layers=3,
    emb_size=128,
    nhead=8,
    src_feature_size=64,
    tgt_feature_size=10
)




modelc = eeg2melTransformer(
    num_encoder_layers=3,
    num_decoder_layers=3,
    emb_size=128,
    nhead=8,
    src_feature_size=64,
    tgt_feature_size=10
)


