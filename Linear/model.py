
import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, integration_window=32, nb_filters=1, nb_channels=64):
        super(SimpleLinearModel, self).__init__()
        # Define the 1D convolutional layer
        self.conv1d = nn.Conv1d(nb_channels, nb_filters, integration_window)

    def forward(self, x):
        # Forward pass through the convolutional layer
        out = self.conv1d(x)
        return out
