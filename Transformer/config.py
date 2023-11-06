
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from dataset import meleeg_dataset
from model import *

num_workers=3
batch_size =768
modelName = "model1"


train_dataset = meleeg_dataset(file_types=["train"])
val_dataset = meleeg_dataset(file_types=["val"])