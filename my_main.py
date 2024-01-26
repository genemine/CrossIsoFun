import sys
import my_utils
from train_test import *
import numpy as np
import torch
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)



data_folder= './data_demo/'

train_test(data_folder)