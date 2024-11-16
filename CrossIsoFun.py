import sys
import my_utils
from train_test import *
import numpy as np
import torch
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

import argparse
import os



device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)



train_test(train_data_folder, test_data_folder, train_label_folder, output_folder)