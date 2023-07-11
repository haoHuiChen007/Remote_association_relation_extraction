import os.path
import time
import torch
import numpy as np
from importlib import import_module
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Bruce-Bert-Text-Classification")
parser.add_argument("--model", type=str, default="BruceBertCrf",
                    help="choose a model BruceBertCrf BruceBertAttention")
args = parser.parse_args()
if __name__ == '__main__':
    print_hi('PyCharm')

