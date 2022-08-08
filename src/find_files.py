import sys 
import os
import json
import pandas as pd
import os 
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import os
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import torchvision
from torch.utils.data import random_split
import time
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from random import sample


def lm_find_files(path, target, result): #find json under some path folder #Dataset

    files = os.listdir(path);
    for f in files:
        npath = path + '/' + f
        if(os.path.isfile(npath)):
            if(os.path.splitext(npath)[1] == target):
                result.append(npath)
        if(os.path.isdir(npath)):
            if (f[0] == '.'):
                pass
            else:
                lm_find_files(npath, target, result)
    return result