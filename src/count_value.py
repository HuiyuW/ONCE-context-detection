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


def count(img_path_list,dict_json_w,dict_json_p): #tell img class distribution for reweighting #src
    label_weather_list = []
    label_period_list = []
    for i in range(len(img_path_list)):
        img_path = img_path_list[i]
        img_folder = os.path.split(img_path)[0]
        img_above_folder = os.path.split(img_folder)[0]
        img_above_folder_name = os.path.split(img_above_folder)[1]

        label_weather = dict_json_w[img_above_folder_name]
        label_period = dict_json_p[img_above_folder_name]


        label_weather_list.append(int(label_weather))
        label_period_list.append(int(label_period))
    dict_p = {}
    for key in label_period_list:
        dict_p[key] = dict_p.get(key, 0) + 1 #count num in each classes
    dict_w = {}
    for key in label_weather_list:
        dict_w[key] = dict_w.get(key, 0) + 1
    # print(dict)
    return dict_p,dict_w