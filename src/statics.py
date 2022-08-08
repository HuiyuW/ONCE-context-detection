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
from src.find_files import lm_find_files


label_period_dict = {'morning': 0, 'noon': 1, 'afternoon': 2,'night':3}
label_weather_dict = {'sunny': 0, 'cloudy': 1, 'rainy': 2}

def return_jsondict(path):   #src
    result=[]
    result = lm_find_files(path,".json",result) #all json path
    label_weather_list = []
    label_period_list = []
    json_num_list = []
    for i in range(len(result)):

        with open(result[i],'r',encoding='utf-8') as f:
         jsonload = json.load(f)
        weather = jsonload['meta_info']['weather']
        period = jsonload['meta_info']['period']
        label_period = label_period_dict[str(period)]
        label_weather = label_weather_dict[str(weather)]
        label_weather_list.append(label_weather)
        label_period_list.append(label_period)
        json_folder = os.path.split(result[i])[0]
        json_above_folder = os.path.split(json_folder)[1]
        json_num_list.append(json_above_folder)

    dict_json_w = dict(zip(json_num_list,label_weather_list))
    dict_json_p = dict(zip(json_num_list,label_period_list)) #count label in dict is really fast 43 seconds update

    return dict_json_w,dict_json_p