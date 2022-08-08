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
import random
from src.statics import return_jsondict #when import this function from main, path is on main


class Once_dataset(Dataset):  # dataset

    def __init__(self, path,phase = 'train',transform = None):
        super(Once_dataset, self).__init__()
        self.phase = phase
        self.transform = transform
        self.path = path
        dict_json_w,dict_json_p = return_jsondict(path)
        self.dict_json_w = dict_json_w
        self.dict_json_p = dict_json_p
        img_path_list = self.find_img()
        num = 1000
        random.seed(10)
        torch.manual_seed(0)
        img_path_list = sample(img_path_list,num)
        test_split = 0.2
        test_size = int(test_split * len(img_path_list))
        # print(test_size)
        train_size = len(img_path_list) - test_size
        train_img_path_list, test_img_path_list = random_split(img_path_list,[train_size, test_size])

        val_split = 0.25
        val_size = int(val_split * len(train_img_path_list))
        # print(test_size)
        train_size = len(train_img_path_list) - val_size
        train_img_path_list, val_img_path_list = random_split(train_img_path_list,[train_size, val_size])
        self.train_img_path_list = train_img_path_list
        self.val_img_path_list = val_img_path_list
        self.test_img_path_list = test_img_path_list

    def return_list(self):
        return self.train_img_path_list,self.val_img_path_list,self.test_img_path_list

    def return_json(self):  
        return self.dict_json_w,self.dict_json_p

    def is_image_file(self,filename): #tell img format used with find_img#Dataset
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def find_img(self):#Dataset
        g = os.walk(self.path)  
        img_path_list = []
        for path,dir_list,file_list in g:   
            for file_name in file_list:  
                if self.is_image_file(file_name):
                    img_path_list.append(os.path.join(path, file_name))
        return img_path_list


    def __getitem__(self, index): # set for following dataloader
  
        torch.manual_seed(0)
        
        if self.phase == 'train':
            img_path = self.train_img_path_list[index]
            img_folder = os.path.split(img_path)[0]
            img_above_folder = os.path.split(img_folder)[0]
            img_above_folder_name = os.path.split(img_above_folder)[1]
            label_weather = self.dict_json_w[img_above_folder_name]
            label_period = self.dict_json_p[img_above_folder_name]
            img = plt.imread(img_path)

            if self.transform is not None:
                img = self.transform(img)

            return img, label_period,label_weather
        elif self.phase == 'val':
            img_path = self.val_img_path_list[index]
            img_folder = os.path.split(img_path)[0]
            img_above_folder = os.path.split(img_folder)[0]
            img_above_folder_name = os.path.split(img_above_folder)[1]
            label_weather = self.dict_json_w[img_above_folder_name]
            label_period = self.dict_json_p[img_above_folder_name]
            img = plt.imread(img_path)

            if self.transform is not None:
                img = self.transform(img)

            return img, label_period,label_weather
        elif self.phase == 'test':
            img_path = self.test_img_path_list[index]
            img_folder = os.path.split(img_path)[0]
            img_above_folder = os.path.split(img_folder)[0]
            img_above_folder_name = os.path.split(img_above_folder)[1]
            label_weather = self.dict_json_w[img_above_folder_name]
            label_period = self.dict_json_p[img_above_folder_name]
            img = plt.imread(img_path)

            if self.transform is not None:
                img = self.transform(img)

            return img, label_period,label_weather
    def __len__(self):
        if self.phase == 'train':
            return len(self.train_img_path_list)
        elif self.phase == 'val':
            return len(self.val_img_path_list)
        elif self.phase == 'test':
            return len(self.test_img_path_list)
