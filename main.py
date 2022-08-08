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
from src.Once_dataset import Once_dataset
from src.count_value import count
from src.model_parameter import initialize_model


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes_p = ('morning', 'noon', 'afternoon','night')
    classes_w = ('sunny', 'cloudy', 'rainy')
    transform = transforms.Compose([transforms.ToPILImage(),#transform will not change main
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
    train_dataset = Once_dataset(path = './data',phase = 'train',transform=transform)
    val_dataset = Once_dataset(path = './data',phase = 'val',transform=transform)
    test_dataset = Once_dataset(path = './data',phase = 'test',transform=transform)


    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    train_img_path_list,val_img_path_list,test_img_path_list = train_dataset.return_list()
    dict_json_w,dict_json_p = train_dataset.return_json()
    train_dict_p,train_dict_w = count(train_img_path_list,dict_json_w,dict_json_p)
    train_class_p = {'morning': train_dict_p[0], 'noon': train_dict_p[1], 'afternoon': train_dict_p[2],'night':train_dict_p[3]}
    train_class_w = {'sunny': train_dict_w[0], 'cloudy': train_dict_w[1], 'rainy': train_dict_w[2]}
    print('period class distribution',train_class_p)
    print('weather class distribution',train_class_w)

    model_name = "resnet"
    feature_extract = True       
    model_p, input_size = initialize_model(model_name, num_classes=4, feature_extract = True, use_pretrained=True)
    model_w, input_size = initialize_model(model_name, num_classes=3, feature_extract = True, use_pretrained=True)
    model_p = model_p.to(device)
    model_w = model_w.to(device)
    # print("Params to learn:")
    if feature_extract:
        params_to_update_p = []                            
        for name,param in model_p.named_parameters():   
            if param.requires_grad == True:              
                params_to_update_p.append(param)          
                # print("\t",name)
                
    else:                                               
        for name,param in model_p.named_parameters():
            if param.requires_grad == True:
                # print("\t",name)
                print("\t")

    if feature_extract:
        params_to_update_w = []                            
        for name,param in model_w.named_parameters():   
            if param.requires_grad == True:              
                params_to_update_w.append(param)         
                # print("\t",name)
    else:                                             
        for name,param in model_w.named_parameters():
            if param.requires_grad == True:
                # print("\t",name)
                print("\t")
    parameters = params_to_update_w + params_to_update_p
    parameters = set(parameters)
    class_sample_counts_w = [train_dict_w[0], train_dict_w[1], train_dict_w[2]]
    weights_w = 1. / torch.tensor(class_sample_counts_w, dtype=torch.float)
    weights_w = weights_w.to(device)
    class_sample_counts_p = [train_dict_p[0], train_dict_p[1], train_dict_p[2], train_dict_p[3]]
    weights_p = 1. / torch.tensor(class_sample_counts_p, dtype=torch.float)
    weights_p = weights_p.to(device)
    optimizer_ft = optim.SGD(parameters, lr=0.001, momentum=0.9) 
    criterion_p = nn.CrossEntropyLoss(weight=weights_p)  
    criterion_w = nn.CrossEntropyLoss(weight =weights_w)


    val_acc_history_p = []  
    val_acc_history_w = []         # remember val acc of each AL round
    val_loss_history = []
    train_acc_history_p = []  
    train_acc_history_w = []         # remember val acc of each AL round
    train_loss_history = []
    test_acc_history_p = []  
    test_acc_history_w = [] 
    test_loss_history = []
    since = time.time()
    limit = 15
    for it in range(limit): # after get train set learn by cnn times by times
        print("-"*10)
        print('Epoch',it)
########################################################################################################Train       
        running_loss = 0.
        running_corrects_p = 0.
        running_corrects_w = 0.
        model_p.train()
        model_w.train()
        for inputs, label_period, label_weather in train_loader:  
            inputs = inputs.to(device)               #inputs.shape = torch.Size([32, 3, 224, 224])
            label_period = label_period.to(device) 
            label_weather = label_weather.to(device) 
            loss = 0         #labels.shape = torch.Size([32])

            with torch.autograd.set_grad_enabled(True):    #torch.autograd.set_grad_enabled
                outputs_p = model_p(inputs)
                outputs_w = model_w(inputs)              #outputs.shape = torch.Size([32, 10])
                loss += criterion_p(outputs_p, label_period)
                loss += criterion_w(outputs_w, label_weather)

            _, preds_p = torch.max(outputs_p, 1)
            _, preds_w = torch.max(outputs_w, 1)       

    
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * inputs.size(0)                                 
            running_corrects_p += torch.sum(preds_p.view(-1) == label_period.view(-1)).item()      # count accuracy
            running_corrects_w += torch.sum(preds_w.view(-1) == label_weather.view(-1)).item()      # count accuracy
        # best_model_wts = copy.deepcopy(model.state_dict())   
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_p = running_corrects_p / len(train_loader.dataset)
        epoch_acc_w = running_corrects_w / len(train_loader.dataset)
        print("train_Loss: {} Period_Acc: {} Weather_Acc: {}".format(epoch_loss, epoch_acc_p,epoch_acc_w))
        train_acc_history_p.append(epoch_acc_p)
        train_acc_history_w.append(epoch_acc_w)
        train_loss_history.append(epoch_loss)

########################################################################################################Val  
        running_loss = 0.
        running_corrects_p = 0.
        running_corrects_w = 0.
        model_p.eval()
        model_w.eval()
        for inputs, label_period, label_weather in val_loader:   
            inputs = inputs.to(device)               #inputs.shape = torch.Size([32, 3, 224, 224])
            label_period = label_period.to(device) 
            label_weather = label_weather.to(device) 
            loss = 0         #labels.shape = torch.Size([32])

            with torch.autograd.set_grad_enabled(False):    
                outputs_p = model_p(inputs)
                outputs_w = model_w(inputs)              #outputs.shape = torch.Size([32, 10])
                loss += criterion_p(outputs_p, label_period) #use normal criterion or weighted criterion?
                loss += criterion_w(outputs_w, label_weather)

            _, preds_p = torch.max(outputs_p, 1)
            _, preds_w = torch.max(outputs_w, 1)        

            running_loss += loss.item() * inputs.size(0)                                 
            running_corrects_p += torch.sum(preds_p.view(-1) == label_period.view(-1)).item()      # count accuracy
            running_corrects_w += torch.sum(preds_w.view(-1) == label_weather.view(-1)).item()      # count accuracy
        # best_model_wts = copy.deepcopy(model.state_dict())    
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc_p = running_corrects_p / len(val_loader.dataset)
        epoch_acc_w = running_corrects_w / len(val_loader.dataset)



        print("val_Loss: {} Period_Acc: {} Weather_Acc: {}".format(epoch_loss, epoch_acc_p,epoch_acc_w))
        val_acc_history_p.append(epoch_acc_p)
        val_acc_history_w.append(epoch_acc_w)
        val_loss_history.append(epoch_loss)
    
    time_elapsed = time.time() - since   #count train time
    print("Training Epoch compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60)) # _ m _s
    print("-"*10)

########################################################################################################Test 
    class_correct_p = list(0. for i in range(4))
    class_correct_w = list(0. for i in range(3))
    class_total_p = list(0. for i in range(4))
    class_total_w = list(0. for i in range(3))
    running_loss = 0.
    running_corrects_p = 0.
    running_corrects_w = 0.
    model_p.eval()
    model_w.eval()
    for inputs, label_period, label_weather in test_loader:   
        inputs = inputs.to(device)               #inputs.shape = torch.Size([32, 3, 224, 224])
        label_period = label_period.to(device) 
        label_weather = label_weather.to(device) 
        loss = 0         #labels.shape = torch.Size([32])

        with torch.autograd.set_grad_enabled(False):  
            outputs_p = model_p(inputs)
            outputs_w = model_w(inputs)              #outputs.shape = torch.Size([32, 10])
            loss += criterion_p(outputs_p, label_period)
            loss += criterion_w(outputs_w, label_weather)

        _, preds_p = torch.max(outputs_p, 1)
        _, preds_w = torch.max(outputs_w, 1)        

        running_loss += loss.item() * inputs.size(0)                                 
        running_corrects_p += torch.sum(preds_p.view(-1) == label_period.view(-1)).item()      # count accuracy
        running_corrects_w += torch.sum(preds_w.view(-1) == label_weather.view(-1)).item()      # count accuracy
        c_p = (preds_p == label_period).squeeze()   # get accuracy for each classes learned from pytorch tutorial
        for i in range(len(label_period)):
            label_p = label_period[i]
            class_correct_p[label_p] += c_p[i].item()
            class_total_p[label_p] += 1
        c_w = (preds_w == label_weather).squeeze()   # get accuracy for each classes learned from pytorch tutorial
        for i in range(len(label_weather)):
            label_w = label_weather[i]
            class_correct_w[label_w] += c_w[i].item()
            class_total_w[label_w] += 1
    # best_model_wts = copy.deepcopy(model.state_dict())    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc_p = running_corrects_p / len(test_loader.dataset)
    epoch_acc_w = running_corrects_w / len(test_loader.dataset)

    for i in range(4):
        print('Accuracy of %5s : %2d %%' %(classes_p[i],100*class_correct_p[i]/class_total_p[i]))
    print("-"*10)
    for i in range(3):
        print('Accuracy of %5s : %2d %%' %(classes_w[i],100*class_correct_w[i]/class_total_w[i]))

    print("test_Loss: {} Period_Acc: {} Weather_Acc: {}".format(epoch_loss, epoch_acc_p,epoch_acc_w))
    test_acc_history_p.append(epoch_acc_p)
    test_acc_history_w.append(epoch_acc_w)
    test_loss_history.append(epoch_loss)

    fig = plt.figure(1)
    plt.title("Validation Accuracy weather vs. Train Accuracy weather")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,len(train_acc_history_w)+1),train_acc_history_w,label="Train_weather")
    plt.plot(range(1,len(val_acc_history_w)+1),val_acc_history_w,label="Val_weather")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(train_acc_history_w)+1, 1.0))
    plt.legend()
    pic_acc_name = './Results/Weather_acc.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')

    fig = plt.figure(2)
    plt.title("Validation Accuracy period vs. Train Accuracy period")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,len(train_acc_history_p)+1),train_acc_history_p,label="Train_period")
    plt.plot(range(1,len(val_acc_history_p)+1),val_acc_history_p,label="Val_period")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(train_acc_history_w)+1, 1.0))
    plt.legend()
    pic_acc_name = './Results/Period_acc.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')


    fig = plt.figure(3)
    plt.title("Validation Loss vs. Train LOss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,len(train_loss_history)+1),train_loss_history,label="Train_Loss")
    plt.plot(range(1,len(val_loss_history)+1),val_loss_history,label="Val_Loss")
    plt.ylim((0,3.))
    plt.xticks(np.arange(1, len(val_loss_history)+1, 1.0))
    plt.legend()
    pic_acc_name = './Results/Loss.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')

if __name__ == '__main__':
    main()
