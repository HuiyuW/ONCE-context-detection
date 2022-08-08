# Context_Detect

## Contents  
- [Environment](#Environment)  
- [Structure](#Structure)  
- [Dataset](#Dataset)   
- [Training](#Training)
- [Results](#Results) 


## Environment
* Linux version 4.15.0-175-generic
* Python 3.7.2  
* Pytorch1.0.0.dev20190219

## Structure

  ├── [main.py](./main.py): Training pipeline   
  ├── data: ONCE dataset  
  ├── [src](./src): Functions   
  ├────[count_value.py](./src/count_value.py): Count sample classes distribution according to path list    
  ├────[find_files.py](./src/find_files.py): Find all json files under path     
  ├────[model_parameter.py](./src/model_parameter.py): Set up model   
  ├────[Once_dataset.py](./src/Once_dataset.py): Set up Once dataset for train val and test   
  └────[statics.py](./src/statics.py): Retrun json dict with json title and label correlation   

## Dataset
* ONCE data downloaded from [ONCE download](https://once-for-auto-driving.github.io/download.html)

## Training
* Make sure to prepare the dataset in advance
* Modify the relative paths of datasets in the code.
* run main.py


## Results
![Figure_1](./results/Loss.png)
![Figure_2](./results/Period_acc.png)
![Figure_2](./results/Weather_acc.png)
