import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from model import *
from config import *
# from dataload import *
from util import *
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
torch.cuda.is_available()
import shutil
from dataloaded import *
from CCDCNet import *

model=CCDC_m(num_classes=1)
if loadstate:
    model.load_state_dict(torch.load(loadstateptfile))
model.to(DEVICE)

optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=0.0001)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
outloss={}

trainloss=outloss.setdefault('trainloss',[])
valloss=outloss.setdefault('valloss',[])


best_val_dice_loss=np.Inf
# best_val_bce_loss=np.Inf
best_val_loss=np.Inf
for i in range(EPOCHS):
    outfile=basedir+rf"jpgoutnew/{str(i)}.jpg"
    # train_loss=1
    # valid_loss=1
    os.makedirs(os.path.split(outfile)[0],exist_ok=True)
    train_loss = train_fn(train_loader,model,optimizer)
    valid_loss = eval_fn(valid_loader,model,outfile)
    scheduler.step()
    # train_dice,train_bce=train_loss
    # valid_dice,valid_bce=valid_loss
    # print(f'Epochs:{i+1}\nTrain_loss --> Dice: {train_dice} BCE: {train_bce} \nValid_loss --> Dice: {valid_dice} BCE: {valid_bce}')
    trainloss.append(train_loss)
    valloss.append(valid_loss)
    print(f'Epochs:{i+1}\nTrain_loss --> {train_loss} \nValid_loss --> { valid_loss:} ')

    if valid_loss< best_val_loss:
        torch.save(model.state_dict(),outptfile)
        print('Model Saved')
        # best_val_dice_loss=valid_dice
        best_val_loss= valid_loss
    if i%50==0:
        torch.save(model.state_dict(),outptfile.replace('.pt',f'_{str(i)}.pt'))

import pandas as pd
outcsv=pd.DataFrame(outloss)
outcsv.to_csv(outptfile.replace('.pt','.csv'))

