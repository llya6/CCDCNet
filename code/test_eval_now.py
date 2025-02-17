import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,classification_report,precision_score,recall_score,cohen_kappa_score
import seaborn as sns
import pandas as pd
from model import *
from loss import *
from evaluate import *
from util import *
from dataloaded import *
from CCDCNet import *



model=CCDC_m(num_classes=1)
if loadstate:
    model.load_state_dict(torch.load(loadstateptfile))

model = model.to(DEVICE)

loss_fn = DiceBCELoss()


# Load model weights
model.load_state_dict(torch.load(outptfile))

def compute_miou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(np.float32)
        label_cls = (labels == cls).astype(np.float32)
        
        intersection = np.sum(pred_cls * label_cls)
        union = np.sum(pred_cls + label_cls) - intersection
        
        if union == 0:
            iou = float('nan') 
        else:
            iou = intersection / union
        
        ious.append(iou)
    return np.nanmean(ious) 



def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_binary = (y_pred > 0.5).float()
            all_preds.append(y_pred_binary.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        epoch_loss = epoch_loss / len(loader)

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    print(classification_report(all_labels, all_preds))
    
    miou = compute_miou(all_preds, all_labels, num_classes)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary') 
    recall = recall_score(all_labels, all_preds, average='binary') 
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    
    sample_num=np.random.randint(0,BATCH_SIZE)
    image,mask=next(iter(loader))
    image=image[sample_num]
    mask=mask[sample_num]
    logits_mask=model(image.to('cuda', dtype=torch.float32).unsqueeze(0))
    pred_mask=torch.sigmoid(logits_mask)
    pred_mask=(pred_mask > ratio)*1.0
    
    
    img=linear_stretch(image.numpy()[[2,1,0]], 2)
    img=np.uint8(img*255)
    # img=np.uint8(linear_stretch(img,2))
    
    # plt.imshow(image[-1]*255,'jet')
    # plt.show()
    
    f, axarr = plt.subplots(1,3) 
    axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray',vmin=0, vmax=1)
    axarr[0].imshow(np.transpose(img, (1,2,0)))
    # if 
    axarr[2].imshow(pred_mask.detach().cpu().squeeze(0)[0].numpy(), cmap='gray',vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(basedir+'/'+'test.jpg',pad_inches=0.01, bbox_inches='tight')
    plt.close()
    return epoch_loss, acc, f1, miou, recall, precision, kappa, conf_matrix

def save_results_to_csv(acc, f1, miou, recall, precision, kappa, conf_matrix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'mIoU', 'Recall', 'Precision', 'Kappa'],
        'Value': [acc, f1, miou, recall, precision, kappa]
    })
    
    results_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(conf_matrix.shape[0])],
                                  columns=[f'Class {i}' for i in range(conf_matrix.shape[1])])
    
    conf_matrix_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_matrix_df.to_csv(conf_matrix_csv_path)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print(f'Evaluation results saved to {results_csv_path}')
    print(f'Confusion matrix saved to {conf_matrix_csv_path}')

num_classes = 2  

output_dir = os.path.join(basedir, 'evaluation_results1')
valid_loss, acc, f1, miou, recall, precision, kappa, conf_matrix = evaluate(model,valid_loader, loss_fn, DEVICE, num_classes)

print(f"valid_loss: {valid_loss}")  
print(f"acc: {acc}")  
print(f"f1: {f1}")  
print(f"recall: {recall}")  
print(f"precision: {precision}")  
print(f"kappa: {kappa}")  
print(f"miou: {miou}")  

print("conf_matrix:")  
print(conf_matrix)

save_results_to_csv(acc, f1, miou, recall, precision, kappa, conf_matrix, output_dir)

