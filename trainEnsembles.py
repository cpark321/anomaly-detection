from bayes_opt import BayesianOptimization

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils import MVTecDataset, MVTecActiveDataset, evaluate_accuracy, getFileList
from models import MVTecCNN_BO

import time
import argparse
import copy

from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--target', required=True, help='target class')
parser.add_argument('-k', '--no_ensemble', default= 3, type=int, help='number of esemble models for active learning')
parser.add_argument('--unlabel_ratio', default= 0.5, type=float, help='the ratio of unlabeled data')
parser.add_argument('--sample_ratio', default= 0.1, type=float, help='the ratio of sampling from unlabeled pool')


parser.add_argument('-c', '--no_cuda', required=False, default=None, help='which cuda')
parser.add_argument('--lr', default= 0.001, type=float , required=False, help='learning rate')
parser.add_argument('--no_epoch', default= 30, type= int, required=False, help='number of epochs')

args = parser.parse_args()


target_class = args.target
num_ensemble = args.no_ensemble
sample_ratio =  args.sample_ratio
unlabeled_ratio =  args.unlabel_ratio

save_path = os.path.join('./saves_active/', target_class)

if not os.path.exists(save_path):
    os.makedirs(save_path)

activeBatchSize = 16
best_accuracies = np.zeros(num_ensemble)
best_models={}


if not os.path.exists(save_path):
    os.makedirs(save_path)

device_type='cuda'

if args.no_cuda is not None:
    device_type = 'cuda:'+str(args.no_cuda)
    
device = torch.device(device_type if torch.cuda.is_available() else 'cpu')



def loadActiveDataset(target_class):
    normal_list_dir = [os.path.join('./data/', target_class, 'train', 'good'), os.path.join('./data/', target_class, 'test', 'good')]

    test_dir = os.path.join('./data/', target_class, 'test')
    test_subfolders = next(os.walk(test_dir))[1]

    abnormal_list_dir=[]
    
    for item in test_subfolders:
        if item != 'good':
            abnormal_list_dir.append(os.path.join('./data/', target_class, 'test', item))
    
    normal_file_list, abnormal_file_list = getFileList(normal_list_dir, abnormal_list_dir)
    normal_rand_idx = np.random.permutation(len(normal_file_list))
    abnormal_rand_idx = np.random.permutation(len(abnormal_file_list))
    
    unlabeled_normal_num = int(unlabeled_ratio*len(normal_file_list))
    unlabeled_abnormal_num = int(unlabeled_ratio*len(abnormal_file_list))
    
    unlabeled_normal_data_list = normal_file_list[normal_rand_idx[:unlabeled_normal_num]]
    unlabeled_abnormal_data_list = abnormal_file_list[abnormal_rand_idx[:unlabeled_abnormal_num]]
    
    labeled_normal_data_list = normal_file_list[normal_rand_idx[unlabeled_normal_num:]]
    labeled_abnormal_data_list = abnormal_file_list[abnormal_rand_idx[unlabeled_abnormal_num:]]
    
    return labeled_normal_data_list, labeled_abnormal_data_list, unlabeled_normal_data_list, unlabeled_abnormal_data_list


labeled_normal_data_list, labeled_abnormal_data_list, unlabeled_normal_data_list, unlabeled_abnormal_data_list = loadActiveDataset(target_class)


unlabeled_dataset = MVTecActiveDataset(unlabeled_normal_data_list, unlabeled_abnormal_data_list, isUnlabeled=True)
labeled_dataset = MVTecActiveDataset(labeled_normal_data_list, labeled_abnormal_data_list, isUnlabeled=False)

val_num = int(len(labeled_dataset)*0.15)
test_num = int(len(labeled_dataset)*0.375)
train_num = len(labeled_dataset)  - val_num - test_num

train_dataset, valid_dataset, test_dataset =random_split(labeled_dataset,[train_num, val_num, test_num])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

active_loader = DataLoader(unlabeled_dataset, batch_size=activeBatchSize, shuffle=False)


data_loaders = {0:train_loader, 1:valid_loader, 2:test_loader, 3:active_loader}

savepath_dataloader = os.path.join(save_path, f"dataloader_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}.pth")
torch.save(data_loaders, savepath_dataloader)

# save ensembles 
def save_esemble_models(best_val_acc, net):
    if np.min(best_accuracies) < best_val_acc:
        idx = np.argmin(best_accuracies)
        best_models[idx] = net
        best_accuracies[idx] = best_val_acc    
        
        
def train(lr, num_channel):        
    net = MVTecCNN_BO(num_channel).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_val_acc = 0.    
    num_epoch = 25
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(num_epoch):
        loss_count=0
        loss_sum=0
        for idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device, dtype=torch.float)
            label = label.view(-1,1)
            pred = net(img)
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
            loss_count+=1
            if idx%10==0:
                val_acc = evaluate_accuracy(net, valid_loader, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc                    
                best_model = copy.deepcopy(net)
        scheduler.step()
        
    save_esemble_models(best_val_acc, best_model.eval())
    return best_val_acc

def cnn_function(lr, num_channel):
    num_channel = int(8 + num_channel*54)   # min 8, max 64
    best_val_accuracy = train(lr, num_channel)    
    return best_val_accuracy

# Bounded region of parameter space
pbounds = {'lr': (1e-3, 0.1), 'num_channel':(0, 1)}

optimizer = BayesianOptimization(
    f=cnn_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=3,
    n_iter=40
)

savepath_ensembles = os.path.join(save_path, f"ensemble_models_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}.pth")

best_models_cpu={}
for i in best_models:
    best_models_cpu[i] = best_models[i].cpu()
    

torch.save(best_models_cpu, savepath_ensembles)

original_model = best_models[np.argmax(best_accuracies)].eval()
original_model_test_acc = evaluate_accuracy(original_model.cpu(), test_loader, 'cpu')

with open(os.path.join(save_path,'active_results.txt'), 'a') as f:
    f.write(f'original labeled dataset\n {optimizer.max}\n')
    f.write(f'original best valid accuracies -{best_accuracies}\n')
    f.write(f'original_model_test_acc: {original_model_test_acc}\n')

    
    
    
    



