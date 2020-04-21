from bayes_opt import BayesianOptimization
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
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
# model_freeze = args.freeze

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


def augmentDataset(train_dataset, valid_dataset, unlabeled_dataset, idx):
#     active_data_list = np.array(unlabeled_dataset.data_list)[idx]
#     active_label_list = np.array(unlabeled_dataset.label_list)[idx]
    labeled_dataset = ConcatDataset([Subset(active_loader.dataset, idx), train_dataset, valid_dataset])
#     labeled_dataset.data_list = train_dataset.data_list + valid_dataset.data_list + active_data_list.tolist()
#     labeled_dataset.label_list = train_dataset.label_list + valid_dataset.label_list + active_label_list.tolist()

    val_num = int(len(labeled_dataset)*0.20)
    train_num = len(labeled_dataset)  - val_num

    new_train_dataset, new_valid_dataset =random_split(labeled_dataset,[train_num, val_num])

    new_train_loader = DataLoader(new_train_dataset, batch_size=8, shuffle=True, drop_last=True)
    new_valid_loader = DataLoader(new_valid_dataset, batch_size=8, shuffle=False)
    
    return new_train_loader, new_valid_loader


def activeTrain(idx, lr, isActiveLearn):
    
    if isActiveLearn:
        method='active'
        train_loader = active_train_loader
        valid_loader = active_valid_loader
    else:
        method='normal'
        train_loader = normal_train_loader
        valid_loader = normal_valid_loader
    
    net = copy.deepcopy(best_models[idx])
    net.to(device)
    net.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_val_acc = 0.
    
    num_epoch = 20
    
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
                net.eval()
                val_acc = evaluate_accuracy(net, valid_loader, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = copy.deepcopy(net)

                net.train()
        scheduler.step()   
#     save_esemble_models(best_val_acc, net.eval())
    return best_val_acc, best_model

def cnn_active_function(lr, model_index):    
    global global_best_active_acc
    if model_index==1:
        idx=num_ensemble-1
    else:
        idx = int(model_index*num_ensemble)      
    best_acc, best_model = activeTrain(idx, lr, isActiveLearn=True)
    
    if best_acc > global_best_active_acc:
        global_best_active_acc = best_acc
        torch.save(best_model.state_dict(), savepath_active)
    
    return best_acc

def cnn_normal_function(lr, model_index):
    global global_best_normal_acc
    if model_index==1:
        idx=num_ensemble-1
    else:
        idx = int(model_index*num_ensemble)      
    best_acc, best_model = activeTrain(idx, lr, isActiveLearn=False)
    
    if best_acc > global_best_normal_acc:
        global_best_normal_acc = best_acc
        torch.save(best_model.state_dict(), savepath_normal)
    return best_acc



savepath_dataloader = os.path.join(save_path, f"dataloader_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}_annotate-{sample_ratio}.pth")
savepath_ensembles = os.path.join(save_path, f"ensemble_models_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}_annotate-{sample_ratio}.pth")

# 데이터 로더 Load
data_loaders = torch.load(savepath_dataloader)

train_loader = data_loaders[0]
valid_loader =  data_loaders[1]
test_loader =  data_loaders[2]
active_loader =  data_loaders[3]

train_dataset = train_loader.dataset
valid_dataset = valid_loader.dataset
unlabeled_dataset = active_loader.dataset


# 앙상블 모델 Load
best_models = torch.load(savepath_ensembles)
best_model_num_channels=[]
best_accuracies =[]
for i in range(num_ensemble):
    best_model_num_channels.append(best_models[i].num_channel)    
    best_accuracies.append(evaluate_accuracy(best_models[i].eval(), valid_loader, device='cpu'))

original_model = best_models[np.argmax(best_accuracies)].eval()
original_model_test_acc = evaluate_accuracy(original_model.cpu(), test_loader, 'cpu')
    
    
    
# 모델 K개의 ensemble prediction을 통해 unlabeled dataset 각 sample의 엔트로피 계산
for idx, (img, label) in enumerate(active_loader):
    img = img.to('cpu')
    label = label.to('cpu', dtype=torch.float)
       
    for model_idx in best_models:
        best_models[model_idx].eval()
        label = label.view(-1,1)
        if model_idx==0:
            total_tensor = best_models[model_idx](img)
        else:
            total_tensor = torch.cat((total_tensor, best_models[model_idx](img)), dim=1)        
    
    p = torch.mean(total_tensor, dim=1)
    H = -p*torch.log(p)
    
    if idx==0:
        total_H = H
    else:
        total_H = torch.cat((total_H, H))
    
# 우리가 Annotation 요청할 샘플 수 및 요청 샘플 인덱스

query_num = int(len(unlabeled_dataset)*sample_ratio) 
active_vals, active_idx = torch.topk(total_H, query_num)


# Uncertainty 높은 Top K개와 Random K개 각각 추출
active_train_loader, active_valid_loader = augmentDataset(train_dataset, valid_dataset, unlabeled_dataset, active_idx.cpu())

rand_idx = np.random.permutation(len(unlabeled_dataset))[:query_num]
normal_train_loader, normal_valid_loader = augmentDataset(train_dataset, valid_dataset, unlabeled_dataset, rand_idx)

# Active learning 학습 진행 (기존 뽑힌 Top K개 모델만 재학습)
global_best_active_acc = 0
savepath_active = os.path.join(save_path, f"active_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}_annotate-{sample_ratio}.pth")

active_pbounds = {'lr': (1e-5, 0.001), 'model_index':(0, 1)}

optimizer = BayesianOptimization(
    f=cnn_active_function,
    pbounds=active_pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=30,   # 수정 30
)

# Active learning의 최종 Test accuracy 
model_idx = int(optimizer.max['params']['model_index']*num_ensemble)
if model_idx == num_ensemble:
    model_idx = num_ensemble-1

active_final_model = MVTecCNN_BO(best_model_num_channels[model_idx]).to(device)
active_final_model.load_state_dict(torch.load(savepath_active, map_location=device))

active_model_test_acc = evaluate_accuracy(active_final_model.eval(), test_loader, device)

# Random 추출도 일반적인 학습 진행
global_best_normal_acc = 0
savepath_normal = os.path.join(save_path, f"normal_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}_annotate-{sample_ratio}.pth")

optimizer = BayesianOptimization(
    f=cnn_normal_function,
    pbounds=active_pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=30,   # 수정 30
)

# Random 추출의 최종 Test accuracy
model_idx = int(optimizer.max['params']['model_index']*num_ensemble)
if model_idx == num_ensemble:
    model_idx = num_ensemble-1

normal_final_model = MVTecCNN_BO(best_model_num_channels[model_idx]).to(device)
normal_final_model.load_state_dict(torch.load(savepath_normal, map_location=device))

normal_model_test_acc = evaluate_accuracy(normal_final_model.eval(), test_loader, device)

with open(os.path.join(save_path,'active_results.txt'), 'a') as f:
    
    f.write(f"normal_target-{target_class}_ensemble-{num_ensemble}_unlabel-{unlabeled_ratio}_annotate-{sample_ratio}\n")
    f.write(f'original_model_test_acc:{original_model_test_acc}\nnormal_model_test_acc:{normal_model_test_acc}\nactive_model_test_acc:{active_model_test_acc}\n\n')    
    
    



