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
    num_epoch = 30  #수정 30
    
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
        
    save_esemble_models(best_val_acc, best_model.eval())
    return best_val_acc

def cnn_function(lr, num_channel):
    num_channel = int(8 + num_channel*54)   # min 8, max 64
    best_val_accuracy = train(lr, num_channel)    
    return best_val_accuracy


def augmentDataset(labeled_dataset, unlabeled_dataset, idx):
    active_data_list = np.array(unlabeled_dataset.data_list)[idx]
    active_label_list = np.array(unlabeled_dataset.label_list)[idx]
    
    labeled_dataset.data_list = labeled_dataset.data_list + active_data_list.tolist()
    labeled_dataset.label_list = labeled_dataset.label_list + active_label_list.tolist()

    val_num = int(len(labeled_dataset)*0.20)
    train_num = len(labeled_dataset)  - val_num

    new_train_dataset, new_valid_dataset =random_split(labeled_dataset,[train_num, val_num])

    new_train_loader = DataLoader(new_train_dataset, batch_size=8, shuffle=True)
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
    net.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_val_acc = 0.
    
    num_epoch = 30  #수정 30
    
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


# unlabled, labeled 데이터 분리 
labeled_normal_data_list, labeled_abnormal_data_list, unlabeled_normal_data_list, unlabeled_abnormal_data_list = loadActiveDataset(target_class)
unlabeled_dataset = MVTecActiveDataset(unlabeled_normal_data_list, unlabeled_abnormal_data_list, isUnlabeled=True)
labeled_dataset = MVTecActiveDataset(labeled_normal_data_list, labeled_abnormal_data_list, isUnlabeled=False)


val_num = int(len(labeled_dataset)*0.15)
test_num = int(len(labeled_dataset)*0.15)
train_num = len(labeled_dataset)  - val_num - test_num

train_dataset, valid_dataset, test_dataset =random_split(labeled_dataset,[train_num, val_num, test_num])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

active_loader = DataLoader(unlabeled_dataset, batch_size=activeBatchSize, shuffle=False)

# 라벨된 데이터만 활용해 기본 모델  학습 및 베스트 모델 Top K개 뽑아서 uncertainty estimation에  활용
pbounds = {'lr': (1e-3, 0.1), 'num_channel':(0, 1)}

optimizer = BayesianOptimization(
    f=cnn_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=3,
    n_iter=40,  # 수정 40
)

# 나중에 모델 생성을 위한 베스트 모델들의 channel depth 기록
best_model_num_channels=[]
for i in range(num_ensemble):
    best_model_num_channels.append(best_models[i].num_channel)

# 기본 모델의 테스트셋 정확도
original_model = best_models[np.argmax(best_accuracies)].eval()
original_model_test_acc = evaluate_accuracy(original_model, test_loader, device)

# 모델 K개의 ensemble prediction을 통해 unlabeled dataset 각 sample의 엔트로피 계산
for idx, (img, label) in enumerate(active_loader):
    img = img.to(device)
    label = label.to(device, dtype=torch.float)
       
    for model_idx in best_models:

        best_models[model_idx]
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
active_train_loader, active_valid_loader = augmentDataset(labeled_dataset, unlabeled_dataset, active_idx.cpu())

rand_idx = np.random.permutation(len(unlabeled_dataset))[:query_num]
normal_train_loader, normal_valid_loader = augmentDataset(labeled_dataset, unlabeled_dataset, rand_idx)

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
    f.write(f'original_model_test_acc: {original_model_test_acc} \t active_model_test_acc:{active_model_test_acc} \
\t normal_model_test_acc:{normal_model_test_acc}\n\n')
    
    
    



