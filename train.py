import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils import MVTecDataset, evaluate_accuracy
from models import MVTecSimpleCNN

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--target', required=True, help='target class')
parser.add_argument('-c', '--no_cuda', required=False, default=None, help='which cuda')
parser.add_argument('--lr', default= 0.001, type=float , required=False, help='learning rate')
parser.add_argument('--no_epoch', default= 30, type= int, required=False, help='number of epochs')

args = parser.parse_args()

target_class = args.target
save_path = os.path.join('./saves/', target_class)

if not os.path.exists(save_path):
    os.makedirs(save_path)

device_type='cuda'

if args.no_cuda is not None:
    device_type = 'cuda:'+str(args.no_cuda)

device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

normal_list_dir = [os.path.join('./data/', target_class, 'train', 'good'), os.path.join('./data/', target_class, 'test', 'good')]

test_dir = os.path.join('./data/', target_class, 'test')
test_subfolders = next(os.walk(test_dir))[1]

abnormal_list_dir=[]

for item in test_subfolders:
    if item != 'good':
        abnormal_list_dir.append(os.path.join('./data/', target_class, 'test', item))

# abnormal_list_dir = ['./data/bottle/test/broken_large/', './data/bottle/test/broken_small/', './data/bottle/test/contamination/']


dataset = MVTecDataset(normal_list_dir, abnormal_list_dir)

val_num = int(len(dataset)*0.15)
test_num = int(len(dataset)*0.15)
train_num = len(dataset) - val_num - test_num

train_dataset, valid_dataset, test_dataset =random_split(dataset,[train_num, val_num, test_num])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


learning_rate = args.lr
num_epoch = args.no_epoch


net = MVTecSimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

best_val_acc = 0.

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
                saved_model = os.path.join(save_path,   'epoch-{:02d}-acc-{:.3f}.pth'.format(epoch, val_acc))
                torch.save(net.state_dict(), saved_model)
                print('best valid accuracy: {:.3f} test accuracy: {:.3f}'.format(best_val_acc, evaluate_accuracy(net, test_loader, device)))
            net.train()

    print('epoch : {:02d} loss: {:.4f} train accuracy: {:.3f} '.format(epoch, loss_sum/loss_count, evaluate_accuracy(net, train_loader, device)))

model = MVTecSimpleCNN().to(device)
print('saved model: {}'.format(saved_model))
model.load_state_dict(torch.load(saved_model))
model.eval()
test_acc = evaluate_accuracy(model, test_loader, device)
print('Final test acc : ', test_acc)

with open('./saves/test_results.txt', 'a') as f:
    f.write('{:15}:{:.4f}\n'.format(target_class, test_acc))



