import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import BowelDataset
from model import resnet18
from torchvision import transforms
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random

# 定义设备位置
import os

# 中间过程存储函数
def saverecord(savepath, epoch, epochmean, epochmean_val, acc_adenoma, acc_cancer, acc_normal, acc_polyp, acc_class_mean, acc_mean, acc_adenoma_val, acc_cancer_val, acc_normal_val, acc_polyp_val, acc_class_mean_val, acc_mean_val):
    fo = open(savepath, 'a+')
    fo.write('train_loss_mean{}:{}'.format(epoch, epochmean))
    fo.write('\n')
    fo.write('val_loss_mean{}:{}'.format(epoch, epochmean_val))
    fo.write('\n')
    fo.write('train_acc_adenoma_{}:{}'.format(epoch, acc_adenoma))
    fo.write('\n')
    fo.write('train_acc_cancer_{}:{}'.format(epoch, acc_cancer))
    fo.write('\n')
    fo.write('train_acc_normal_{}:{}'.format(epoch, acc_normal))
    fo.write('\n')
    fo.write('train_acc_polyp_{}:{}'.format(epoch, acc_polyp))
    fo.write('\n')
    fo.write('train_acc_class_mean{}:{}'.format(epoch, acc_class_mean))
    fo.write('\n')
    fo.write('train_acc_mean{}:{}'.format(epoch, acc_mean))
    fo.write('\n')
    fo.write('val_acc_adenoma_{}:{}'.format(epoch, acc_adenoma_val))
    fo.write('\n')
    fo.write('val_acc_cancer_{}:{}'.format(epoch, acc_cancer_val))
    fo.write('\n')
    fo.write('val_acc_normal_{}:{}'.format(epoch, acc_normal_val))
    fo.write('\n')
    fo.write('val_acc_polyp_{}:{}'.format(epoch, acc_polyp_val))
    fo.write('\n')
    fo.write('val_acc_class_mean{}:{}'.format(epoch, acc_class_mean_val))
    fo.write('\n')
    fo.write('val_acc_mean{}:{}'.format(epoch, acc_mean_val))
    fo.write('\n')
    fo.close()

def classacc(predicted, label):  # 计算每一类的准确率
    acc = label.float()+predicted.float()
    acc_adenoma = (acc == 0).sum().float()
    acc_polyp = (acc == 6).sum().float()
    acc_temp = label.float()*predicted.float()
    acc_cancer = (acc_temp == 1).sum().float()
    acc_normal = (acc_temp == 4).sum().float()
    return acc_adenoma.item(), acc_cancer.item(), acc_normal.item(), acc_polyp.item()

# 定义模型训练函数
def train_model(model, criterion, optimizer, train_dataloaders, val_dataloaders, num_epochs, record_path, model_path, best_path):
    print(num_epochs, len(train_dataloaders))
    dt_size = len(train_dataloaders.dataset)
    dt_size_val = len(val_dataloaders.dataset)
    best_val_acc = 0
    for epoch in np.arange(0, num_epochs) + 1:
        model.train()
        print("=======Epoch:{}=======".format(epoch))
        epoch_start_time = time.time()  #获取当前时间
        epoch_loss = 0   # 这个里面存的是每一轮里面的累计loss值
        # acc = 0
        step = 0
        correct = 0.0
        correct_adenoma = 0.0
        correct_cancer = 0.0
        correct_normal = 0.0
        correct_polyp = 0.0
        total = 0.0
        total_adenoma = 0.0
        total_cancer = 0.0
        total_normal = 0.0
        total_polyp = 0.0
        for idx, (input1, labels) in tqdm(enumerate(train_dataloaders), total=len(train_dataloaders)):   # 这边一次取出一个batchsize的东西
            step += 1
            input1, labels = input1.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input1)
            loss  = criterion(outputs, labels)
            current_batchsize = outputs.size()[0]
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            epoch_loss += loss.item()*current_batchsize 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            correct_adenoma_temp, correct_cancer_temp, correct_normal_temp, correct_polyp_temp = classacc(predicted, labels)
            correct_adenoma += correct_adenoma_temp
            correct_cancer += correct_cancer_temp
            correct_normal += correct_normal_temp
            correct_polyp  += correct_polyp_temp
            total_adenoma += (labels == 0).sum().float()
            total_cancer += (labels == 1).sum().float()
            total_normal += (labels == 2).sum().float()
            total_polyp += (labels == 3).sum().float()
            #print("%d/%d,train_loss:%0.8f,Acc: %.3f%% " % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item(),100. * correct / total))   # 这个是输出每一步的loss值，step/一共要多少轮
        # 一轮训练结束之后
        # print("train_loss_%d"%epoch, epoch_loss)
        epochmean = epoch_loss/total
        acc_adenoma = correct_adenoma/total_adenoma
        acc_cancer = correct_cancer/total_cancer
        acc_normal = correct_normal/total_normal
        acc_polyp = correct_polyp/total_polyp
        acc_class_mean = (acc_adenoma + acc_cancer + acc_normal + acc_polyp)/4
        acc_mean = correct/total
        print("train_loss_mean_%d"%epoch, epochmean)
        print("train_acc_adenama_mean_%d"%epoch, acc_adenoma)
        print("train_acc_cancer_mean_%d"%epoch, acc_cancer)
        print("train_acc_normal_mean_%d"%epoch, acc_normal)
        print("train_acc_polyp_mean_%d"%epoch, acc_polyp)
        print("train_acc_class_mean_%d"%epoch, acc_class_mean)
        print("train_acc_mean_%d"%epoch, acc_mean)

        # 在验证集上进行评估
        model.eval()
        with torch.no_grad():  
            epoch_loss_val = 0
            step_val = 0
            correct = 0.0
            correct_adenoma = 0.0
            correct_cancer = 0.0
            correct_normal = 0.0
            correct_polyp = 0.0
            total = 0.0
            total_adenoma = 0.0
            total_cancer = 0.0
            total_normal = 0.0
            total_polyp = 0.0
            for idx, (input1, labels) in tqdm(enumerate(val_dataloaders), total=len(val_dataloaders)):    
                step_val += 1
                input1, labels = input1.to(device), labels.to(device)
                outputs = model(input1)
                loss  = criterion(outputs, labels)
                current_batchsize = outputs.size()[0]
                epoch_loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_adenoma_temp, correct_cancer_temp, correct_normal_temp, correct_polyp_temp = classacc(predicted, labels)
                correct_adenoma += correct_adenoma_temp
                correct_cancer += correct_cancer_temp
                correct_normal += correct_normal_temp
                correct_polyp += correct_polyp_temp
                total_adenoma += (labels == 0).sum().float()
                total_cancer += (labels == 1).sum().float()
                total_normal += (labels == 2).sum().float()
                total_polyp += (labels == 3).sum().float()
                #print("%d/%d,val_loss:%0.8f,Acc: %.3f%% " % (step, (dt_size - 1) // val_dataloaders.batch_size + 1, loss.item(),100. * correct / total))   # 这个是输出每一步的loss值，step/一共要多少轮
            # 一轮训练结束之后
            # print("val_loss_%d"%epoch, epoch_loss)
            epochmean_val = epoch_loss_val/dt_size_val
            
            acc_adenoma_val = correct_adenoma/total_adenoma
            acc_cancer_val = correct_cancer/total_cancer
            acc_normal_val = correct_normal/total_normal
            acc_polyp_val = correct_polyp/total_polyp
            acc_class_mean_val = (acc_adenoma_val + acc_cancer_val + acc_normal_val + acc_polyp_val)/4
            acc_mean_val = correct/total
            print("val_acc_adenoma_mean_%d"%epoch, acc_adenoma_val)
            print("val_acc_cancer_mean_%d"%epoch, acc_cancer_val)
            print("val_acc_normal_mean_%d"%epoch, acc_normal_val)
            print("val_acc_polyp_mean_%d"%epoch, acc_polyp_val)
            print("val_acc_class_mean_%d"%epoch, acc_class_mean_val)
            print("val_acc_mean_%d"%epoch, acc_mean_val)
            if acc_mean_val > best_val_acc:
                best_val_acc = acc_mean_val
                torch.save(model.state_dict(), best_path + '{}_{}.pth'.format(epoch, acc_class_mean_val))

            print("%2.2f sec(s)"%(time.time() - epoch_start_time))
        # 对结果进行存储
        saverecord(record_path, epoch, epochmean, epochmean_val, acc_adenoma, acc_cancer, acc_normal, acc_polyp, acc_class_mean, acc_mean, acc_adenoma_val, acc_cancer_val, acc_normal_val, acc_polyp_val, acc_class_mean_val, acc_mean_val)
        torch.cuda.empty_cache()
        if epoch % 5 == 0:  # 每隔5轮存一下
            torch.save(model.state_dict(), model_path + '{}.pth'.format(epoch))   

    return model


if __name__ == '__main__':
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    # 定义device
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    # 定义网络及相关训练参数
    model = resnet18(num_classes=4, inputchannel=3).to(device)
    batch_size = 32
    num_epochs = 600
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lr_decay = 0.965
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    transform_train = transforms.Compose([
        transforms.Resize([400,400]),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        # transforms.RandomRotation(60),
        # transforms.RandomRotation(120),
        transforms.RandomVerticalFlip(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize([400,400]),
    ])
    # 定义数据变换及数据集

    train_dataset = BowelDataset(root = 'training set path',transform = transform_train)    # 这是训练集
    print('训练集长度:', len(train_dataset))
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = BowelDataset(root = 'val set path',transform = transform_val)       # 这是验证集
    print('验证集长度:', len(val_dataset))
    val_dataloaders = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=2)

    # 设置存储路径
    record_path = 'The path where the training results are stored'  # 一些训练结果存储
    model_path = 'The path where the model is stored'  # 间隔存一下模型
    best_path = 'The path where the model with the best result on the validation set is stored'  # 存验证集上最好模型
    train_model(model, criterion, optimizer, train_dataloaders, val_dataloaders, num_epochs, record_path, model_path, best_path)