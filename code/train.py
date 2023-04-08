import numpy as np
import argparse
import os
import sys
import shutil
import time
import logging
import boto3

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as torchData
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import smdebug.pytorch as smd

from torch.optim import lr_scheduler
from PIL import ImageFile
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


learning_rate = 1e-3
learning_rate_decay = 10
best_prec = 0
train_loss_list = []
val_acc_list= []
test_acc_list= []

def test(model, test_loader, criterion, device, hook, file_out):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    batch_time = AverageMeter()
    losses = AverageMeter()
    test_acc = AverageMeter()
    
    if file_out==True:
        f = open('testing_result.txt','w') 

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            test_loss = criterion(outputs, labels)

            # measure accuracy and record loss
            score,idx = torch.max(outputs.data,1)
            correct = (labels==idx)
            acc = float(correct.sum())/inputs.size(0)

            if file_out==True:
                for j in range(inputs.size(0)):
                    f.write('%d\n' % idx[j])

            # measure accuracy and record loss
            losses.update(test_loss.data, inputs.size(0))
            test_acc.update(acc, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Testing: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {test_acc.val:.3f} ({test_acc.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   test_acc=test_acc))

        print('*Testing Precision: {test_acc.avg:.3f}'.format(test_acc=test_acc))
        if file_out==True:
            f.close()

        test_acc_list.append(test_acc.avg)


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook, checkpoint_path, file_out):
    global best_prec
    image_dataset={'train':train_loader, 'valid':validation_loader}

    snapshot_fname = checkpoint_path.replace("resnet34_best.pth.tar", "resnet34.pth.tar")
    snapshot_best_fname = checkpoint_path

    for epoch in range(epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, learning_rate, learning_rate_decay)
        logger.info(f"Epoch: {epoch}")
        
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)

                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                train_acc = AverageMeter()

                end = time.time()
                for i, (inputs, labels) in enumerate(image_dataset[phase]):
                    data_time.update(time.time() - end)

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    input_var = torch.autograd.Variable(inputs)
                    target_var = torch.autograd.Variable(labels)

                    outputs = model(input_var)
                    loss = criterion(outputs, target_var)

                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # measure accuracy and record loss
                    score,idx = torch.max(outputs.data,1)
                    correct = (labels==idx)
                    acc = float(correct.sum())/inputs.size(0)

                    losses.update(loss.data, inputs.size(0))
                    train_acc.update(acc, inputs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    print('Epoch: [{0}][{1}/{2}] lr {cur_lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec {train_acc.val:.3f} ({train_acc.avg:.3f})'.format(
                           epoch, i, len(train_loader), cur_lr=cur_lr, batch_time=batch_time,
                           data_time=data_time, loss=losses, train_acc=train_acc))
                    train_loss_list.append(losses.val)
            
            if phase=='valid':
                model.eval()
                hook.set_mode(smd.modes.EVAL)

                batch_time = AverageMeter()
                losses = AverageMeter()
                val_acc = AverageMeter()
                if file_out==True:
                    f = open('counting_result.txt','w') 

                end = time.time()
           
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(image_dataset[phase]):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        valid_loss = criterion(outputs, labels)

                        # measure accuracy and record loss
                        score,idx = torch.max(outputs.data,1)
                        correct = (labels==idx)
                        acc = float(correct.sum())/inputs.size(0)

                        if file_out==True:
                            for j in range(inputs.size(0)):
                                f.write('%d\n' % idx[j])

                        # measure accuracy and record loss
                        losses.update(valid_loss.data, inputs.size(0))
                        val_acc.update(acc, inputs.size(0))

                        # measure elapsed time
                        batch_time.update(time.time() - end)
                        end = time.time()

                        print('validation: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(
                               i, len(validation_loader), batch_time=batch_time, loss=losses,
                               val_acc=val_acc))

                    print('*Validation Precision: {val_acc.avg:.3f}'.format(val_acc=val_acc))
                    if file_out==True:
                        f.close()

                    val_acc_list.append(val_acc.avg)
                    prec = val_acc.avg
        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        torch.save({ 
            'epoch': epoch + 1,
            'arch': "resnet34",
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'train_loss_list': train_loss_list,
            'val_acc_list': val_acc_list,
        }, snapshot_fname)
        if is_best:
            shutil.copyfile(snapshot_fname,snapshot_best_fname)

    return model

    
def net(checkpoint_path, device):
    if(checkpoint_path != None and torch.cuda.is_available()):
        # create model
        print("creating model 'resnet34' from checkpoint")
        model = models.__dict__["resnet34"]()

        in_features = model.fc.in_features
        new_fc = nn.Linear(in_features, 6)
        model.fc = new_fc
        model.to(device)

        cudnn.benchmark = True

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        train_loss_list = checkpoint['train_loss_list']
        val_acc_list = checkpoint['val_acc_list']
        
    else:
        print("creating model 'resnet50' from scratch")
        model = models.resnet34()

        for param in model.parameters():
            param.requires_grad = False   

        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, 6)
        model.to(device)
    return model


def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = datasets.ImageFolder(root=test_data_path, transform=train_transform)
    train_data_loader = torchData.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torchData.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torchData.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    global  best_prec, train_loss_list, val_acc_list, test_acc_list, learning_rate, learning_rate_decay, output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    output_dir = args.output_dir
    checkpoint_path = os.path.join(args.data,'models/resnet34_best.pth.tar')
    learning_rate = args.learning_rate    
    
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    
    s3 = boto3.client('s3')
    s3.download_file('udacity-capstone-project-2023', 'models/resnet34_best.pth.tar', checkpoint_path)    
    
    model=net(checkpoint_path, device)
    
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
#    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(),
                          args.learning_rate,
                         momentum=0.9,
                         weight_decay=1e-4)

    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(criterion)
#    hook = None
    
    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook, checkpoint_path,True)
    
    logger.info("Testing Model")
    test(model, test_loader, criterion, device, hook, True)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, learning_rate, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lrd epochs"""
    lr = learning_rate * (0.1 ** (epoch // learning_rate_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_decay', type=int, default=10)
    parser.add_argument('--epochs', type=int , default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)
