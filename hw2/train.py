import os
import torch

import parser
import models
import data
import test
from mean_iou_evaluate import mean_iou_score

import pdb
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from test import evaluate


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)     

  

if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

#     ''' setup GPU '''
#     torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.ImageDataset(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.ImageDataset(args, mode='val'),
                                               batch_size=args.test_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    
       
    if args.baseline:
        ''' load model '''
        print('===> Prepare Baseline Model ...')
        baseline_model = models.BaselineNet(args)
        baseline_model.to(device)

        ''' define loss '''
        criterion = nn.CrossEntropyLoss()

        ''' setup optimizer '''
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        ''' train baseline model '''
        print('===> Start training baseline model ...')
        best_mean_iou = 0
        for epoch in range(1, args.epoch+1):

            baseline_model.train()
            
            iou_outputs = []
            iou_labels = []

            trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Baseline")
            
            for idx, (imgs, labels, img_paths) in trange:
                ### imgs = [32, 3, 352, 448],  labels = [32, 352, 448] 

                ''' move data to gpu '''
                imgs, labels = imgs.to(device), labels.to(device)

                ''' forward path '''
                output, result = baseline_model(imgs)
                
                iou_outputs.append(result.cpu().detach().numpy())
                iou_labels.append(labels.cpu().detach().numpy())
                

                ''' compute loss, backpropagation, update parameters '''
                loss = criterion(output, labels) # compute loss
                trange.set_postfix({'Epoch': epoch, 'loss' : '{0:.5f}'.format(loss.item())})
                with open(os.path.join(args.save_dir+'Baseline', 'baseline_loss.txt'), 'a') as f:
                    f.write('{0:.4f}'.format(loss.item())+'\n')

                baseline_optimizer.zero_grad()         # set grad of all parameters to zero
                loss.backward()               # compute gradient for each parameters
                baseline_optimizer.step()              # update parameters

            ''' calculate iou score while training '''
            iou_outputs = np.concatenate(iou_outputs)
            iou_labels = np.concatenate(iou_labels)
            print("Training iou:")
            training_mean_iou = mean_iou_score(iou_outputs, iou_labels)
    
    
            ''' evaluate the model, calculate iou score on validation data '''
            output, mean_iou = evaluate(baseline_model, val_loader, device, args) 
            
            with open(os.path.join(args.save_dir+'Baseline', 'baseline_val_iou.txt'), 'a') as f:
                f.write('{0:.4f}'.format(mean_iou)+'\n')

            ''' save best model '''
            path = os.path.join(args.model_dir, 'Baseline')
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                save_model(baseline_model, os.path.join(path, 'baseline_model_best.pth.tar'))
            
    
    #################################################      Improved Model 
    else:
        ''' load model '''
        print('===> Prepare Improved Model ...')
        improved_model = models.ImprovedNet(args)
        improved_model.to(device)

        ''' define loss '''
        criterion = nn.CrossEntropyLoss()

        ''' setup optimizer '''
        improved_optimizer = torch.optim.Adam(improved_model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

        ''' train baseline model '''
        print('===> Start training improved model ...')
        best_mean_iou = 0
        iou_output = []
        iou_label = []
        
        for epoch in range(1, args.epoch+1):

            improved_model.train()
            improved_optimizer.zero_grad()         # set grad of all parameters to zero
            
            trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Improved")
            
            for idx, (imgs, labels, img_paths) in trange:
                ### imgs = [32, 3, 352, 448],  labels = [32, 352, 448] 
                imgs, labels = imgs.to(device), labels.to(device)

                ''' forward path '''
                output, result = improved_model(imgs)
                
                iou_output.append(result.cpu().detach().numpy())
                iou_label.append(labels.cpu().detach().numpy())

                ''' compute loss, backpropagation, update parameters '''
                loss = criterion(output, labels) # compute loss
                trange.set_postfix({'Epoch' : epoch,'loss' : '{0:.5f}'.format(loss.item())})
                with open(os.path.join(args.save_dir+'Improved', 'improved_loss.txt'), 'a') as f:
                    f.write('{0:.4f}'.format(loss.item())+'\n')
                
                loss.backward()               # compute gradient for each parameters
                
                if idx%2 == 1:
                    improved_optimizer.step()             # update parameters
                    improved_optimizer.zero_grad()         # set grad of all parameters to zero

            ''' calculate iou score while training '''
            iou_outputs = np.concatenate(iou_output)
            iou_output.clear()
            iou_labels = np.concatenate(iou_label)
            iou_label.clear()
            print("Training iou:")
            training_mean_iou = mean_iou_score(iou_outputs, iou_labels)
            
        
            ''' evaluate the model, calculate iou score on validation data '''
            output, mean_iou = evaluate(improved_model, val_loader, device, args)      
            with open(os.path.join(args.save_dir+'Improved', 'improved_val_iou.txt'), 'a') as f:
                f.write('{0:.4f}'.format(mean_iou)+'\n')

            ''' save best model '''
            path = os.path.join(args.model_dir, 'Improved')
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                save_model(improved_model, os.path.join(path, 'improved_model_best.pth.tar'))


