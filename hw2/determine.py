import os
import torch

import parser
import models
import data
import test
from mean_iou_evaluate import mean_iou_score

import pdb
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from test import evaluate


def load(path, mode):
    if mode == 'base':
        baseline_model = models.BaselineNet(args)
        baseline_model.load_state_dict(torch.load(path)) 
        return baseline_model.to(device)
    else:
        improved_model = models.ImprovedNet(args)
        improved_model.load_state_dict(torch.load(path)) 
        return improved_model.to(device)

  

if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.test_out_dir):
        os.makedirs(args.test_out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    test_loader = torch.utils.data.DataLoader(data.ImageDataset(args, mode='test'),
                                               batch_size=args.test_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    
       
    if args.baseline:
        ''' load model '''
        print('===> Prepare Baseline Model ...')
#         baseline_model = load(os.path.join(args.model_dir+'Baseline', 'baseline_model_best.pth.tar'), 'base')
        baseline_model = load(args.model_dir, 'base')

        ''' test baseline model '''
        print('===> Start teting baseline model ...')
        trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Baseline")
            
        for idx, (imgs, img_paths) in trange:
            ### imgs = [32, 3, 352, 448],  labels = [32, 352, 448] 
            ''' move data to gpu '''
            imgs = imgs.to(device)

            ''' forward path '''
            output, result = baseline_model(imgs)
            
            result = result.cpu().detach().numpy()
            
            ''' getting result ans save '''
            
            for i, class_map in enumerate(result):
                ## class_map = (352, 448)
                new_im = Image.fromarray(np.uint8(class_map))
                filename = img_paths[i].split('/')[-1]
                save_map_path = os.path.join(args.test_out_dir, filename)
                new_im.save(save_map_path)
   
    
    #################################################      Improved Model 
    else:
        ''' load model '''
        print('===> Prepare Improved Model ...')
#         improved_model = load(os.path.join(args.model_dir+'Improved', 'improved_model_14_best.pth.tar'), 'improved')
        improved_model = load(args.model_dir, 'improved')


        ''' test improved model '''
        print('===> Start teting improved model ...')
        trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Improved")
            
        for idx, (imgs, img_paths) in trange:
            ### imgs = [32, 3, 352, 448],  labels = [32, 352, 448] 
            ''' move data to gpu '''
            imgs = imgs.to(device)

            ''' forward path '''
            output, result = improved_model(imgs)
            
            result = result.cpu().detach().numpy()
            
            ''' getting result ans save '''
            
            for i, class_map in enumerate(result):
                ## class_map = (352, 448)
                new_im = Image.fromarray(np.uint8(class_map))
                filename = img_paths[i].split('/')[-1]
                save_map_path = os.path.join(args.test_out_dir, filename)
                new_im.save(save_map_path)
                


