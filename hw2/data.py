import os
import json
import glob
import torch
import scipy.misc
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ImageDataset(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data = []
        if mode == 'train':
            self.data_dir = args.train_data_dir
            self.img_dir = os.path.join(self.data_dir, 'img')
            self.label_dir = os.path.join(self.data_dir, 'seg')
            self.data = self.loop_all_file(self.img_dir, self.label_dir, mode) # Return a list of tupple(img_path, label_path)
        elif mode == 'val':
            self.data_dir = args.valid_data_dir
            self.img_dir = os.path.join(self.data_dir, 'img')
            self.label_dir = os.path.join(self.data_dir, 'seg')
            self.data = self.loop_all_file(self.img_dir, self.label_dir, mode) # Return a list of tupple(img_path, label_path)
        elif mode == 'test':
            self.img_dir = args.test_data_dir
            self.data = self.loop_all_file(self.img_dir, '', mode) # Return a list of tupple(img_path, '')
            ## There is no label_dir

        
        ''' set up image transform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
            
        elif self.mode == 'val':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        ''' get data path '''
        img_path, label_path = self.data[idx] # get a tupple
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        if label_path != '':
            label = Image.open(label_path)
            label = np.array(label)
            label = torch.tensor(label, dtype=torch.long)
                return self.transform(img), label, img_path
        else:
            return self.transform(img), img_path # There is no label when testing
    
    def loop_all_file(self, img_path, label_path, mode):
        all_file_path = []
        if mode == 'train':
            assert len(os.listdir(img_path))/2 == 5460
            for i in range(int(len(os.listdir(img_path))/2)):
                tmp = '0000' + str(i)
                cur_img_path = os.path.join(img_path, tmp[-4:]+'.png')
                cur_label_path = os.path.join(label_path, tmp[-4:]+'.png')
                flip_img_path = os.path.join(img_path, 'flip_' + tmp[-4:]+'.png')
                flip_label_path = os.path.join(label_path, 'flip_' + tmp[-4:]+'.png')
                all_file_path.append((cur_img_path, cur_label_path))
                all_file_path.append((flip_img_path, flip_label_path))
                
        elif mode == 'val':
            for i in range(int(len(os.listdir(img_path)))):
                tmp = '0000' + str(i)
                cur_img_path = os.path.join(img_path, tmp[-4:]+'.png')
                cur_label_path = os.path.join(label_path, tmp[-4:]+'.png')
                all_file_path.append((cur_img_path, cur_label_path))
                
        elif mode == 'test':
            files = glob.glob( os.path.join(img_path, '*.png') )
            files = sorted(files)
            for infile in files:
                all_file_path.append((infile, ''))
            
#              for i in range(len(os.listdir(img_path))):
#                 if i < 10000:
#                     tmp = '0000' + str(i)
#                     cur_img_path = os.path.join(img_path, tmp[-4:]+'.png')
#                     all_file_path.append((cur_img_path, ''))
#                 else:
#                     tmp = '0000' + str(i)
#                     cur_img_path = os.path.join(img_path, tmp[-5:]+'.png')
#                     all_file_path.append((cur_img_path, ''))
            
        return all_file_path
