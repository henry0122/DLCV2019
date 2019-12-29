import os
import torch
import parser
import test
import pdb
import glob
import random
import numpy as np
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.autograd import Function
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
'''  setup random seed and device '''
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)   
    
    
''' --------------------------------------- Start of Dataset -------------------------------------------'''
class MnistmDataset(Dataset):
    def __init__(self, args, mode, data_src='src'):
        self.mode = mode
        ''' set up basic parameters for dataset '''
        if mode == 'train':
            self.data_dir = os.path.join(args.dann_mnistm_data_dir, 'train')
            self.data = self.loop_all_file(self.data_dir)
            ''' Need to load label '''
            label_csv = pd.read_csv(os.path.join(args.dann_mnistm_data_dir, 'train.csv')) 
            self.label = torch.tensor(np.squeeze(label_csv[['label']].to_numpy()), dtype=torch.int64)
        elif mode == 'test':
            self.data_dir = os.path.join(args.dann_mnistm_data_dir, 'test')
            self.data = self.loop_all_file(self.data_dir)
            ''' Need to load label '''
            label_csv = pd.read_csv(os.path.join(args.dann_mnistm_data_dir, 'test.csv')) 
            self.label = torch.tensor(np.squeeze(label_csv[['label']].to_numpy()), dtype=torch.int64)
        elif mode == 'final':
            self.data_dir = args.dann_tar_data_dir
            self.data = self.loop_all_file(self.data_dir)
                
        ''' set up image transform '''
        self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'test':
            img_path = self.data[idx]
            img_label = self.label[idx]
            img = Image.open(img_path).convert('RGB')

            filename = img_path.split('/')[-1]

            ## after transfoming -> [3, 64, 64]
            return self.transform(img), img_label, filename 
        elif self.mode == 'final':
            img_path = self.data[idx]
            img = Image.open(img_path).convert('RGB')

            filename = img_path.split('/')[-1]

            ## after transfoming -> [3, 64, 64]
            return self.transform(img), filename 
    
    def loop_all_file(self, img_path):

        files = glob.glob( os.path.join(img_path, '*.png') )
        files_path = sorted(files)
            
        return files_path
    
    
class SvhnDataset(Dataset):
    def __init__(self, args, mode, data_src='src'):
        self.mode = mode
        ''' set up basic parameters for dataset '''
        if mode == 'train':
            self.data_dir = os.path.join(args.dann_svhn_data_dir, 'train')
            self.data = self.loop_all_file(self.data_dir)
            ''' Need to load label '''
            label_csv = pd.read_csv(os.path.join(args.dann_svhn_data_dir, 'train.csv')) 
            self.label = torch.tensor(np.squeeze(label_csv[['label']].to_numpy()), dtype=torch.int64)
        elif mode == 'test':
            self.data_dir = os.path.join(args.dann_svhn_data_dir, 'test')
            self.data = self.loop_all_file(self.data_dir)
            ''' Need to load label '''
            label_csv = pd.read_csv(os.path.join(args.dann_svhn_data_dir, 'test.csv')) 
            self.label = torch.tensor(np.squeeze(label_csv[['label']].to_numpy()), dtype=torch.int64)
            
        elif mode == 'final':
            self.data_dir = args.dann_tar_data_dir
            self.data = self.loop_all_file(self.data_dir)
                
        ''' set up image transform '''
        self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'test':
            img_path = self.data[idx]
            img_label = self.label[idx]
            img = Image.open(img_path).convert('RGB')

            filename = img_path.split('/')[-1]

            ## after transfoming -> [3, 64, 64]
            return self.transform(img), img_label, filename 
        elif self.mode == 'final':
            img_path = self.data[idx]
            img = Image.open(img_path).convert('RGB')

            filename = img_path.split('/')[-1]

            ## after transfoming -> [3, 64, 64]
            return self.transform(img), filename 
    
    def loop_all_file(self, img_path):

        files = glob.glob( os.path.join(img_path, '*.png') )
        files_path = sorted(files)
            
        return files_path
    
''' --------------------------------------- End of Dataset ------------------------------------------''' 
    
class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
''' --------------------------------------- Start of Model ------------------------------------------ '''
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(True)
            )

        self.class_classifier = nn.Sequential(
                nn.Linear(50 * 4 * 4, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Dropout2d(),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, 10),
                nn.LogSoftmax()
            )

    def forward(self, x):
        x = self.feature(x)
        img_embed = x.view(-1, 50 * 4 * 4)
        output = self.class_classifier(img_embed)
        return output



class DANN(nn.Module):

    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(True)
            )

        self.class_classifier = nn.Sequential(
                nn.Linear(50 * 4 * 4, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Dropout2d(),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, 10),
                nn.LogSoftmax()
            )

        self.domain_classifier = nn.Sequential(
                nn.Linear(50 * 4 * 4, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, 2),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, imgs, alpha):
        img_embed = self.feature(imgs)
        img_embed = img_embed.view(-1, 50 * 4 * 4)
        
        reverse_img_embed = ReverseLayer.apply(img_embed, alpha)
        
        class_output = self.class_classifier(img_embed)
        domain_output = self.domain_classifier(reverse_img_embed)

        return class_output, domain_output, img_embed
    
''' --------------------------------------- End of Model ------------------------------------------ '''
    
    
def train(args):
    ''' only one condition wil be executed '''
    if args.tar_dom_name == 'svhn':
        if args.is_lower:
            ''' train mnistm test on svhn '''
            ''' at the same time, train mnistm test on mnistm '''
            dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'train', 'src'), 
                                                                         batch_size=args.train_batch, shuffle=True, num_workers=8) 
            test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
            upper_test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
            
            cnnNet = CNNNet().to(device)
            optimizer = optim.Adadelta(cnnNet.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            
            for epoch in range(1, args.epoch + 1):
                cnnNet.train()
                trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="Svhn lower-b")
                for i, (imgs, imgs_label, imgs_path) in trange:
                    imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                    optimizer.zero_grad()
                    output = cnnNet(imgs)
                    loss = F.nll_loss(output, imgs_label)
                    loss.backward()
                    optimizer.step()
                    trange.set_postfix({"epoch":"{}".format(epoch),"loss":"{0:.5f}".format(loss.item())})
                    
                ### Evaluate once a epoch
                cnnNet.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for imgs, imgs_label, imgs_path in test_dataloader:
                        imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                        output = cnnNet(imgs)
                        test_loss += F.nll_loss(output, imgs_label, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                        
                test_loss /= len(test_dataloader)
                log = open(os.path.join(args.log_dir, 'm_s_lower.txt'), 'a')
                print('\nEpoch: {} | Svhn testing set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))
                log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(test_dataloader.dataset)) )
                log.close()
                
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for imgs, imgs_label, imgs_path in upper_test_dataloader:
                        imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                        output = cnnNet(imgs)
                        test_loss += F.nll_loss(output, imgs_label, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                        
                test_loss /= len(upper_test_dataloader)
                log = open(os.path.join(args.log_dir, 'm_m_upper.txt'), 'a')
                print('\nEpoch: {} | Mnistm testing set: Accuracy: {}/{}({:.0f}%)\n'.format(
                    epoch, correct, len(upper_test_dataloader.dataset), 100. * correct / len(upper_test_dataloader.dataset)))
                log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(upper_test_dataloader.dataset)) )
                log.close()
                
                scheduler.step()
                
            ''' ------------------ End of training mnistm test on svhn ------------------------ '''
            
        elif args.is_upper:
            ''' train svhn test on svhn '''
            ''' at the same time, train svhn test on mnistm '''
            dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'train', 'src'), 
                                                                         batch_size=args.train_batch, shuffle=True, num_workers=8) 
            test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
            lower_test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
            
            cnnNet = CNNNet().to(device)
            optimizer = optim.Adadelta(cnnNet.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            
            for epoch in range(1, args.epoch + 1):
                cnnNet.train()
                trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="Svhn upper-b")
                for i, (imgs, imgs_label, imgs_path) in trange:
                    imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                    optimizer.zero_grad()
                    output = cnnNet(imgs)
                    loss = F.nll_loss(output, imgs_label)
                    loss.backward()
                    optimizer.step()
                    trange.set_postfix({"epoch":"{}".format(epoch),"loss":"{0:.5f}".format(loss.item())})
                    
                ### Evaluate once a epoch
                cnnNet.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for imgs, imgs_label, imgs_path in test_dataloader:
                        imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                        output = cnnNet(imgs)
                        test_loss += F.nll_loss(output, imgs_label, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                        
                test_loss /= len(test_dataloader)
                log = open(os.path.join(args.log_dir, 's_s_upper.txt'), 'a')
                print('\nEpoch: {} | Svhn testing set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))
                log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(test_dataloader.dataset)) )
                log.close()
                
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for imgs, imgs_label, imgs_path in lower_test_dataloader:
                        imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                        output = cnnNet(imgs)
                        test_loss += F.nll_loss(output, imgs_label, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                        
                test_loss /= len(test_dataloader)
                log = open(os.path.join(args.log_dir, 's_m_lower.txt'), 'a')
                print('\nEpoch: {} | Svhn testing set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, correct, len(lower_test_dataloader.dataset), 100. * correct / len(lower_test_dataloader.dataset)))
                log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(lower_test_dataloader.dataset)) )
                log.close()
                
                scheduler.step()
                
            ''' ------------------ End of training svhn test on svhn ------------------------ '''
            
        else:
            ''' train mnistm and adapt to svhn then test on svhn '''
            source_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'train', 'src'), 
                                                                         batch_size=args.train_batch, shuffle=True, num_workers=8)
            target_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'train', 'src'), 
                                                                         batch_size=args.train_batch, shuffle=True, num_workers=8) 
            test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
            
            dann = DANN().to(device)
            optimizer = optim.Adam(dann.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

            class_loss = torch.nn.NLLLoss().to(device)
            domain_loss = torch.nn.NLLLoss().to(device)

            for p in dann.parameters():
                p.requires_grad = True
                
            best_acc = 40
            
            for epoch in range(1, args.epoch + 1):
                dann.train()
                dataloader_len = min(len(source_dataloader), len(target_dataloader))
                data_source_iter = iter(source_dataloader)
                data_target_iter = iter(target_dataloader)
                i = 0
                while i < dataloader_len:
                    ### some trick
                    p = (float(i + epoch * dataloader_len) / args.epoch) / dataloader_len
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    ''' training model using source data '''
                    source_data = data_source_iter.next()
                    s_img, s_label, _ = source_data
                    s_img, s_label = s_img.to(device), s_label.to(device)

                    dann.zero_grad()
                    
                    batch_size = len(s_label)
                    domain_label = torch.zeros(batch_size).long().to(device)

                    class_output, domain_output, s_img_embed = dann(s_img, alpha)
                    s_label_loss = class_loss(class_output, s_label)
                    s_domain_loss = domain_loss(domain_output, domain_label)

                    ''' training model using target data '''
                    target_data = data_target_iter.next()
                    t_img, _ , _ = target_data
                    t_img = t_img.to(device)

                    t_batch_size = t_img.shape[0]
                    t_domain_label = torch.ones(t_batch_size).long().to(device)

                    _, domain_output, t_img_embed = dann(t_img, alpha)
                    t_domain_loss = domain_loss(domain_output, t_domain_label)
                    total_loss = t_domain_loss + s_domain_loss + s_label_loss
                    total_loss.backward()
                    optimizer.step()

                    i += 1
                    
                scheduler.step()
                
                dann.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for imgs, imgs_label, imgs_path in test_dataloader:
                        imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                        class_output, domain_output, eval_img_embed = dann(imgs, alpha)
                        test_loss += class_loss(class_output, imgs_label).item()  # sum up batch loss
                        pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                        
                test_loss /= len(test_dataloader.dataset)
                log = open(os.path.join(args.log_dir, 'm_s_adapt.txt'), 'a')
                print('\nEpoch: {} | m_s_adapt: Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))
                log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(test_dataloader.dataset)) )
                log.close()
                
                if 100. * correct / len(test_dataloader.dataset) > best_acc:
                    best_acc = 100. * correct / len(test_dataloader.dataset)
                    path = os.path.join(args.model_dir, 'best_m_s_adapt.pth.tar')
                    save_model(dann, path)
                
            ''' --------------------- End of training mnistm and adapt to svhn then test on svhn -------------------- '''
            

    elif args.tar_dom_name == 'mnistm':
        ''' train svhn and adapt to mnistm then test on mnistm '''
        source_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'train', 'src'), 
                                                                     batch_size=args.train_batch, shuffle=True, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'train', 'src'), 
                                                                     batch_size=args.train_batch, shuffle=True, num_workers=8) 
        test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)

        dann = DANN().to(device)
        optimizer = optim.Adam(dann.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        class_loss = torch.nn.NLLLoss().to(device)
        domain_loss = torch.nn.NLLLoss().to(device)

        for p in dann.parameters():
            p.requires_grad = True

        best_acc = 40

        for epoch in range(1, args.epoch + 1):
            dann.train()
            dataloader_len = min(len(source_dataloader), len(target_dataloader))
            data_source_iter = iter(source_dataloader)
            data_target_iter = iter(target_dataloader)
            i = 0
            while i < dataloader_len:
                ### some trick
                p = (float(i + epoch * dataloader_len) / args.epoch) / dataloader_len
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                ''' training model using source data '''
                source_data = data_source_iter.next()
                s_img, s_label, _ = source_data
                s_img, s_label = s_img.to(device), s_label.to(device)

                dann.zero_grad()

                batch_size = len(s_label)
                domain_label = torch.zeros(batch_size).long().to(device)

                class_output, domain_output, s_img_embed = dann(s_img, alpha)
                s_label_loss = class_loss(class_output, s_label)
                s_domain_loss = domain_loss(domain_output, domain_label)

                ''' training model using target data '''
                target_data = data_target_iter.next()
                t_img, _ , _ = target_data
                t_img = t_img.to(device)

                t_batch_size = t_img.shape[0]
                t_domain_label = torch.ones(t_batch_size).long().to(device)

                _, domain_output, t_img_embed = dann(t_img, alpha)
                t_domain_loss = domain_loss(domain_output, t_domain_label)
                total_loss = t_domain_loss + s_domain_loss + s_label_loss
                total_loss.backward()
                optimizer.step()

                i += 1

            scheduler.step()

            dann.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for imgs, imgs_label, imgs_path in test_dataloader:
                    imgs, imgs_label = imgs.to(device), imgs_label.to(device) # [batch, 3, 28, 28]
                    class_output, domain_output, eval_img_embed = dann(imgs, alpha)
                    test_loss += class_loss(class_output, imgs_label).item()  # sum up batch loss
                    pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(imgs_label.view_as(pred)).sum().item()

            test_loss /= len(test_dataloader.dataset)
            log = open(os.path.join(args.log_dir, 's_m_adapt.txt'), 'a')
            print('\nEpoch: {} | s_m_adapt: Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))
            log.write( 'Epoch {} : {:.0f}\n'.format(epoch, 100. * correct / len(test_dataloader.dataset)) )
            log.close()

            if 100. * correct / len(test_dataloader.dataset) > best_acc:
                best_acc = 100. * correct / len(test_dataloader.dataset)
                path = os.path.join(args.model_dir, 'best_s_m_adapt.pth.tar')
                save_model(dann, path)

        ''' --------------------- End of training svhn and adapt to mnistm then test on mnistm ------------------- '''
        
    else:
        print("FUCK !!! Wrong name of dataset.")
    
def test(args):
    if args.tar_dom_name == 'svhn':
        test_dann = DANN().to(device)
#         model_path = os.path.join(args.model_dir, 'best_m_s_adapt.pth.tar')
        test_dann.load_state_dict(torch.load('best_m_s_adapt.pth.tar'))
        test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'final', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        correct = 0
        output = open(args.save_test_dann, 'a')
        output.write('image_name,label\n')
        
        with torch.no_grad():
            for imgs, imgs_path in test_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                 correct += pred.eq(imgs_label.view_as(pred)).sum().item()

                for p in range(len(imgs_path)):
                    output.write('{},{}\n'.format(imgs_path[p], int(pred[p])))
                    
        output.close()
#         print('\nTesting | m_s_adapt Accuracy: {}/{} ({:.0f}%)\n'.format(
#             correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))
    
    elif args.tar_dom_name == 'mnistm':
        test_dann = DANN().to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_adapt.pth.tar')
        test_dann.load_state_dict(torch.load('best_s_m_adapt.pth.tar'))
        test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'final', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        correct = 0
        output = open(args.save_test_dann, 'a')
        output.write('image_name,label\n')
        
        with torch.no_grad():
            for imgs, imgs_path in test_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                 correct += pred.eq(imgs_label.view_as(pred)).sum().item()
                
                for p in range(len(imgs_path)):
                    output.write('{},{}\n'.format(imgs_path[p], int(pred[p])))
                    
        output.close()
#         print('\nTesting | s_m_adapt: Accuracy: {}/{} ({:.0f}%)\n'.format(
#             correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))

    
if __name__=='__main__':
    args = parser.arg_parse()
    
    if not args.test_dann:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if not os.path.exists(args.fig_dir):
            os.makedirs(args.fig_dir)
        train(args)
    else:
        test(args)
    
    