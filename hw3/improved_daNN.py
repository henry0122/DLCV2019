'''
Reference from:

@article{Gen2Adapt,
    author    = {Swami Sankaranarayanan and
           Yogesh Balaji and
           Carlos D. Castillo and
           Rama Chellappa},
    title     = {Generate To Adapt: Aligning Domains using Generative Adversarial Networks},
    journal   = {CoRR},
    volume    = {abs/1704.01705},
    year      = {2017},
    url       = {http://arxiv.org/abs/1704.01705},
}

'''

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
from torch.autograd import Variable, Function
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
'''  setup random seed and device '''
np.random.seed(87)
random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed(87)
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

''' --------------------------------------- Start of Model ------------------------------------------ '''
class Generator(nn.Module):
    def __init__(self, nclasses):
        super(Generator, self).__init__()
        
        self.ndim = 2*64
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512+self.ndim+nclasses+1, 64*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            # [128, 512, 2, 2] 

            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            # [128, 256, 4, 4]

            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
             # [128, 128, 8, 8]

            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
             # [128, 64, 16, 16]

            nn.ConvTranspose2d(64, 3, 4, 2, 3, bias=False),
            # [128, 3, 28, 28]
            nn.Tanh()
        )

    def forward(self, x):   
        batchSize = x.shape[0]
        x = x.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, 512, 1, 1).normal_(0, 1).to(device) 
        output = self.main(torch.cat((x, noise),1)) # [128, 3, 32, 32]
        return output
    
class svhnGenerator(nn.Module):
    def __init__(self, nclasses):
        super(svhnGenerator, self).__init__()
        
        self.ndim = 4*64
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512+self.ndim+nclasses+1, 64*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            # [128, 512, 2, 2] 

            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            # [128, 256, 4, 4]

            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
             # [128, 128, 8, 8]

            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
             # [128, 64, 16, 16]

            nn.ConvTranspose2d(64, 3, 4, 2, 3, bias=False),
            # [128, 3, 28, 28]
            nn.Tanh()
        )

    def forward(self, x):   
        batchSize = x.shape[0]
        x = x.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, 512, 1, 1).normal_(0, 1).to(device) 
        output = self.main(torch.cat((x, noise),1)) # [128, 3, 32, 32]
        return output

class Discriminator(nn.Module):
    def __init__(self, nclasses):
        super(Discriminator, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),            
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            # [128, 64, 14, 14] 

            nn.Conv2d(64, 64*2, 3, 1, 1),         
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            # [128, 128, 7, 7]
            

            nn.Conv2d(64*2, 64*4, 3, 1, 1),           
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            # [128, 256, 3, 3] 
            
            nn.Conv2d(64*4, 64*2, 3, 1, 1),           
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(3,3)
            # [128, 128, 1, 1] 
        )

        ### whether is real image
        self.classifier_s = nn.Sequential(
                                    nn.Linear(64*2, 1), 
                                    nn.Sigmoid()) 
        ### which class
        self.classifier_c = nn.Sequential(
                                    nn.Linear(64*2, 10), 
                                    nn.LogSoftmax()) 

    def forward(self, x):
        output = self.feature(x)
        output_s = self.classifier_s(output.view(-1, 64*2))
        output_s = output_s.view(-1)
        output_c = self.classifier_c(output.view(-1, 64*2))
        return output_s, output_c

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
                    
            nn.Conv2d(64, 64*2, 4, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.feature(x) # [128, 128, 2, 2]
        return output.view(-1, 2*64)
    
class svhnFeature(nn.Module):
    def __init__(self):
        super(svhnFeature, self).__init__()
        
#         resnet18 = tvmodels.resnet18(pretrained=True)
#         self.resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))
        
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # [64, 64, 13, 13]
            
            nn.Conv2d(64, 64*2, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # [64, 128, 5, 5]
            
            nn.Conv2d(64*2, 64*4, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # [64, 256, 1, 1]
                    
#             nn.Conv2d(64*4, 64*4, 3, 1, 0),
#             nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.feature(x) # [128, 128, 2, 2]
#         pdb.set_trace()
        return output.view(-1, 4*64)

class Classifier(nn.Module):
    def __init__(self, nclasses):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(          
            nn.Linear(2*64, 2*64),
            nn.ReLU(inplace=True),
            nn.Linear(2*64, nclasses),
        )

    def forward(self, x):       
        output = self.main(x)
        return output
    
class svhnClassifier(nn.Module):
    def __init__(self, nclasses):
        super(svhnClassifier, self).__init__()
        self.main = nn.Sequential(          
            nn.Linear(4*64, 2*64),
            nn.ReLU(inplace=True),
            nn.Linear(2*64, nclasses),
        )

    def forward(self, x):       
        output = self.main(x)
        return output

''' --------------------------------------- End of Model ------------------------------------------ '''




def train(args):
    if args.tar_dom_name == 'mnistm':
        source_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'train', 'src'), 
                                                                     batch_size=args.train_batch, shuffle=True, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'train', 'src'), 
                                                                          batch_size=args.train_batch, shuffle=True, num_workers=8)
        test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                     batch_size=args.test_batch, num_workers=8) 
        
        ''' Define the model '''
        generator = Generator(10).to(device)
        discriminator = Discriminator(10).to(device)
        feature_extractor = Feature().to(device)
        classifier = Classifier(10).to(device)
        
        ''' Defining initialization '''
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        feature_extractor.apply(weights_init)
        classifier.apply(weights_init)

        ''' Defining loss criterions '''
        criterion_c = nn.CrossEntropyLoss().to(device)
        criterion_s = nn.BCELoss().to(device)

        ''' Defining optimizers '''
        optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerF = optim.Adam(feature_extractor.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerC = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.8, 0.999))
        
#         real_label_val = 1
#         fake_label_val = 0
        ''' Start training '''
        curr_iter = 0
        best_acc = 0
        
        
        
        for epoch in range(args.epoch):
            
            generator.train()    
            discriminator.train()    
            feature_extractor.train()    
            classifier.train()    
        
            for i, (source, target) in enumerate(zip(source_dataloader, target_dataloader)):
                src_inputs, src_labels, src_files = source
                tar_inputs, tar_labels, tar_files = target       
                src_inputs_unnorm = (((src_inputs * 0.5) + 0.5) - 0.5)*2
                
                reallabel = torch.full((src_inputs.shape[0],), 1, device=device)
                fakelabel = torch.full((src_inputs.shape[0],), 0, device=device)
                

                # Creating one hot vector
                ### 10 + 1 : 10 for classes, 1 for the target label
                src_labels_onehot = np.zeros((src_inputs.shape[0], 10+1), dtype=np.float32)
                for p in range(src_inputs.shape[0]):
                    src_labels_onehot[p, src_labels[p]] = 1
                src_labels_onehot = torch.from_numpy(src_labels_onehot)
                
                tar_labels_onehot = np.zeros((tar_inputs.shape[0], 10+1), dtype=np.float32)
                for p in range(tar_inputs.shape[0]):
                    tar_labels_onehot[p, 10] = 1
                tar_labels_onehot = torch.from_numpy(tar_labels_onehot)
                
                src_inputs, src_labels = src_inputs.to(device), src_labels.to(device)
                src_inputs_unnorm = src_inputs_unnorm.to(device)
                tar_inputs = tar_inputs.to(device)
                src_labels_onehot = src_labels_onehot.to(device)
                tar_labels_onehot = tar_labels_onehot.to(device)
                
                # Updating Discriminator
                discriminator.zero_grad()
                src_emb = feature_extractor(src_inputs)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = generator(src_emb_cat)
                
                tar_emb = feature_extractor(tar_inputs)
                tar_emb_cat = torch.cat((tar_labels_onehot, tar_emb),1)
                tar_gen = generator(tar_emb_cat)

                src_realoutputD_s, src_realoutputD_c = discriminator(src_inputs_unnorm)   
                errD_src_real_s = criterion_s(src_realoutputD_s, reallabel) ### whether is real image 
                errD_src_real_c = criterion_c(src_realoutputD_c, src_labels) ### and the result of classification

                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen)
                errD_src_fake_s = criterion_s(src_fakeoutputD_s, fakelabel)

                tar_fakeoutputD_s, tar_fakeoutputD_c = discriminator(tar_gen)
                fakelabel = torch.full((tar_inputs.shape[0],), 0, device=device)
                errD_tar_fake_s = criterion_s(tar_fakeoutputD_s, fakelabel)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tar_fake_s
                errD.backward(retain_graph=True)    
                optimizerD.step()
                
                # Updating Generator
                generator.zero_grad()
                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen)
                errG_s = criterion_s(src_fakeoutputD_s, reallabel)
                errG_c = criterion_c(src_fakeoutputD_c, src_labels)
                errG = errG_s + errG_c
                errG.backward(retain_graph=True)
                optimizerG.step()
                
                # Updating Classifier
                classifier.zero_grad()
                outC = classifier(src_emb)   
                errC = criterion_c(outC, src_labels)
                errC.backward(retain_graph=True)    
                optimizerC.step()

                # Updating Feature extractor
                feature_extractor.zero_grad()
                errF_fromC = criterion_c(outC, src_labels)        

                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen) ## src_gen is generated from feature extractor
                errF_src_fromD = criterion_c(src_fakeoutputD_c, src_labels) * 0.1

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = discriminator(tar_gen) ## tar_gen is generated from feature extractor
                reallabel = torch.full((tar_inputs.shape[0],), 1, device=device)
                errF_tar_fromD = criterion_s(tgt_fakeoutputD_s, reallabel) * (0.1 * 0.3)
                
                errF = errF_fromC + errF_src_fromD + errF_tar_fromD
                errF.backward()
                optimizerF.step()        
                
                curr_iter += 1
                
            ################################ Evaluate ################################
            feature_extractor.eval()
            classifier.eval()
            total = 0
            correct = 0

            for i, test in enumerate(test_dataloader):
                inputs, labels, files = test
                inputs, labels = inputs.to(device), labels.to(device)

                outC = classifier(feature_extractor(inputs))
                pred = outC.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
                total += labels.shape[0]

            acc = 100 * (float(correct) / total)
            print('Epoch: {} | Test Accuracy: {:.0f}%'.format(epoch, acc))
            if acc > best_acc:
                best_acc = acc
                pathF = os.path.join(args.model_dir, 'best_s_m_feature.pth.tar')
                pathC = os.path.join(args.model_dir, 'best_s_m_classifier.pth.tar')
                save_model(feature_extractor, pathF)
                save_model(classifier, pathC)
        
        
    elif args.tar_dom_name == 'svhn':
        source_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'train', 'src'), 
                                                                     batch_size=args.train_batch, shuffle=True, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'train', 'src'), 
                                                                          batch_size=args.train_batch, shuffle=True, num_workers=8)
        test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                     batch_size=args.test_batch, num_workers=8) 
        
        ''' Define the model '''
        generator = Generator(10).to(device)
        discriminator = Discriminator(10).to(device)
        feature_extractor = Feature().to(device)
        classifier = Classifier(10).to(device)
        
        ''' Defining initialization '''
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        feature_extractor.apply(weights_init)
        classifier.apply(weights_init)

        ''' Defining loss criterions '''
        criterion_c = nn.CrossEntropyLoss().to(device)
        criterion_s = nn.BCELoss().to(device)

        ''' Defining optimizers '''
        optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerF = optim.Adam(feature_extractor.parameters(), lr=args.lr, betas=(0.8, 0.999))
        optimizerC = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.8, 0.999))
        
        ''' Start training '''
        curr_iter = 0
        best_acc = 0
        
        for epoch in range(args.epoch):
            
            generator.train()    
            discriminator.train()    
            feature_extractor.train()    
            classifier.train()    
        
            for i, (source, target) in enumerate(zip(source_dataloader, target_dataloader)):
                src_inputs, src_labels, src_files = source
                tar_inputs, tar_labels, tar_files = target       
                src_inputs_unnorm = (((src_inputs * 0.5) + 0.5) - 0.5)*2
                
                reallabel = torch.full((src_inputs.shape[0],), 1, device=device)
                fakelabel = torch.full((src_inputs.shape[0],), 0, device=device)
                

                # Creating one hot vector
                ### 10 + 1 : 10 for classes, 1 for the target label
                src_labels_onehot = np.zeros((src_inputs.shape[0], 10+1), dtype=np.float32)
                for p in range(src_inputs.shape[0]):
                    src_labels_onehot[p, src_labels[p]] = 1
                src_labels_onehot = torch.from_numpy(src_labels_onehot)
                
                tar_labels_onehot = np.zeros((tar_inputs.shape[0], 10+1), dtype=np.float32)
                for p in range(tar_inputs.shape[0]):
                    tar_labels_onehot[p, 10] = 1
                tar_labels_onehot = torch.from_numpy(tar_labels_onehot)
                
                src_inputs, src_labels = src_inputs.to(device), src_labels.to(device)
                src_inputs_unnorm = src_inputs_unnorm.to(device)
                tar_inputs = tar_inputs.to(device)
                src_labels_onehot = src_labels_onehot.to(device)
                tar_labels_onehot = tar_labels_onehot.to(device)
                
                # Updating Discriminator
                discriminator.zero_grad()
                src_emb = feature_extractor(src_inputs)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = generator(src_emb_cat)
                
                tar_emb = feature_extractor(tar_inputs)
                tar_emb_cat = torch.cat((tar_labels_onehot, tar_emb),1)
                tar_gen = generator(tar_emb_cat)

                src_realoutputD_s, src_realoutputD_c = discriminator(src_inputs_unnorm)   
                errD_src_real_s = criterion_s(src_realoutputD_s, reallabel) ### whether is real image 
                errD_src_real_c = criterion_c(src_realoutputD_c, src_labels) ### and the result of classification

                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen)
                errD_src_fake_s = criterion_s(src_fakeoutputD_s, fakelabel)

                tar_fakeoutputD_s, tar_fakeoutputD_c = discriminator(tar_gen)
                fakelabel = torch.full((tar_inputs.shape[0],), 0, device=device)
                errD_tar_fake_s = criterion_s(tar_fakeoutputD_s, fakelabel)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tar_fake_s
                errD.backward(retain_graph=True)    
                optimizerD.step()
                
                # Updating Generator
                generator.zero_grad()
                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen)
                errG_s = criterion_s(src_fakeoutputD_s, reallabel)
                errG_c = criterion_c(src_fakeoutputD_c, src_labels)
                errG = errG_s + errG_c
                errG.backward(retain_graph=True)
                optimizerG.step()
                
                # Updating Classifier
                classifier.zero_grad()
                outC = classifier(src_emb)   
                errC = criterion_c(outC, src_labels)
                errC.backward(retain_graph=True)    
                optimizerC.step()

                # Updating Feature extractor
                feature_extractor.zero_grad()
                errF_fromC = criterion_c(outC, src_labels)        

                src_fakeoutputD_s, src_fakeoutputD_c = discriminator(src_gen) ## src_gen is generated from feature extractor
                errF_src_fromD = criterion_c(src_fakeoutputD_c, src_labels) * 0.1

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = discriminator(tar_gen) ## tar_gen is generated from feature extractor
                reallabel = torch.full((tar_inputs.shape[0],), 1, device=device)
                errF_tar_fromD = criterion_s(tgt_fakeoutputD_s, reallabel) * (0.1 * 0.3)
                
                errF = errF_fromC + errF_src_fromD + errF_tar_fromD
                errF.backward()
                optimizerF.step()        
                
                curr_iter += 1
                
            ################################ Evaluate ################################
            feature_extractor.eval()
            classifier.eval()
            total = 0
            correct = 0

            for i, test in enumerate(test_dataloader):
                inputs, labels, files = test
                inputs, labels = inputs.to(device), labels.to(device)

                outC = classifier(feature_extractor(inputs))
                pred = outC.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
                total += labels.shape[0]

            acc = 100 * (float(correct) / total)
            print('Epoch: {} | Test Accuracy: {:.0f}%'.format(epoch, acc))
            if acc > best_acc:
                best_acc = acc
                pathF = os.path.join(args.model_dir, 'best_m_s_feature.pth.tar')
                pathC = os.path.join(args.model_dir, 'best_m_s_classifier.pth.tar')
                save_model(feature_extractor, pathF)
                save_model(classifier, pathC)


                
def test(args):
    if args.tar_dom_name == 'svhn':
        feature_extractor = Feature().to(device)
        classifier = Classifier(10).to(device)
#         model_path = os.path.join(args.model_dir, 'best_m_s_feature.pth.tar')
        feature_extractor.load_state_dict(torch.load('best_m_s_feature.pth.tar'))
#         model_path = os.path.join(args.model_dir, 'best_m_s_classifier.pth.tar')
        classifier.load_state_dict(torch.load('best_m_s_classifier.pth.tar'))
        test_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'final', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        correct = 0
        output = open(args.save_test_improved_dann, 'a')
        output.write('image_name,label\n')
        
        with torch.no_grad():
            for i, test in enumerate(test_dataloader):
                inputs, files = test
                inputs = inputs.to(device)

                outC = classifier(feature_extractor(inputs))
                pred = outC.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                 correct += pred.eq(labels.view_as(pred)).sum().item()

                for p in range(len(files)):
                    output.write('{},{}\n'.format(files[p], int(pred[p])))
                    
        output.close()
    
    elif args.tar_dom_name == 'mnistm':
        feature_extractor = Feature().to(device)
        classifier = Classifier(10).to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_feature.pth.tar')
        feature_extractor.load_state_dict(torch.load('best_s_m_feature.pth.tar'))
#         model_path = os.path.join(args.model_dir, 'best_s_m_classifier.pth.tar')
        classifier.load_state_dict(torch.load('best_s_m_classifier.pth.tar'))
        test_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'final', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        correct = 0
        output = open(args.save_test_improved_dann, 'a')
        output.write('image_name,label\n')
        
        with torch.no_grad():
            for i, test in enumerate(test_dataloader):
                inputs, files = test
                inputs = inputs.to(device)

                outC = classifier(feature_extractor(inputs))
                pred = outC.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                for p in range(len(files)):
                    output.write('{},{}\n'.format(files[p], int(pred[p])))
                    
        output.close()


if __name__=='__main__':
    args = parser.arg_parse()

    if not args.test_improved_dann:
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
        
        
        