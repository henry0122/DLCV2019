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

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold


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
    
''' --------------------------------------- End of Model ------------------------------------------ '''
    
    
def dann_tsne(args):
    if args.tar_dom_name == 'mnistm':
        test_dann = DANN().to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_adapt.pth.tar')
        test_dann.load_state_dict(torch.load('model_log/dann/best_s_m_adapt.pth.tar'))
        source_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
        

        src_features_emb = []
        src_labels = []
        src_domains = []

        tar_features_emb = []
        tar_labels = []
        tar_domains = []
        
        with torch.no_grad():
            for imgs, labels, imgs_path in source_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                src_features_emb.append(eval_img_embed.detach().cpu().numpy())
                src_labels.append(labels.detach().cpu().numpy())
                src_domains.append([0]*b_size)
                
            for imgs, labels, imgs_path in target_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                tar_features_emb.append(eval_img_embed.detach().cpu().numpy())
                tar_labels.append(labels.detach().cpu().numpy())
                tar_domains.append([1]*b_size)
                
        src_features_emb = np.concatenate(src_features_emb)
        src_labels = np.concatenate(src_labels)
        src_domains = np.concatenate(src_domains)

        tar_features_emb = np.concatenate(tar_features_emb)
        tar_labels = np.concatenate(tar_labels)
        tar_domains = np.concatenate(tar_domains)
             
            
        tsne_emb = []
        tsne_labels = []
        tsne_domains = []

        sample_size = 500

        for i in range(10):
            source_select_idxs = np.where(src_labels == i)[0]
            target_select_idxs = np.where(tar_labels == i)[0]

            print(source_select_idxs.shape, target_select_idxs.shape)

            if source_select_idxs.shape[0] > 1000:
                source_sample = random.sample(source_select_idxs.tolist(), sample_size)
            else:
                source_sample = source_select_idxs.tolist()

            if target_select_idxs.shape[0] > 1000:
                target_sample = random.sample(target_select_idxs.tolist(), sample_size)
            else:
                target_sample = target_select_idxs.tolist()

            tsne_emb.append(src_features_emb[source_sample])
            tsne_emb.append(tar_features_emb[target_sample])

            size = len(source_sample) + len(target_sample)
            tsne_labels.append([i]*size)

            tsne_domains.append([0]*len(source_sample))
            tsne_domains.append([1]*len(target_sample))
            
        tsne_emb = np.concatenate(tsne_emb)
        tsne_labels = np.concatenate(tsne_labels)
        tsne_domains = np.concatenate(tsne_domains)

        tsne = manifold.TSNE(n_components=2, verbose=1)
        tsne_embed = tsne.fit_transform(tsne_emb)
        
        plt.figure(figsize=(8, 6))
        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 1)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i))

        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 0)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i), label=i)
        plt.legend()

        plt.savefig('fig/dann/s_m_label.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for i in range(2):
            select_idxs = np.where(tsne_domains == i)[0]
            plt.scatter(x=tsne_embed[select_idxs, 0], y=tsne_embed[select_idxs, 1], c='C1' if i == 0 else 'C2', label='source' if i == 0 else 'target')
        plt.legend()
        plt.savefig('fig/dann/s_m_domain.png')
        plt.close()
        
            
    elif args.tar_dom_name == 'svhn':
        test_dann = DANN().to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_adapt.pth.tar')
        test_dann.load_state_dict(torch.load('model_log/dann/best_m_s_adapt.pth.tar'))
        source_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
        

        src_features_emb = []
        src_labels = []
        src_domains = []

        tar_features_emb = []
        tar_labels = []
        tar_domains = []
        
        with torch.no_grad():
            for imgs, labels, imgs_path in source_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                src_features_emb.append(eval_img_embed.detach().cpu().numpy())
                src_labels.append(labels.detach().cpu().numpy())
                src_domains.append([0]*b_size)
                
            for imgs, labels, imgs_path in target_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                class_output, domain_output, eval_img_embed = test_dann(imgs, 0)
                pred = class_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                tar_features_emb.append(eval_img_embed.detach().cpu().numpy())
                tar_labels.append(labels.detach().cpu().numpy())
                tar_domains.append([1]*b_size)
                
        src_features_emb = np.concatenate(src_features_emb)
        src_labels = np.concatenate(src_labels)
        src_domains = np.concatenate(src_domains)

        tar_features_emb = np.concatenate(tar_features_emb)
        tar_labels = np.concatenate(tar_labels)
        tar_domains = np.concatenate(tar_domains)
             
            
        tsne_emb = []
        tsne_labels = []
        tsne_domains = []

        sample_size = 500

        for i in range(10):
            source_select_idxs = np.where(src_labels == i)[0]
            target_select_idxs = np.where(tar_labels == i)[0]

            print(source_select_idxs.shape, target_select_idxs.shape)

            if source_select_idxs.shape[0] > 1000:
                source_sample = random.sample(source_select_idxs.tolist(), sample_size)
            else:
                source_sample = source_select_idxs.tolist()

            if target_select_idxs.shape[0] > 1000:
                target_sample = random.sample(target_select_idxs.tolist(), sample_size)
            else:
                target_sample = target_select_idxs.tolist()

            tsne_emb.append(src_features_emb[source_sample])
            tsne_emb.append(tar_features_emb[target_sample])

            size = len(source_sample) + len(target_sample)
            tsne_labels.append([i]*size)

            tsne_domains.append([0]*len(source_sample))
            tsne_domains.append([1]*len(target_sample))
            
        tsne_emb = np.concatenate(tsne_emb)
        tsne_labels = np.concatenate(tsne_labels)
        tsne_domains = np.concatenate(tsne_domains)

        tsne = manifold.TSNE(n_components=2, verbose=1)
        tsne_embed = tsne.fit_transform(tsne_emb)
        
        plt.figure(figsize=(8, 6))
        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 1)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i))

        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 0)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i), label=i)
        plt.legend()

        plt.savefig('fig/dann/m_s_label.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for i in range(2):
            select_idxs = np.where(tsne_domains == i)[0]
            plt.scatter(x=tsne_embed[select_idxs, 0], y=tsne_embed[select_idxs, 1], c='C1' if i == 0 else 'C2', label='source' if i == 0 else 'target')
        plt.legend()
        plt.savefig('fig/dann/m_s_domain.png')
        plt.close()
        
        
def improved_dann_tsne(args):
    if args.tar_dom_name == 'mnistm':
        feature_extractor = Feature().to(device)
#         classifier = Classifier(10).to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_feature.pth.tar')
        feature_extractor.load_state_dict(torch.load('model_log/improved_dann/best_s_m_feature.pth.tar'))
#         model_path = os.path.join(args.model_dir, 'best_s_m_classifier.pth.tar')
#         classifier.load_state_dict(torch.load('model_log/improved_dann/best_s_m_classifier.pth.tar'))
        
        source_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        
        

        src_features_emb = []
        src_labels = []
        src_domains = []

        tar_features_emb = []
        tar_labels = []
        tar_domains = []
        
        with torch.no_grad():
            for imgs, labels, imgs_path in source_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                eval_img_embed = feature_extractor(imgs)
#                 pred = outC.argmax(dim=1, keepdim=True)
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                src_features_emb.append(eval_img_embed.detach().cpu().numpy())
                src_labels.append(labels.detach().cpu().numpy())
                src_domains.append([0]*b_size)
                
            for imgs, labels, imgs_path in target_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                eval_img_embed = feature_extractor(imgs)
#                 pred = outC.argmax(dim=1, keepdim=True)
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                tar_features_emb.append(eval_img_embed.detach().cpu().numpy())
                tar_labels.append(labels.detach().cpu().numpy())
                tar_domains.append([1]*b_size)
                
        src_features_emb = np.concatenate(src_features_emb)
        src_labels = np.concatenate(src_labels)
        src_domains = np.concatenate(src_domains)

        tar_features_emb = np.concatenate(tar_features_emb)
        tar_labels = np.concatenate(tar_labels)
        tar_domains = np.concatenate(tar_domains)
             
            
        tsne_emb = []
        tsne_labels = []
        tsne_domains = []

        sample_size = 500

        for i in range(10):
            source_select_idxs = np.where(src_labels == i)[0]
            target_select_idxs = np.where(tar_labels == i)[0]

            print(source_select_idxs.shape, target_select_idxs.shape)

            if source_select_idxs.shape[0] > 1000:
                source_sample = random.sample(source_select_idxs.tolist(), sample_size)
            else:
                source_sample = source_select_idxs.tolist()

            if target_select_idxs.shape[0] > 1000:
                target_sample = random.sample(target_select_idxs.tolist(), sample_size)
            else:
                target_sample = target_select_idxs.tolist()

            tsne_emb.append(src_features_emb[source_sample])
            tsne_emb.append(tar_features_emb[target_sample])

            size = len(source_sample) + len(target_sample)
            tsne_labels.append([i]*size)

            tsne_domains.append([0]*len(source_sample))
            tsne_domains.append([1]*len(target_sample))
            
        tsne_emb = np.concatenate(tsne_emb)
        tsne_labels = np.concatenate(tsne_labels)
        tsne_domains = np.concatenate(tsne_domains)

        tsne = manifold.TSNE(n_components=2, verbose=1)
        tsne_embed = tsne.fit_transform(tsne_emb)
        
        plt.figure(figsize=(8, 6))
        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 1)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i))

        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 0)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i), label=i)
        plt.legend()

        plt.savefig('fig/improved_dann/s_m_label.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for i in range(2):
            select_idxs = np.where(tsne_domains == i)[0]
            plt.scatter(x=tsne_embed[select_idxs, 0], y=tsne_embed[select_idxs, 1], c='C1' if i == 0 else 'C2', label='source' if i == 0 else 'target')
        plt.legend()
        plt.savefig('fig/improved_dann/s_m_domain.png')
        plt.close()
        
    elif args.tar_dom_name == 'svhn':
        feature_extractor = Feature().to(device)
#         classifier = Classifier(10).to(device)
#         model_path = os.path.join(args.model_dir, 'best_s_m_feature.pth.tar')
        feature_extractor.load_state_dict(torch.load('model_log/improved_dann/best_m_s_feature.pth.tar'))
#         model_path = os.path.join(args.model_dir, 'best_s_m_classifier.pth.tar')
#         classifier.load_state_dict(torch.load('model_log/improved_dann/best_s_m_classifier.pth.tar'))
        
        source_dataloader = torch.utils.data.DataLoader(MnistmDataset(args, 'test', 'tar'), 
                                                                              batch_size=args.test_batch, num_workers=8)
        target_dataloader = torch.utils.data.DataLoader(SvhnDataset(args, 'test', 'tar'), 
                                                                          batch_size=args.test_batch, num_workers=8)
        
        

        src_features_emb = []
        src_labels = []
        src_domains = []

        tar_features_emb = []
        tar_labels = []
        tar_domains = []
        
        with torch.no_grad():
            for imgs, labels, imgs_path in source_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                eval_img_embed = feature_extractor(imgs)
#                 pred = outC.argmax(dim=1, keepdim=True)
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                src_features_emb.append(eval_img_embed.detach().cpu().numpy())
                src_labels.append(labels.detach().cpu().numpy())
                src_domains.append([0]*b_size)
                
            for imgs, labels, imgs_path in target_dataloader:
                imgs = imgs.to(device) # [batch, 3, 28, 28]
                eval_img_embed = feature_extractor(imgs)
#                 pred = outC.argmax(dim=1, keepdim=True)
                eval_img_embed = F.relu(eval_img_embed)
                b_size = eval_img_embed.shape[0]
                tar_features_emb.append(eval_img_embed.detach().cpu().numpy())
                tar_labels.append(labels.detach().cpu().numpy())
                tar_domains.append([1]*b_size)
                
        src_features_emb = np.concatenate(src_features_emb)
        src_labels = np.concatenate(src_labels)
        src_domains = np.concatenate(src_domains)

        tar_features_emb = np.concatenate(tar_features_emb)
        tar_labels = np.concatenate(tar_labels)
        tar_domains = np.concatenate(tar_domains)
             
            
        tsne_emb = []
        tsne_labels = []
        tsne_domains = []

        sample_size = 500

        for i in range(10):
            source_select_idxs = np.where(src_labels == i)[0]
            target_select_idxs = np.where(tar_labels == i)[0]

            print(source_select_idxs.shape, target_select_idxs.shape)

            if source_select_idxs.shape[0] > 1000:
                source_sample = random.sample(source_select_idxs.tolist(), sample_size)
            else:
                source_sample = source_select_idxs.tolist()

            if target_select_idxs.shape[0] > 1000:
                target_sample = random.sample(target_select_idxs.tolist(), sample_size)
            else:
                target_sample = target_select_idxs.tolist()

            tsne_emb.append(src_features_emb[source_sample])
            tsne_emb.append(tar_features_emb[target_sample])

            size = len(source_sample) + len(target_sample)
            tsne_labels.append([i]*size)

            tsne_domains.append([0]*len(source_sample))
            tsne_domains.append([1]*len(target_sample))
            
        tsne_emb = np.concatenate(tsne_emb)
        tsne_labels = np.concatenate(tsne_labels)
        tsne_domains = np.concatenate(tsne_domains)

        tsne = manifold.TSNE(n_components=2, verbose=1)
        tsne_embed = tsne.fit_transform(tsne_emb)
        
        plt.figure(figsize=(8, 6))
        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 1)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i))

        for i in range(10):
            select_idxs = np.where(tsne_labels == i)[0]
            select_domain_idxs = np.where(tsne_domains[select_idxs] == 0)[0]
            plt.scatter(x=tsne_embed[select_idxs[select_domain_idxs], 0], y=tsne_embed[select_idxs[select_domain_idxs], 1], c='C{}'.format(i), label=i)
        plt.legend()

        plt.savefig('fig/improved_dann/m_s_label.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for i in range(2):
            select_idxs = np.where(tsne_domains == i)[0]
            plt.scatter(x=tsne_embed[select_idxs, 0], y=tsne_embed[select_idxs, 1], c='C1' if i == 0 else 'C2', label='source' if i == 0 else 'target')
        plt.legend()
        plt.savefig('fig/improved_dann/m_s_domain.png')
        plt.close()
    
if __name__=='__main__':
    args = parser.arg_parse()
    
    if args.plot_dann:
        if not os.path.exists(args.fig_dir):
            os.makedirs(args.fig_dir)
        dann_tsne(args)
            
    elif args.plot_improved:
        if not os.path.exists(args.fig_dir):
            os.makedirs(args.fig_dir)
        improved_dann_tsne(args)