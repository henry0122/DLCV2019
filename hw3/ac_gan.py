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
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
'''  setup random seed and device '''
np.random.seed(99)
random.seed(99)
torch.manual_seed(456) 
torch.cuda.manual_seed(456) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)   
    
''' --------------------------------------- Start of Dataset -------------------------------------------'''
class CelebADataset(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.acgan_train_data_dir
        self.data = self.loop_all_file(self.data_dir) # Return a list of tupple(img_path, label_path)

        ''' set up image transform '''
        self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        
        ''' Need to load label '''
        self.train_label = pd.read_csv(args.acgan_train_csv) 
        self.smile_label = torch.tensor(np.squeeze(self.train_label[['Smiling']].to_numpy()), dtype=torch.int64)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.data[idx]
        img_label = self.smile_label[idx]
        img = Image.open(img_path).convert('RGB')
        
        ## after transfoming -> [3, 64, 64]
        return self.transform(img), img_label, img_path 
    
    def loop_all_file(self, img_path):

        files = glob.glob( os.path.join(img_path, '*.png') )
        files = sorted(files)
            
        return files
    
''' --------------------------------------- End of Dataset ------------------------------------------''' 
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
''' --------------------------------------- Start of Generator ------------------------------------------''' 
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.smile_emb = nn.Embedding(2, 100)
        # self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)
#         self.hair_emb = nn.Embedding(6, 40)
#         self.eye_emb = nn.Embedding(4, 26)
#         self.face_emb = nn.Embedding(3, 21)
#         self.glasses_emb = nn.Embedding(2, 13)
        

#         self.init_size = args.img_size // 8  # Initial size before upsampling
        self.init_size = 64 // 8  # Initial size before upsampling

        # Deeper network
        # self.init_size = opt.img_size // 32  # Initial size before upsampling

        self.l1 = nn.Sequential(nn.Linear(100 + 1, 64 * 8 * self.init_size ** 2)) ### 64 * 8 is random set

        self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(64 * 8),
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(64 * 8, 64 * 4, 3, stride=1, padding=1)),
                
                nn.BatchNorm2d(64 * 4, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(64 * 4, 64 * 2, 3, stride=1, padding=1)),
                
                nn.BatchNorm2d(64 * 2, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(64 * 2, 64, 3, stride=1, padding=1)),
                
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 3, 3, stride=1, padding=1)), ## first of '3' represents channel
                
                nn.Tanh(),
            )

    def forward(self, noise, smile):
        ''' noise: [batch, 100], smile: [batch]'''
        smile_emb = self.smile_emb(smile) # [128, 100] ####????????????????????????????????????????????

        # gen_input = torch.mul(self.label_emb(labels), noise)
#         gen_input = torch.mul(final_emb, noise)
#         gen_input = torch.cat((final_emb, noise), dim=1)
        
        gen_input = torch.cat((smile.unsqueeze(1), noise), dim=1).type(torch.cuda.FloatTensor)
        out = self.l1(gen_input) # [128, 43392]
        out = out.view(out.shape[0], 64 * 8, self.init_size, self.init_size) # [128, 678, 8, 8]
        img = self.conv_blocks(out) # [128, 3, 64, 64]
        return img
''' --------------------------------------- End of Generator ------------------------------------------''' 

''' --------------------------------------- Start of Discriminator ---------------------------------------''' 
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
#             block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
#                      nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 64, bn=False), ## 3 is the channel of image
            *discriminator_block(64, 64 * 2),
            *discriminator_block(64 * 2, 64 * 4),
            *discriminator_block(64 * 4, 64 * 8),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4

        # Output layers
        # one hot for concat : opt.latent_dim -> opt.n_classes
        self.adv_layer = nn.Sequential(nn.Linear(64 * 8 * ds_size ** 2 + 1, 1), nn.Sigmoid())
        self.aux_smile_layer = nn.Sequential(nn.Linear(64 * 8 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, label):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        
        v_out = torch.cat((out, label.unsqueeze(1).type(torch.cuda.FloatTensor)), dim=1)
        validity = self.adv_layer(v_out) # [128, 1]
        
        label_smile = self.aux_smile_layer(out) # [128, 1]

        return validity.squeeze(), label_smile.squeeze()
    
''' --------------------------------------- End of Discriminator ---------------------------------------''' 
    
    
def train(args):
    dataloader = torch.utils.data.DataLoader(CelebADataset(args), batch_size=args.train_batch,
                                             shuffle=True) ## , num_workers=args.workers
    
#     real_batch, img_label, img_path = next(iter(dataloader))
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss().to(device)
    auxiliary_loss = torch.nn.BCELoss().to(device)
#     auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)

    # Initialize generator and discriminator
    generator = Generator(args).to(device)
    discriminator = Discriminator(args).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    fixed_noise = torch.randn(10, 100, device=device).type(torch.cuda.LongTensor)
    
    img_list = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.epoch):
        # run batches
        gloss = 0
        dloss = 0
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="AC-GAN")

        for i, (imgs, labels, paths) in trange:

            imgs = imgs.to(device)
            labels = labels.to(device)
            float_labels = labels.type(torch.cuda.FloatTensor)
            
            b_size = imgs.shape[0]

            # Adversarial ground truths
            true_label = torch.full((b_size,), 1.0, device=device)
            fake_label = torch.full((b_size,), 0.0, device=device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            noise = torch.randn(b_size, 100, device=device).type(torch.cuda.LongTensor)


            # Generate a batch of images
            gen_imgs = generator(noise, labels)

            # Loss measures generator's ability to fool the discriminator

            # use one hot directly
            validity, pred_smile = discriminator(gen_imgs, labels)

            auxi_loss = auxiliary_loss(pred_smile, float_labels)
            g_loss = (adversarial_loss(validity, true_label) + auxi_loss)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_smile = discriminator(imgs, labels.detach())
            auxi_loss = auxiliary_loss(real_smile, float_labels)
            d_real_loss = (adversarial_loss(real_pred, true_label) + auxi_loss) 

            # Loss for fake images
            fake_pred, fake_smile = discriminator(gen_imgs.detach(), labels.detach()) # no back-propagation so use detach
            auxi_loss = auxiliary_loss(fake_smile, float_labels)
            d_fake_loss = (adversarial_loss(fake_pred, fake_label) + auxi_loss)


            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            optimizer_D.step()



            gloss += (g_loss.item())
            dloss += (d_real_loss.item() + d_fake_loss.item()) /2 

            trange.set_postfix({"epoch":"{}".format(epoch),"g_loss":"{0:.5f}".format(gloss / (i + 1)), "d_loss":"{0:.5f}".format(dloss / (i + 1))})

            iters += 1
            
#         if (iters % 1000 == 0) or ((epoch == args.epoch-1) and (i == len(dataloader)-1)):
        ### End of one Epoch
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                smile_label = torch.full((10,), 1, device=device).type(torch.cuda.LongTensor)
                unsmile_label = torch.full((10,), 0, device=device).type(torch.cuda.LongTensor)
                show_image = []
                for f in range(fixed_noise.shape[0]):
                    fake_smile_img = generator(fixed_noise[f].unsqueeze(0), smile_label[f].unsqueeze(0)).detach().cpu()
                    fake_unsmile_img = generator(fixed_noise[f].unsqueeze(0), unsmile_label[f].unsqueeze(0)).detach().cpu()
                    show_image.append(fake_smile_img)
                    show_image.append(fake_unsmile_img)
                show_image = torch.stack(show_image).squeeze()
                show_image = vutils.make_grid(show_image, padding=2, normalize=True)
                plot(show_image, args.fig_dir, epoch)
                
            model_path = os.path.join(args.model_dir, 'p2_{}_generator.pth.tar'.format(str(epoch)))
            save_model(generator, model_path)

        
def test(args):
    test_generator = Generator(args).to(device)
#     model_path = os.path.join(args.model_dir, 'p2_499_generator.pth.tar')
    ''' Notice the model path'''
    test_generator.load_state_dict(torch.load('p2_499_generator.pth.tar'))
    fixed_noise = torch.randn(10, 100, device=device).type(torch.cuda.LongTensor)
    
    smile_label = torch.full((10,), 1, device=device).type(torch.cuda.LongTensor)
    unsmile_label = torch.full((10,), 0, device=device).type(torch.cuda.LongTensor)
    show_image = []
    for f in range(fixed_noise.shape[0]):
        fake_smile_img = test_generator(fixed_noise[f].unsqueeze(0), smile_label[f].unsqueeze(0)).detach().cpu()
        fake_unsmile_img = test_generator(fixed_noise[f].unsqueeze(0), unsmile_label[f].unsqueeze(0)).detach().cpu()
        show_image.append(fake_smile_img)
        show_image.append(fake_unsmile_img)
    show_image = torch.stack(show_image).squeeze()
    show_image = vutils.make_grid(show_image, padding=2, normalize=True)
#     plot(show_image, args.fig_dir, epoch)
    
    
    plt.axis("off")
    plt.title("10 pairs of Smile/Unsmile Images")
    plt.imshow( np.transpose(show_image,(1,2,0)) )
    save_path = os.path.join(args.save_test_acgan, 'fig2_2.jpg')
    plt.savefig(save_path)
    plt.close()


def plot(show_image, path, epoch):
    plt.axis("off")
    plt.title("Smile/Unsmile Images")
    plt.imshow( np.transpose(show_image,(1,2,0)) )
    
    save_path = os.path.join(path, 'acgan_{}_result.png'.format(str(epoch)))
    plt.savefig(save_path)
    plt.close()
    
if __name__=='__main__':
    
    args = parser.arg_parse()
    
    if not args.test_acgan:
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
    
    
    
    