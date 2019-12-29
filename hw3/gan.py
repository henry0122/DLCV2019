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
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
'''  setup random seed and device '''
np.random.seed(42)
random.seed(42)
torch.manual_seed(456)
torch.cuda.manual_seed(456)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)   
    
''' --------------------------------------- Start of Dataset -------------------------------------------'''
class CelebADataset(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.gan_train_data_dir
        self.data = self.loop_all_file(self.data_dir) # Return a list of tupple(img_path, label_path)

        ''' set up image transform '''
        self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        
        ## after transfoming -> [3, 64, 64]
        return self.transform(img), img_path 
    
    def loop_all_file(self, img_path):
#         all_file_path = []
#         assert len(os.listdir(img_path))/2 == 5460
#         for i in range(int(len(os.listdir(img_path))/2)):
#             tmp = '0000' + str(i)
#             cur_img_path = os.path.join(img_path, tmp[-4:]+'.png')
#             cur_label_path = os.path.join(label_path, tmp[-4:]+'.png')
#             flip_img_path = os.path.join(img_path, 'flip_' + tmp[-4:]+'.png')
#             flip_label_path = os.path.join(label_path, 'flip_' + tmp[-4:]+'.png')
#             all_file_path.append((cur_img_path, cur_label_path))
#             all_file_path.append((flip_img_path, flip_label_path))
                
#         elif mode == 'test':
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
    def __init__(self):
        super(Generator, self).__init__()
#         self.ngpu = ngpu
        self.generate = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            spectral_norm(nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            spectral_norm(nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            spectral_norm(nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.generate(input)

''' --------------------------------------- End of Generator ------------------------------------------''' 

''' --------------------------------------- Start of Discriminator ---------------------------------------''' 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
#         self.ngpu = ngpu
        self.discriminate = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.discriminate(input)
    
''' --------------------------------------- End of Discriminator ---------------------------------------''' 
    
def train(args):
    
    dataset = dset.ImageFolder(root='hw3_data/face',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch,
                                             shuffle=True, num_workers=args.workers)
#     dataloader = torch.utils.data.DataLoader(CelebADataset(args), batch_size=args.train_batch,
#                                              shuffle=True, num_workers=args.workers)
    
#     real_batch, target = next(iter(dataloader))


    ''' Create Model '''
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # Apply the weights_init function to randomly initialize all weights to mean = 0, stdev = 0.2
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    ''' Loss function, Optimizer and some Parameters '''
    criterion = nn.BCELoss().to(device)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.epoch):
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="Problem 1")
        for i, (img, lable) in trange:

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()

            img_tensor = img.to(device)
            b_size = img_tensor.shape[0]
            label = torch.full((b_size,), real_label, device=device)

            output = discriminator(img_tensor).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            label.fill_(fake_label)
            
            fake_img = generator(noise)
            ### detached from the current graph, and not require gradient
            output = discriminator(fake_img.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_img).view(-1)
            errG = criterion(output, label)
            errG.backward()

            optimizerG.step()


            trange.set_postfix({"epoch":"{}".format(epoch),"g_loss":"{0:.5f}".format(errG.item()), "d_loss":"{0:.5f}".format(errD.item())})
            # Output training stats
#             if i % 50 == 0:
#                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                       % (epoch, args.epoch, i, len(dataloader),
#                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == args.epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_img = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_img[:32], padding=2, normalize=True))

            iters += 1
            
        if (epoch + 1) % 5 == 0:
            compare_Real_Fake(args.fig_dir, next(iter(dataloader)), img_list[-1], epoch)
            
    ''' Plot loss and result '''
    plot(args.fig_dir, G_losses, D_losses)
    compare_Real_Fake(args.fig_dir, next(iter(dataloader)), img_list[-1], epoch)
    model_path = os.path.join(args.model_dir, 'p1_generator.pth.tar')
    save_model(generator, model_path)
    
    
def test(args):
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    
    test_generator = Generator().to(device)
#     model_path = os.path.join(args.model_dir, 'p1_generator.pth.tar') 
    ''' Notice the model path'''
    test_generator.load_state_dict(torch.load('p1_generator.pth.tar'))
    fake_img = test_generator(fixed_noise).detach().cpu()
    
    plot_img = vutils.make_grid(fake_img[:32], padding=2, normalize=True)
    plt.figure(figsize=(7,7))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow( np.transpose(plot_img,(1,2,0)) )
    save_path = os.path.join(args.save_test_gan, 'fig1_2.jpg')
    plt.savefig(save_path)
    plt.close()
            
def plot(path, G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="Generator")
    plt.plot(D_losses,label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    save_path = os.path.join(path, 'gan_loss.png')
    plt.savefig(save_path)
    plt.close()
    
def compare_Real_Fake(path, real_batch, fake_img, epoch):

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow( np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),(1,2,0)) )

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow( np.transpose(fake_img,(1,2,0)) )
    
    save_path = os.path.join(path, 'gan_{}_result.png'.format(str(epoch)))
    plt.savefig(save_path)
    plt.close()
    
    
if __name__=='__main__':
    
    args = parser.arg_parse()
        
    if not args.test_gan:
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
    
        
    
    
    