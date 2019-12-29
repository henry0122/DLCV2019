import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

import pdb

class BaselineNet(nn.Module):

    def __init__(self, args):
        super(BaselineNet, self).__init__()  

        ''' declare layers used in this network'''
        resnet18 = tvmodels.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))
        
        # first block
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        
        # second block
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        
        # third block
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        
        # forth block
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        
        # fifth block
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        
        self.conv1 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):

        img = self.resnet18(img) # resnet50 = [32, 2048, 11, 14], resnet18 = [32, 512, 11, 14] 
        
        x = self.relu1(self.up1(img))
        x = self.relu2(self.up2(x))
        x = self.relu3(self.up3(x))
        x = self.relu4(self.up4(x))
        x = self.relu5(self.up5(x))
        
        out = self.conv1(x)
        
        results = F.softmax(out, dim=1)
        _, results  = torch.max(results, dim=1)

        return out, results

    
########################################## Improved ################################################
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pre_trained, range_m, dropout=False):
        super(EncoderBlock, self).__init__()
        self.encode = pre_trained
        if range_m:
            self.encode = nn.Sequential(*pre_trained)

    def forward(self, x):
        return self.encode(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=2, stride=2)
        
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, x, residual_one):
        # up first, crop then cat, finally decode
        uped = self.up(x)
        
        diffY = residual_one.size()[2] - uped.size()[2]
        diffX = residual_one.size()[3] - uped.size()[3]
        uped = F.pad(uped, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        out = torch.cat([residual_one, uped], 1)

        return self.decode(out)


######### U-net architecture
class ImprovedNet(nn.Module): 
    def __init__(self, num_classes):
        super(ImprovedNet, self).__init__()
        
        self.base_model = tvmodels.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.enc1 = EncoderBlock(3, 64, self.base_layers[:3], True)
        self.enc2 = EncoderBlock(64, 128, self.base_layers[3:5], True)
        self.enc3 = EncoderBlock(128, 256, self.base_layers[5], False)
        self.enc4 = EncoderBlock(256, 512, self.base_layers[6], False, dropout=True)
        self.center = EncoderBlock(512, 1024, self.base_layers[7], False, dropout=True)
        
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        self.dec4 = DecoderBlock(1024, 512, 256)
        self.dec3 = DecoderBlock(512, 256, 128)
        self.dec2 = DecoderBlock(256, 128, 64)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        self.final = nn.Conv2d(64, 9, kernel_size=1) # num_classes = 9
    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.center(enc4)
        
        center = self.layer4_1x1(enc5)

        dec4 = self.dec4(center, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        final = self.final(dec1) # final = [batch, 9, 164, 260] 
        out = F.upsample(final, x.size()[2:], mode='bilinear') # [batch, 9, 352, 448] 
        
        result = F.softmax(out, dim=1)
        _, result  = torch.max(result, dim=1)

        return out, result

############################################ Best #################################################
    
#     def __init__(self, n_class):
#         super(ImprovedNet, self).__init__()

#         self.base_model = tvmodels.resnet18(pretrained=True)
#         self.base_layers = list(self.base_model.children())

#         self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convrelu(64, 64, 1, 0)
#         self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convrelu(128, 128, 1, 0)
#         self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convrelu(256, 256, 1, 0)
#         self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convrelu(512, 512, 1, 0)
# #######################################################################
# #         self.layer4_1x1 = convrelu(512, 1024, 1, 0)
        
# # #         self.transconv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, padding=0, stride=2)
# # #         self.transconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2)
# # #         self.transconv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2)
# # #         self.transconv_0 = convrelu(64 + 256, 128, 3, 1)

# #         self.centerconv = convrelu(1024, 1024, 1, 0)
    
# #         self.transup_0 = nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=0, stride=2)
        
# #         self.conv_3 = convrelu(256 + 512, 512, 3, 1)
# #         self.conv_2 = convrelu(128 + 512, 256, 3, 1)
# #         self.conv_1 = convrelu(64 + 256, 256, 3, 1)
# #         self.conv_0 = convrelu(64 + 256, 128, 3, 1)
    
# ######################################################################
    
    

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
#         self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
#         self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
#         self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

#         self.conv_last = nn.Conv2d(64, 9, 1)

#     def forward(self, input):
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)

#         layer0 = self.layer0(input) # 64
#         layer1 = self.layer1(layer0) # 64
#         layer2 = self.layer2(layer1) # 128
#         layer3 = self.layer3(layer2) # 256
#         layer4 = self.layer4(layer3) # 512 
#         layer4 = self.layer4_1x1(layer4) 

#         #########################################################
# #         center = self.layer4_1x1(layer4) # 1024
# #         ceneter = self.centerconv(center) # 1024
        
# #         import pdb
# #         pdb.set_trace()
        
# #         dec0 = self.transup_0(ceneter)
# #         x = torch.cat([x, layer3], dim=1)
# #         x = self.conv_3(x)
        
#         #########################################################
        
        
#         x = self.upsample(layer4)
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)

#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)

#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)

#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)

        
        
        
#         x = self.upsample(x)
#         x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)

#         out = self.conv_last(x)
        
#         result = F.softmax(out, dim=1)
#         _, result  = torch.max(result, dim=1)

#         return out, result
    