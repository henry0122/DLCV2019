import os
import pdb
import numpy as np
from PIL import Image

### Train_data
train_img_path = 'hw2_data/train/img/'
train_seg_path = 'hw2_data/train/seg/'
for i in range(5460):
    tmp = '0000' + str(i)
    cur_img_path = os.path.join(train_img_path, tmp[-4:]+'.png')
    cur_seg_path = os.path.join(train_seg_path, tmp[-4:]+'.png')
    cur_img = Image.open(cur_img_path).convert('RGB')
    cur_seg = Image.open(cur_seg_path)
    cur_img = np.array(cur_img) # (352, 448, 3) 
    cur_seg = np.array(cur_seg) # (352, 448) 
    
    ## flip 'cur_img', 'cur_seg' horizontally 
    flip_img = np.flip(cur_img, 1)
    flip_seg = np.flip(cur_seg, 1)
    
    ## Save 
    flip_path = '0000' + str(i)
    fliped_img_path = os.path.join(train_img_path, 'flip_'+flip_path[-4:]+'.png')
    fliped_seg_path = os.path.join(train_seg_path, 'flip_'+flip_path[-4:]+'.png')
        
    flip_img = Image.fromarray(np.uint8(flip_img))
    flip_img.save(fliped_img_path)
    flip_seg = Image.fromarray(np.uint8(flip_seg))
    flip_seg.save(fliped_seg_path)
    
### Val_data
val_img_path = 'hw2_data/val/img/'
val_seg_path = 'hw2_data/val/seg/'
for i in range(500):
    tmp = '0000' + str(i)
    cur_img_path = os.path.join(val_img_path, tmp[-4:]+'.png')
    cur_seg_path = os.path.join(val_seg_path, tmp[-4:]+'.png')
    cur_img = Image.open(cur_img_path).convert('RGB')
    cur_seg = Image.open(cur_seg_path)
    cur_img = np.array(cur_img) # (352, 448, 3) 
    cur_seg = np.array(cur_seg) # (352, 448) 
    
    ## flip 'cur_img', 'cur_seg' horizontally 
    flip_img = np.flip(cur_img, 1)
    flip_seg = np.flip(cur_seg, 1)
    
    ## Save 
    flip_path = '0000' + str(i)
    fliped_img_path = os.path.join(val_img_path, 'flip_'+flip_path[-4:]+'.png')
    fliped_seg_path = os.path.join(val_seg_path, 'flip_'+flip_path[-4:]+'.png')
        
    flip_img = Image.fromarray(np.uint8(flip_img))
    flip_img.save(fliped_img_path)
    flip_seg = Image.fromarray(np.uint8(flip_seg))
    flip_seg.save(fliped_seg_path)
