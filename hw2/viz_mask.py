import os
import argparse
import scipy.misc
import scipy.ndimage

import numpy as np 

from matplotlib import colors as mcolors

voc_cls = {'background':0, 
           'aeroplane': 2,# 1, 
           'bus':3, #6, 
           'car':8, #7, 
           'cat':7, # 8,
           'dog':6, #12, 
           'horse':5, #13, 
           'person':1, # 15, 
           'tv/monitor':4} # 20}
cls_color = {
    0:  [0, 0, 0],
    2:  [128, 0, 0],
    3:  [0, 128, 128],
    8:  [128, 128, 128],
    7:  [64, 0, 0],
    6: [64, 0, 128],
    5: [192, 0, 128],
    1: [192, 128, 128],
    4: [0, 64, 128]
}
def mask_edge_detection(mask, edge_width):
    h = mask.shape[0]
    w = mask.shape[1]

    edge_mask = np.zeros((h,w))
    
    for i in range(h):
        for j in range(1,w):
            j_prev = j - 1 
            # horizantal #
            if not mask[i][j] == mask[i][j_prev]: # horizontal
                if mask[i][j]==1: # 0 -> 1
                    edge_mask[i][j] = 1
                    for add in range(1,edge_width):
                        if j + add < w and mask[i][j+add] == 1:
                            edge_mask[i][j+add] = 1

                        
                else : # 1 -> 0
                    edge_mask[i][j_prev] = 1
                    for minus in range(1,edge_width):
                        if j_prev - minus >= 0 and mask[i][j_prev - minus] == 1: 
                            edge_mask[i][j_prev - minus] = 1
            # vertical #
            if not i == 0 :
                i_prev = i - 1
                if not mask[i][j] == mask[i_prev][j]: 
                    if mask[i][j]==1: # 0 -> 1
                        edge_mask[i][j] = 1 
                        for add in range(1,edge_width):
                            if i + add < h and mask[i+add][j] == 1:
                                edge_mask[i+add][j] = 1 
                    else : # 1 -> 0
                        edge_mask[i_prev][j] = 1
                        for minus in range(1,edge_width):
                            if i_prev - minus >= 0 and mask[i_prev-minus][j] == 1:
                                edge_mask[i_prev-minus][j] == 1
    return edge_mask

def viz_data(im, seg, color, inner_alpha = 0.3, edge_alpha = 1, edge_width = 5):
     
    edge = mask_edge_detection(seg, edge_width)

    color_mask = np.zeros((edge.shape[0]*edge.shape[1], 3))
    l_loc = np.where(seg.flatten() == 1)[0]
    color_mask[l_loc, : ] = color
    color_mask = np.reshape(color_mask, im.shape)
    mask = np.concatenate((seg[:,:,np.newaxis],seg[:,:,np.newaxis],seg[:,:,np.newaxis]), axis = -1)
    
    color_edge = np.zeros((edge.shape[0]*edge.shape[1], 3))
    l_col = np.where(edge.flatten() == 1)[0]
    color_edge[l_col,:] = color
    color_edge = np.reshape(color_edge, im.shape)
    edge = np.concatenate((edge[:,:,np.newaxis],edge[:,:,np.newaxis],edge[:,:,np.newaxis]), axis = -1)


    im_new = im*(1-mask) + im*mask*(1-inner_alpha) + color_mask * inner_alpha
    im_new =  im_new*(1-edge) + im_new*edge*(1-edge_alpha) + color_edge*edge_alpha

    return im_new 

def arg_parse():
    parser = argparse.ArgumentParser(description='Tools to visualize semantic segmentation map.')

    # Datasets parameters
    parser.add_argument('--img_path', type=str, default='', 
                    help="path to RGB image")
    parser.add_argument('--seg_path', type=str, default='', 
                    help="path to seg")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cmap = cls_color
    args = arg_parse()

    img_path = args.img_path
    seg_path = args.seg_path

    img = scipy.misc.imread(img_path)
    seg = scipy.misc.imread(seg_path)

    cs = np.unique(seg)

    for c in cs:
        if not c == 0:
            mask = np.zeros((img.shape[0], img.shape[1]))
            ind = np.where(seg==c)
            mask[ind[0], ind[1]] = 1
            img = viz_data(img, mask, color=cmap[c])
    scipy.misc.imsave('./exp.png', img)
