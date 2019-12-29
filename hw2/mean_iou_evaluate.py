import numpy as np
import scipy.misc
import argparse
import os
import glob

def read_masks(seg_dir, seg_h=352, seg_w=448):
    '''
    Read masks from directory
    '''
    seg_paths = glob.glob(os.path.join(seg_dir,'*.png'))
    seg_paths.sort()
    segs = np.zeros((len(seg_paths), seg_h, seg_w))
    for idx, seg_path in enumerate(seg_paths):
        seg = scipy.misc.imread(seg_path)
        segs[idx, :, :] = seg
    
    return segs

def mean_iou_score(pred, labels, num_classes=9):
    '''
    Compute mean IoU score over 9 classes
    '''
    mean_iou = 0
    for i in range(num_classes):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
    args = parser.parse_args()

    pred = read_masks(args.pred)
    labels = read_masks(args.labels)
        
    print('# preds: {}; pred.shape: {}'.format(len(pred), pred.shape))
    print('# labels: {}; labels.shape: {}'.format(len(labels), labels.shape))
    mean_iou_score(pred, labels)
