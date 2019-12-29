from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='HW2 for Semantic Segmentation using pytorch')

    # Datasets parameters
    parser.add_argument('--train_data_dir', type=str, default='hw2_data/train/', help="root path to training data directory")
    parser.add_argument('--valid_data_dir', type=str, default='hw2_data/val/', help="root path to validating data directory")
    parser.add_argument('--test_data_dir', type=str, default='hw2_data/val/img/', help="root path to testing data directory")
    parser.add_argument('--workers', default=1, type=int,
                    help="number of data loading workers (default: 2)")
    
    # training parameters
    parser.add_argument('--gpu', default=1, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=300, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=8, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=8, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.00002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0, type=float,
                    help="initial learning rate")
    
    # train baseline model or not
    parser.add_argument('--baseline', action='store_true', help="whether training baseline model or improved one")
    # others
    parser.add_argument('--save_dir', type=str, default='log/', help="Record train/val/test acc, loss or something like that")
    parser.add_argument('--output_dir', type=str, default='output_dir/val/', help="Output the result of model")
    parser.add_argument('--random_seed', type=int, default=42)
    
    parser.add_argument('--test', action='store_true', help="whether test mode or not")
    parser.add_argument('--model_dir',  type=str, default='model_log/', help="Directory for saving trained model")
    parser.add_argument('--test_out_dir',  type=str, default='test_log/', help="Directory for saving teting output")

    args = parser.parse_args()

    return args
