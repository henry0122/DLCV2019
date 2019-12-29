from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='HW3 for GAN, AC-GAN, DANN, improved-DANN using pytorch')

    # Datasets parameters
    ### Problem 1
    parser.add_argument('--gan_train_data_dir', type=str, default='hw3_data/face/train/', help="path to training data directory")
    
    ### Problem 2
    parser.add_argument('--acgan_train_data_dir', type=str, default='hw3_data/face/train/', help="path to training data directory")
    parser.add_argument('--acgan_train_csv', type=str, default='hw3_data/face/train.csv', help="path to training data csv")
    
    ### Problem 3
    parser.add_argument('--is_lower', action='store_true', help="whether training lower bound model or not")
    parser.add_argument('--is_upper', action='store_true', help="whether training upper bound model or not")
    parser.add_argument('--tar_dom_name', type=str, default='svhn', help="name of the target domain we want ot adaption to")
    parser.add_argument('--dann_mnistm_data_dir', type=str, default='hw3_data/digits/mnistm/', help="path to mnistm data directory")
    parser.add_argument('--dann_svhn_data_dir', type=str, default='hw3_data/digits/svhn/', help="path to svhn data directory")
    
    ### Problem 4
    parser.add_argument('--improved_dann_mnistm_data_dir', type=str, default='hw3_data/digits/mnistm/', help="path to mnistm data directory")
    parser.add_argument('--improved_dann_svhn_data_dir', type=str, default='hw3_data/digits/svhn/', help="path to svhn data directory")
    
#     parser.add_argument('--valid_data_dir', type=str, default='hw2_data/val/', help="root path to validating data directory")
#     parser.add_argument('--test_data_dir', type=str, default='hw2_data/val/img/', help="root path to testing data directory")
#     parser.add_argument('--workers', default=1, type=int, help="number of data loading workers (default: 2)")
    
    # training parameters
    parser.add_argument('--gpu', default=1, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=50, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=64, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--gamma', default=0.9, type=float,
                    help="initial learning rate")
    
    # others
    parser.add_argument('--log_dir', type=str, default='log/gan/', help="Record train/val/test acc, loss or something like that")
    parser.add_argument('--output_dir', type=str, default='output_dir/gan/', help="Output the result of model")
    parser.add_argument('--fig_dir', type=str, default='fig/', help="Path of saving figure of learning curve")
    parser.add_argument('--model_dir',  type=str, default='model_log/gan/', help="Directory for saving trained model")
    parser.add_argument('--plot_dann', action='store_true', help="whether plot t-SNE of dann")
    parser.add_argument('--plot_improved', action='store_true', help="whether plot t-SNE of improved_dann")
    
    # Testing
    parser.add_argument('--test_gan', action='store_true', help="whether test mode or not")
    parser.add_argument('--save_test_gan',  type=str, default='test_log/gan/test.png', help="Directory for saving testing output")
    ## Problem 2
    parser.add_argument('--test_acgan', action='store_true', help="whether test mode or not")
    parser.add_argument('--save_test_acgan',  type=str, default='test_log/acgan/test.png', help="Directory for saving testing output")
    ## Problem 3
    parser.add_argument('--test_dann', action='store_true', help="whether test mode or not")
    parser.add_argument('--dann_tar_data_dir',  type=str, default='test_log/dann/test.csv', help="Directory for saving testing output")
    parser.add_argument('--save_test_dann',  type=str, default='test_log/dann/test.csv', help="Directory for saving testing output")
    ## Problem 4
    parser.add_argument('--test_improved_dann', action='store_true', help="whether test mode or not")
    parser.add_argument('--save_test_improved_dann',  type=str, default='test_log/improved_dann/test.csv', help="Directory for saving testing output")
    
    args = parser.parse_args()

    return args
