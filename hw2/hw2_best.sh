 #!/bin/bash
# TODO: create shell script for running the testing code of your improved model
#
#    Parameter: 
#    ${1}: testing images directory(e.g. test/images/) 
#    ${2}: output prediction directory (e.g. test/labelTxt_hbb_pred/) 

wget -O improved_model_best.pth.tar "https://www.dropbox.com/s/90dtd9m7xqgp761/improved_model_best.pth.tar?dl=0"
python3 determine.py --test_out_dir ${2} --test_data_dir ${1} --model_dir improved_model_best.pth.tar --test_batch 8

 exit 0