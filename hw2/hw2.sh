 #!/bin/bash
# TODO: create shell script for running the testing code of the baseline model
#    Parameter: 
#    ${1}: testing images directory(e.g. test/images/) 
#    ${2}: output prediction directory (e.g. test/labelTxt_hbb_pred/) 

wget -O baseline_model_best.pth.tar "https://www.dropbox.com/s/wfy603hlmw1yymm/baseline_model_best.pth.tar?dl=0"
python3 determine.py --test_out_dir ${2} --test_data_dir ${1} --model_dir baseline_model_best.pth.tar --baseline --test_batch 32

 exit 0