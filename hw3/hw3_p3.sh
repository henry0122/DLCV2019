 #!/bin/bash
# TODO: create shell script for running the testing code of your improved model
#
#    Parameter: 
#    ${1}: the directory of testing images in the target domain (e.g. hw3_data/digits/svhn/test).
#    ${2}: a string that indicates the name of the target domain, which will be either mnistm or svhn.
#    ${3}: the path to your output prediction file (e.g. hw3_data/digits/svhn/test_pred.csv).

wget -O best_m_s_adapt.pth.tar "https://www.dropbox.com/s/m5nhm546y6paknn/best_m_s_adapt.pth.tar?dl=0"
wget -O best_s_m_adapt.pth.tar "https://www.dropbox.com/s/h0ubr3mhgmohezu/best_s_m_adapt.pth.tar?dl=0"
python3 daNN.py --tar_dom_name ${2} --dann_tar_data_dir ${1} --test_batch 64 --save_test_dann ${3} --test_dann

 exit 0