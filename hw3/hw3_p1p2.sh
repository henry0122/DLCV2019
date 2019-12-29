 #!/bin/bash
# TODO: create shell script for running the testing code of DLCV hw3 p1 & p2
#
#    Parameter: 
#    ${1} is the folder to which you should output your fig1_2.jpg and fig2_2.jpg.

wget -O p1_generator.pth.tar "https://www.dropbox.com/s/ss8r04cqs2tcb1m/p1_generator.pth.tar?dl=0"
python3 gan.py --test_gan --save_test_gan ${1}
wget -O p2_499_generator.pth.tar "https://www.dropbox.com/s/dsxf2w2wsp928vo/p2_499_generator.pth.tar?dl=0"
python3 ac_gan.py --test_acgan --save_test_acgan ${1}

 exit 0