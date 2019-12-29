




# HW3 ― GAN, ACGAN and UDA
In this assignment, you are given datasets of human face and digit images. You will need to implement the models of both GAN and ACGAN for generating human face images, and the model of DANN for classifying digit images from different domains.

For more details, please check the slides of HW3.

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2019/hw3-<username>.git
    
Note that you should replace `<username>` with your own GitHub username.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw3_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://www.dropbox.com/s/65qdt9rkt808an4/hw3_data.zip) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> - You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw3_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.
> - Using external dataset is forbidden for this homework.
> - Imagenet pre-trained weight and bias are not allowed to be used in this homework. 

### Evaluation
To evaluate your UDA models in Problems 3 and 4, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw3_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw3_data/digits/svhn/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw3_data/digits/svhn/test.csv`)

Note that for `hw3_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

# Submission Rules
### Deadline
108/12/03 (Tue.) 03:00 AM

### Late Submission Policy
You have a five-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

**NOTE:** To encourage students to submit homework on time, students using no more than three late-day quota will receive a bonus of two points in their final scores. Students using four late-day quota in this semester will receive a bonus of one point in their final scores.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above.

⚠️ Late Submission Forms⚠️ 
> **If your are going to submit your homework after the homework deadline, please follow the steps below carefully.**
> - You must fill in this form befofre the deadline of hw3 to notify the TAs that you have not finished your homework. Then, we will not grade your homewrok imediately. Please note that once you submit this form your homework will be classified as late submitted ones. Even if you send the TAs an email before the homework deadline, your homework will be regarded as one day late. Therefore, before filling in this form, you must be 100% sure that you really are going to submit your homework after the deadline.
>
>    https://forms.gle/Cz6M4t3zccZwSbVa8
>
> - After you push your homework to the github repository, please fill in the form below immedaitely. We will calculate the number of late days according to the time we receive the response of this form. Please note that you will have access to this form after 108/12/03 10:00 A.M. Please also note that you can only fill in this form **once**, so you **must** make sure your homework is ready to be graded before submitting this form.
>
>    https://forms.gle/LvqnF2s81syDymwe6
>

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw3_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 1.   `hw3_p1p2.sh`  
The shell script file for running your GAN and ACGAN models. This script takes as input a folder and should output two images named `fig1_2.jpg` and `fig2_2.jpg` in the given folder.
 1.   `hw3_p3.sh`  
The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 1.   `hw3_p4.sh`  
The shell script file for running your improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

We will run your code in the following manner:

    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p1p2.sh $1
    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p3.sh $2 $3 $4
    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p4.sh $2 $3 $4

-   `$1` is the folder to which you should output your `fig1_2.jpg` and `fig2_2.jpg`.
-   `$2` is the directory of testing images in the **target** domain (e.g. `hw3_data/digits/svhn/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `svhn`, you should make your prediction using your "mnistm→svhn" model, **NOT** your "svhn→mnistm→" model.
-   `$4` is the path to your output prediction file (e.g. `hw3_data/digits/svhn/test_pred.csv`).

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> - For the sake of conformity, please use the -**python3** command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.
> - You must **not** use commands such as **rm, sudo, CUDA_VISIBLE_DEVICES**, cp, mv, mkdir, cd, pip or other commands to change the Linux environment.
> - We will execute you code on Linux system, so please make sure you code can be executed on **Linux** system before submitting your homework.
> - **DO NOT** hard code any path in your file or script except for the path of your trained model.
> - The execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
> - Use the wget command in your script to download you model files. Do not use the curl command.
> - **Do not delete** your trained model before the TAs disclose your homework score and before you make sure that your score is correct.
> - Do not create the `$1` and `$4` directories in your shell script or python codes.
> - If you use matplotlib in your code, please add matplotlib.use(“Agg”) in you code or we will not be able to execute your code.
> - Do not use imshow() or show() in your code or your code will crash.
> - Use os.path.join to deal with path issues as often as possible.
> - Please do not upload your training information generated by tensorboard to github.

### Packages
This homework should be done using python3.6 and you can use all the python3.6 standard libraries. For a list of third-party packages allowed to be used in this assignment, please refer to the requirments.txt for more details.
You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).

- **If we can not reproduce the scores or images in your report, you will get 0 points in the corresponding problem.**
- **If we can not execute your code, we will give you a chance to make minor modifications to your code. After you modify your code**
    - If we can execute your code and reproduce your results in the report, you will still receive a 30% penalty in your homework score.
    - If we can run your code but cannot reproduce your results in the report, you will get 0 points in the corresponding problem.
    - If we still cannot execute your code, you will get 0 in the corresponding problem.

# Q&A
If you have any problems related to HW3, you may
- Use TA hours (please check [course website](http://vllab.ee.ntu.edu.tw/dlcv.html) for time/location)
- Contact TAs by e-mail ([ntudlcvta2019@gmail.com](mailto:ntudlcvta2019@gmail.com))
- Post your question in the comment section of [this post]()

