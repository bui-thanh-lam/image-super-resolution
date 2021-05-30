# Single image super resolution with SRCNN & SRCNN++

## 1. Overview

Super resolution is a ill-posed problem in Computer Vision: given a low resolution image (LR), you must produce a high resolution one (SR) 
which is comparable to a reference image (HR).

Before deep learning era, there have been many super resolution methods such as bilinear, bicubic. In this project, we re-implement SRCNN (Chao Dong et al., 2015),
which is the very first method using CNN to solve the problem. Furthermore, we explore some pros and cons of SRCNN and then propose a modified version called SRCNN++.

We take the main idea of SRCNN, but then use sub-pixel layer (W. Shi et al., 2016) instead of bicubic layer, and a combined loss (C. Ledig et al., 2017) instead of
pixel loss. Our method produce slightly higher PSNR than the original SRCNN, while training and inferrence time is competitive.

## 2. How to use

### Prerequisites: Python 3.6+

### Clone project

`` git clone https://github.com/bui-thanh-lam/image-super-resolution.git ``

`` cd image-super-resolution ``

### Create virtual environment

*(optional, recommended)*

`` virtualenv venv ``

`` source venv/bin/acitvate ``

### Install requirements

`` pip install -r requirements.txt ``

`` cd src ``


### Use for inference

*(one image, .jpg or .png)*

`` python infer.py -i [path_to_input] -o [path_to_output] -m [model_name] -W [path_to_weights] ``

or

`` python infer.py --input [path_to_input] --output [path_to_output] --model [model_name] --weight [path_to_weights] ``

### Use for training

*Note: the data folder structure must be like this:*

> data

>> train *(required)*

>>> HR

>>> LR

>> validation *(optional)*

>>> HR

>>> LR

>> test *(optional)*

>>> HR

>>> LR

`` python train.py --model [model_name] --n_examples [num_of_training_examples] 
--n_epochs [num_of_epochs] --weight [path_to_saved_weights] --pretrained [path_to_pretrained_weights]``

We trained our models on a P100 GPU with 8000 images. 
It took about 2.5 GB VRAM and 30 seconds/epoch (except i/o reading time which is about 1.5 hours).

For other hyper parameter tuning, please read the code and modify it if you want.

# How to contribute

If there are any issues with this repository, feel free to leave it to Issue tab.

If you have any idea to extend this project or improve models' performance, please make a pull request.

You can use this code totally free for any purposes.
