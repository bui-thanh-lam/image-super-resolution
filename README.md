# Single image super resolution with SRCNN & SRCNN++

## 1. Overview

Super resolution is a ill-posed problem in Computer Vision: given a low resolution image (LR), you must produce a high resolution one (SR) 
which is comparable to a reference image (HR).

Before deep learning era, there have been many super resolution methods such as bilinear, bicubic. In this project, we re-implement SRCNN (Chao Dong et al., 2015),
which is the very first method using CNN to solve the problem. Furthermore, we explore some pros and cons of SRCNN and then propose a modified version called SRCNN++.

We take the main idea of SRCNN, but then use sub-pixel layer (W. Shi et al., 2016) instead of bicubic layer, and a combined loss (C. Ledig et al., 2017) instead of
pixel loss. Our method produce slightly higher PSNR than the original SRCNN, while training and prediction time is competitive.

## 2. How to use

### Prerequisites: Python 3.6+ (test on 3.6, 3.7)

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


### Use for prediction

*(one image, .jpg or .png)*

`` python infer.py \

--lr_file [path_to_input_file] --sr_file [path_to_output_file] \

--scale_factor [scale_factor] --color_mode [color_mode] \

--model [model_type] --from_pretrained [path_to_weights] ``

### Use for training

You must provide a metadata file that contains all of your image file names. It is a text file like this:

img_000.png
img_002.png
...
img_999.png

This filename is identical for each input image and its target

If you have another metadata file format, please write your custom PyTorch Dataset

Command:

`` python train.py \

--lr_dir [path_to_input_dir] --hr_dir [path_to_target_dir] --meta_file [path_to_metadata_file] \

--model [model_type] --loss [loss_type] \

--scale_factor [scale_factor] --color_mode [color_mode] \

--lr [learning_rate] --bs [batch_size] --n_epochs [num_of_epochs] \

--saved_weights_file [path_to_save_the_weights]``

### Use for evaluation

We provide Set14 dataset and its metadata file for benchmark (compute PSNR on test set).

You can use any other dataset if you want; however, you must also provide a metadata file.

Command:

`` python evaluate.py \

--lr_dir [path_to_input_dir] --hr_dir [path_to_target_dir] --meta_file [path_to_metadata_file] \

--model [model_type] \

--scale_factor [scale_factor] --color_mode [color_mode] \

--bs [eval_batch_size] \

--from_pretrained [path_to_weights] ``

## Some helpful notes

Run python [script_name].py --help if you need help command usage.

We trained our models on a P100 GPU with 8000 images, target size 100x100 pixels. PSNR on Set14 was ~ 30 dB.

We produce LR from HR by resize with bicubic, nearest neighbor, bilinear, lanczos,...We will provide the script to do this job soon.

For hyper parameter tuning, please read the code and modify it if you want. Combined loss may slow down training time.

For prediction, you should not feed a high resolution to the model. Model performs best when new image size is nearly the same as training images size.

If you use SRCNN++, it cannot perform multiple scale factors. You must train separated models for each one.

## 3. How to contribute

If there are any issues with this repository, feel free to leave it to Issues tab.

If you have any idea to extend this project or improve models' performance, please fork and make a Pull Request.

You can use this code totally free for any purposes. But remember, this is a research code.
