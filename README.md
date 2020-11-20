# TensorFlow-Water-Net
This is the code of the implementation of the underwater image enhancement network (Water-Net) described in "Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , <An Underwater Image Enhancement Benchmark Dataset and Beyond> IEEE TIP 2019". If you use our code or dataset for academic purposes, please consider citing our paper. Thanks.

# Requirement
TensorFlow 1.x, Cuda 8.0, and Matlab.
The missed vgg.py has been added. 
The requirement.txt has been added.

## **Usage**

## Testing
1. Clone the repo
2. Download the checkpoint from Dropbox: https://www.dropbox.com/s/fkoox0t3jwrf92q/checkpoint.rar?dl=0 or Baidu Cloud: https://pan.baidu.com/s/1aWckT66dWbB0-h1DJxIUsg
3. Generate the preprocessing data by using the "generate_test_data.m" in folder named generate_test_data
(Also, there is a modified code that includes WB, HE and GC in Python code without a need for preprocessing by MATLAB. Thanks a lot, Branimir Ambrekovic <branimir@wave-tech.com>. Branimir also upgraded it to work with TF2.0. You can find the modified code in folder named testing_code_by_Branimir Ambrekovic. More details can be found in B's codes.) 
4. Put the inputs to corresponding folders (raw images to "test_real",  WB images to "wb_real", GC images to "gc_real", HE images to "ce_real")
5. Python main_test.py
6. Find the result in "test_real"

## Training
1. Clone the repo
2. Download the VGG-pretrained model from Dropbox: https://drive.google.com/open?id=1asWe_rCduu6f09uiAz_aEP4KAiuoVSRS or Baidu Cloud: https://pan.baidu.com/s/1seDVBooFkmaJ6qF5kuAIsQ (Password: c0nj) (It's preparing for perception loss.)
2. Set the network parameters, including learning rate, batch, weights of losses, etc., according to the paper
3. Generate the preprocessing training data by using the "generate_training_data.m" in folder named generate_test_data
4. Put the training data to corresponding folders (raw images to "input_train",  WB images to "input_wb_train", GC images to "input_gc_train", HE images to "input_ce_train", Ground Truth images to "gt_train"); We randomly select the training data from our released dataset. The performance of different training data is almost same
5. In this code, you can add validation data by preprocessing your validation data (with GT) by the "generate_validation_data.m" in folder named generate_test_data, then put them to the corresponding folders (raw images to "input_test",  WB images to "input_wb_test", GC images to "input_gc_test", HE images to "input_ce_test", Ground Truth images to "gt_test")
6. For your convenience, we provide a set of training and testing data. You can find them by unziping "a set of training and testing data". However, the training data and testing data are diffrent from those used in our paper.
5. Python main_.py
6. Find checkpoint in the ./checkpoint/coarse_112


# Contact Us
If you have any questions, please contact us (lichongyi25@gmail.com or lichongyi@tju.edu.cn).
