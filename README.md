# TensorFlow-Water-Net
This is the code of the implementation of the underwater image enhancement network (Water-Net) described in "Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , <An Underwater Image Enhancement Benchmark Dataset and Beyond> IEEE TIP 2019". If you use our code or dataset for academic purposes, please consider citing our paper. Thanks.

# Requirment
TensorFlow 1.x and Matlab.

## **Usage**

## Testing
1. Clone the repo
2. Download the checkpoint from Dropbox: https://www.dropbox.com/s/fkoox0t3jwrf92q/checkpoint.rar?dl=0 or Baidu Cloud: https://pan.baidu.com/s/1aWckT66dWbB0-h1DJxIUsg
3. Generate the preprocessing data by using the "generate_test_data.m" in folder named generate_test_data
(Also, there is a modified code that include WB, HE and GC in Python code without a need for preprocessing by MATLAB. Thanks a lot, Branimir Ambrekovic <branimir@wave-tech.com>. Branimir also upgraded it to work with TF2.0. You can find the modified code in folder named testing_code_by_Branimir Ambrekovic. More details can be found in B's codes.) 
4. Put the inputs to corresponding folders (raw images to "test_real",  WB images to "wb_real", GC images to "gc_real", HE images to "ce_real")
5. Python main_test.py
6. Find the result in "test_real"

## Training




# Contact Us
If you have any questions, please contact us (lichongyi25@gmail.com or lichongyi@tju.edu.cn).
