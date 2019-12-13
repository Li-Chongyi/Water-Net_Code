# TensorFlow-Water-Net
This is the code of the implementation of the underwater image enhancement network (Water-Net) described in "Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , <An Underwater Image Enhancement Benchmark Dataset and Beyond> IEEE TIP 2019". If you use our code or dataset for academic purposes, please consider citing our paper. Thanks.

# Requirment
TensorFlow and Matlab.

## **Usage**
1. Clone the repo
2. Download the checkpoint from https://www.dropbox.com/s/fkoox0t3jwrf92q/checkpoint.rar?dl=0;
3. Generate the preprocessing data by usint the "generate_test_data.m" in the generate_test_data.
4. Put the inputs to corresponding folders (raw images to "test_real",  WB images to "wb_real", GC images to "gc_real", HE images to "ce_real");
5. Python main_test.py;
6. Find the result in "test_real".

# Contact Us
If you have any questions, please contact us (lichongyi25@gmail.com or lichongyi@tju.edu.cn).
