# TensorFlow-DUIENet
TensorFlow implementation of the underwater image enhancement network (Water-Net) described in "Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , <An Underwater Image Enhancement Benchmark Dataset and Beyond> IEEE TIP 2019". If you use our code or dataset for academic purposes, please consider citing our paper. Thanks.

# Dependencies:
Tensorflow and Matlab.

## **How to enhance the underwater images**
1. Download the checkpoint from https://www.dropbox.com/s/fkoox0t3jwrf92q/checkpoint.rar?dl=0;
2. Generate the preprocessing data by usint the "generate_test_data.m" in the generate_test_data.
3. Put the inputs to corresponding folders (raw images to "test_real",  WB images to "wb_real", GC images to "gc_real", HE images to "ce_real");
4. Python main_test.py;
5. Find the result in "test_real".


