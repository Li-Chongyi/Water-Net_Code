

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 %%%%%%%%%%%%%%%%%                                                                                                           %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

folder2 = 'folder of raw images';
filepaths2 = dir(fullfile(folder2,'*.jpg'));
folder3 = 'folder of GT images';
filepaths3 = dir(fullfile(folder3,'*.jpg'));

global count
count =1;

if ~exist('input_test') 
    mkdir('input_test')         
end 

if ~exist('input_wb_test') 
    mkdir('input_wb_test')         
end 
if ~exist('input_ce_test') 
    mkdir('input_ce_test')         
end 


if ~exist('input_gc_test') 
    mkdir('input_gc_test')         
end 

if ~exist('gt_test') 
    mkdir('gt_test')         
end 
for i=1:10000
    
    image1 = im2double(imread(fullfile(folder2,filepaths2(i).name)));
    GT = imread(fullfile(folder3,filepaths3(i).name));
    %%% white balance
    hazy_wb = SimplestColorBalance(uint8(255*image1));
    hazy_wb = uint8(hazy_wb);
    
    %%% CLAHE
    lab1 = rgb_to_lab(uint8(255*image1));
    lab2 = lab1;
    lab2(:, :, 1) = adapthisteq(lab2(:, :, 1));
    img2 = lab_to_rgb(lab2);
    hazy_cont = img2;
    %%% gamma correction
    hazy_gamma = image1.^0.7;
    hazy_gamma =  uint8(255*hazy_gamma);
    
    image1 = uint8(255*image1);
    
    imwrite(image1,fullfile('input_test',filepaths2(i).name ));
    imwrite(hazy_wb,fullfile('input_wb_test',filepaths2(i).name ));
    imwrite(hazy_cont,fullfile('input_ce_test',filepaths2(i).name ));
    imwrite(hazy_gamma,fullfile('input_gc_test',filepaths2(i).name ));
    imwrite(GT,fullfile('gt_test',filepaths2(i).name ));