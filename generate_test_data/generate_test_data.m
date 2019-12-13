

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 %%%%%%%%%%%%%%%%%                                                                                                           %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

folder2 = 'folder of raw images';
filepaths2 = dir(fullfile(folder2,'*.jpg'));

global count
count =1;

if ~exist('test_real') 
    mkdir('test_real')         
end 

if ~exist('wb_real') 
    mkdir('wb_real')         
end 
if ~exist('ce_real') 
    mkdir('ce_real')         
end 


if ~exist('gc_real') 
    mkdir('gc_real')         
end 

for i=1:10000
    
    image1 = im2double(imread(fullfile(folder2,filepaths2(i).name)));
    
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
    
    imwrite(image1,fullfile('test_real',filepaths2(i).name ));
    imwrite(hazy_wb,fullfile('wb_real',filepaths2(i).name ));
    imwrite(hazy_cont,fullfile('ce_real',filepaths2(i).name ));
    imwrite(hazy_gamma,fullfile('gc_real',filepaths2(i).name ));


end

