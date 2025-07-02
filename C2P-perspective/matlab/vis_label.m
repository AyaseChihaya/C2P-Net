clear;clc;

load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/training.mat
img_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/training/image/';
seg_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/training/layout_seg/';


i = 3000;


i = i + 1;

img = imread([img_path data(i).image]);
figure,imshow(img)
seg = imread([seg_path data(i).layout_seg]);
figure,imshow(seg==3)