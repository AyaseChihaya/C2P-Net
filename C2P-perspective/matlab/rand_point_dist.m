% clear;clc;
% 
% x00 = [0,0];
% x01 = [0,1];
% x10 = [1,0];
% x11 = [1,1];


for i = 1:100000

p = rand(1,2);

d00 = sum(abs(p-x00));
d01 = sum(abs(p-x01));
d10 = sum(abs(p-x10));
d11 = sum(abs(p-x11));

dsort = sort([d00,d01,d10,d11]);

dmin(i) = dsort(1);

end
hist(dmin,100)

mdm = mean(dmin);