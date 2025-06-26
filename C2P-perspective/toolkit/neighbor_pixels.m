function [p1,p2] = neighbor_pixels(row,col,h,w,d);

p1 = row + [-d:d];
p1 = p1(p1>=1&p1<=h);
% p1 = min(max(p1,1),h);

p2 = col + [-d:d];
% p2 = min(max(p2,1),w);
p2 = p2(p2>=1&p2<=w);
