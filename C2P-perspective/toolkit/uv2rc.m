function [row,col] = uv2rc(point)

row = max(round(point(2)),1);
col = max(round(point(1)),1);