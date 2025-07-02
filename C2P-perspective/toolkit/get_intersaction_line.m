function [a,b,c] = get_intersaction_line(model)
a = model(1, 1) - model(2, 1);
b = model(1, 2) - model(2, 2);
c = model(1, 3) - model(2, 3);