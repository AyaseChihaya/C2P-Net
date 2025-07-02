function corner_xyz = trans_eval(cor_id, im_h, im_w)
    c_h = 1;  
    %im_h = 512;
    %im_w = 1024;

    cor_a_x = cor_id(1) - im_w/2;
    cor_a_y = -(im_h - cor_id(2))+im_h/2;
%     cor_a_y_ = (im_h - cor_id(2))-im_h/2;
    theta_x = 2*pi*cor_a_x/im_w;
    theta_y = pi*cor_a_y/im_h;
%     theta_y_ = pi*cor_a_y_/im_h;%角度
    r = abs(cot(theta_x))*c_h;%半径-点到坐标原点的距离
    cor_a_X = r*cos(-theta_x+pi/2);% *ch
    cor_a_Z = r*sin(-theta_x+pi/2);% *ch 
    cor_a_Y_ = r*tan(-theta_y)+c_h;
 
    corner_xyz = [cor_a_X, cor_a_Y_, cor_a_Z]; 
%     box_h = mean([cor_a_Y_, cor_b_Y_, cor_c_Y_, cor_d_Y_]);%, cor_e_Y_, cor_f_Y_]);