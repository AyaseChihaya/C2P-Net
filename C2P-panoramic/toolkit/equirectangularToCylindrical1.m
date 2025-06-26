function [X,Y,Z] = equirectangularToCylindrical1(equirectangularImage,depth)
    % 定义圆柱面的参数
    height = size(equirectangularImage, 1);
    width = size(equirectangularImage, 2);
    radius = width / (2 * pi); % 圆柱面半径
    
    [sz(1),sz(2)]= size(depth);
    coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
    coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
    phi = pi * (2 * coord_x - 1);
    theta = pi * coord_y;

    sin_theta_cos_phi = sin(theta) .* cos(phi);
    sin_theta_sin_phi = sin(theta) .* sin(phi);
    cos_theta = cos(theta);

    X = depth.*sin_theta_cos_phi;
    Y = depth.*sin_theta_sin_phi;
    Z = depth.*cos_theta;    
%     cyl = warp(X,Y,-Z,equirectangularImage);
%     surf(X,Y,Z);
%     theta_a = atan2(Y,X);
%     phi_a = atan2(sqrt(X.^2 + Y.^2),Z);
  
    %转到圆柱坐标
%     x_interp1 = sin(phi);
%     y_interp1 = cos(phi);
%     z_interp1 = theta;
%     cyl = warp(x_interp1,y_interp1,-z_interp1,equirectangularImage);
%     surf(x_interp1,y_interp1,z_interp1);
%     theta_b = (atan2(y_interp1,x_interp1)/(2*pi))*sz(2); %弧长
%  
%     Hd = max(z_interp1(:))-min(z_interp1(:));
%     Wd = 2*radius;
%     cylinder_y = Hd / 2.0 - y_interp1;
%     cylinder_x = Wd / 2.0 + x_interp1;
%      % 使用插值方法将矩阵 A 映射到圆柱面上
% %     equirectangularImage = double(equirectangularImage(:,:,1))/255;
%     cylinder_image = interp2(1:size(equirectangularImage, 2), 1:size(equirectangularImage, 1), equirectangularImage, cylinder_x, cylinder_y);
    % imshow(cylinder_image);
    
    