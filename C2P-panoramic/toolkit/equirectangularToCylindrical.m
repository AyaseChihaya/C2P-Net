function cylindricalImage = equirectangularToCylindrical(equirectangularImage)
      
    height = size(equirectangularImage, 1);
    width = size(equirectangularImage, 2);
    radius = width / (2 * pi); %圆柱半径

    % 生成圆柱面网格
%     [theta, rho] = meshgrid(linspace(0, 2*pi, size(equirectangularImage, 2)), linspace(0, height, size(equirectangularImage, 1)));
    
    cylindricalImage = zeros(height, width, size(equirectangularImage, 3));
    for y = 1:height
        for x = 1:width

            phi = (y - height/2) / radius;
            theta = (x - width/2) / radius;

            lon = theta;%横坐标
            lat = phi;%纵坐标
            %球面坐标XYZ
            x_sphere = sin(lon) * cos(lat);
            y_sphere = sin(lon)* sin(lat);
            z_sphere = cos(lon);

            theta_a = atan2(y_sphere,x_sphere);
            phi_a = atan2(sqrt(x_sphere^2 + y_sphere^2),z_sphere);
            %球面坐标转到圆柱坐标
            x_interp1 = radius * sin(theta_a);
            y_interp1 = radius * cos(theta_a);

            y_interp = height / 2.0 - y_interp1;
            x_interp = width / 2.0 + x_interp1;
            
            % Perform bilinear interpolation
            x1 = floor(x_interp);
            x2 = ceil(x_interp);
            y1 = floor(y_interp);
            y2 = ceil(y_interp);

            Q11 = equirectangularImage(y1, x1, :);
            Q12 = equirectangularImage(y2, x1, :);
            Q21 = equirectangularImage(y1, x2, :);
            Q22 = equirectangularImage(y2, x2, :);
% 
%             % Interpolate along x-axis
            R1 = (x_interp - x1) * Q21 + (x2 - x_interp) * Q11;
            R2 = (x_interp - x1) * Q22 + (x2 - x_interp) * Q12;

            % Interpolate along y-axis
            cylindricalImage(y, x, :) = (y_interp - y1) * R2 + (y2 - y_interp) * R1;

%             cylindricalImage(y, x, :) = interp2(1:size(equirectangularImage, 2), 1:size(equirectangularImage, 1), equirectangularImage, x_interp, y_interp);
            
        end
    end
end