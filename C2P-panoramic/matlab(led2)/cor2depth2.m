clc;clear;
% path
addpath(genpath('./affine_fit'));
addpath(genpath('./plane_line_intersect'));
addpath(genpath('./geom3d/'));
addpath(genpath('./panoContext_code/'));
data_path = '../predict_param/param_(1-17)train/';
img_path = '/home/ps/data/Z/LED3/mp3d1/image/'
d = dir([data_path, '*.mat']);
t = dir([img_path, '*.png']);
save_path = '../result_depth/';
im_h = 512;
im_w = 1024;
gt_h = 512;%512; smaller size for rendering speed
gt_w = 1024;%1024;
c_h = 1;

for i = 1:numel(d)
%     disp(i);
    i=6

    %load prediction
    load([data_path d(i).name]); 
    img = imread([img_path t(i).name]);
    height = (room_height+1)*1.6;
    loc_corner_map1 = squeeze(corner_map(1,1,:,:));%预测的1通道为角点坐标
    loc_corner_map2 = double(sigmoid(loc_corner_map1));
    loc_corner_map3 = gaussian_thresholding_nms(loc_corner_map2, 0.44, 30);
    h = ones(9);
    b = ordfilt2(loc_corner_map3,81,h);
    loc_corner_map = double(loc_corner_map3 > 0);
    [index_inn_cory1, index_inn_corx1] = find(loc_corner_map==1);
    pixel_corner = [index_inn_corx1,index_inn_cory1]; %角点坐标

    loc_normal_map1 = squeeze(normal_pred(1,:,:,:));
    normal = zeros(length(pixel_corner) ,2);
    for n = 1:length(pixel_corner)
        normal(n,:) = loc_normal_map1(:,index_inn_cory1(n),index_inn_corx1(n));
%         radius = 5; % 圆的半径  
%         %确保索引位置在合法范围内
%         y = max(1, min(index_inn_cory1(n), size(loc_normal_map1, 2)));
%         x = max(1, min(index_inn_corx1(n), size(loc_normal_map1, 3)));
%         %获取索引位置周围的圆内范围
%         circle_values = loc_normal_map1(:, max(1, y-radius):min(size(loc_normal_map1, 2), y+radius), max(1, x-radius):min(size(loc_normal_map1, 3), x+radius));
%         %计算圆内范围内所有值的平均值
%         a_mean = mean(circle_values(1,:));
%         c_mean = mean(circle_values(2,:));
%         normal(n,:) = [a_mean c_mean];
    end
    ceiling_normal = [0,0,1];
    floor_normal = [0,0,1];
    normal = [normal(:,1) normal(:,2) zeros(length(pixel_corner),1)];
    normal_all = [ceiling_normal; floor_normal; normal];
%     normal_all = normal_all(1:length(normal_all)-1,:);

    xyz_center = zeros(length(pixel_corner) ,3);
    for ii = 1:length(xyz_center) 
        xyz_center(ii,:) = uvtoxyz(pixel_corner(ii,:));
%         xyz_center(ii,:) = trans_eval(pixel_corner(ii,:));
    end
%     xyz_center(:,2) = mean(xyz_center(:,2));
%     xyz_center = xyz_center(1:length(xyz_center)-1,:);
    xyz_center(:,2) = xyz_center(:,2) - c_h;
    xyz_center = [xyz_center(:,1) xyz_center(:,3) xyz_center(:,2)];
    xyz = xyz_center;
    [~,~,p1] = affine_fit(xyz);%天花板的一个点
    ceil = [p1(:,1) p1(:,2) 1*room_height/2];
    floor = [p1(:,1) p1(:,2) -1*room_height/2]; 
    xyz_all = [ceil;floor;xyz];
    
    if 1
    % 投射
    im_depth = zeros(gt_h, gt_w);%初始化
    im_seg = zeros(gt_h, gt_w); %初始化
    %创建两个矩阵im_X和im_Y。im_X将包含X坐标，im_Y将包含水平方向从1到gt_w，垂直方向从1到gt_h的网格的Y坐标
    [im_X,im_Y] = meshgrid(1:gt_w, 1:gt_h); 
    %将im_X和im_Y组合成一个矩阵im_cor，其中每一行代表一对X和Y坐标。
    im_cor = [im_X(:),im_Y(:)];  
    [uv_im] = coords2uv(im_cor, gt_w, gt_h); %将im_cor坐标转换成uv坐标
    [ xyz_im ] = uv2xyzN(uv_im);%将uv坐标转换成xyz坐标
    cen = [0 0 0];
    
    n_all = normal_all;
    p_all = xyz_all;
    for j = 1:size(xyz_im,1)%循环每个像素点，用（0,0,0）和每个像素点构成的线段
        if mod(size(xyz,1),2)~=0
            check = zeros(1+(1+size(xyz,1))/2,1);
            I = zeros(1+(1+size(xyz,1))/2,3);
        end
        if mod(size(xyz,1),2)==0
            check = zeros(2+size(xyz,1)/2,1);
            I = zeros(2+size(xyz,1)/2,3);
        end
        %plane_line_intersect计算平面和直线的交点
        %输入:一个平面：平面由法向量和属于平面的任一点表示；线段由中心和像素xyz坐标构成
        %输出:
        % I是相交的点
        % Check是一个指示器:
        % 0:不相交
        % 1:平面与线段相交于唯一点I
        % 2:线段位于平面上
        % 3:交点位于线段之外
        for ch = 1 : size(n_all,1)
          [aa,bb] = plane_line_intersect(n_all(ch,:),p_all(ch,:),cen,xyz_im(j,:)*1000);
          I(ch,:) = aa;
          check(ch,:) = bb;
        end
%         ray = [cen xyz_im(j,:)]; %光线是原点和方向向量组成
%         for k = 1:2:size(poly,1)
%             %计算3D射线和3D多边形（输入光线和多边形3个顶点的坐标）
%             [inter1, inside1]= intersectRayPolygon3d(ray, reshape(poly(k,:),3,3));
%             % inter是包含交点坐标，如果光线不相交，则为[NaN NaN NaN]。
%             [inter2, inside2]= intersectRayPolygon3d(ray, reshape(poly(k+1,:),3,3));
% %             [I1,check1]=plane_line_intersect(n_all(k,:),p_all(k,:),cen,xyz_im(j,:)*100);
%             if sum(isnan(inter1))==0
%                 check((k+1)/2+2)= 1;
%                 I((k+1)/2+2,:)= inter1;
%             elseif sum(isnan(inter2))==0
%                 check((k+1)/2+2)= 1;
%                 I((k+1)/2+2,:)= inter2;
%             end
%         end
        id = find(check == 1);%1=平面与线段相交于唯一点
        
        if isempty(id)
           im_depth(im_cor(j,2), im_cor(j,1)) = im_depth(im_cor(j-1,2), im_cor(j-1,1));
           continue
        end
        dist = I(id,:);
        dist = sqrt(sum(dist.*dist,2));
        [dep, idx] = min(dist);
        im_depth(im_cor(j,2), im_cor(j,1)) = dep;%深度图
        im_seg(im_cor(j,2), im_cor(j,1)) = id(idx);%分割图
    end
%     imshow(im_depth,[]);
    %% 显示
    %im_seg = imresize(im_seg,[512,1024],'nearest')
    layout_edge = edge(im_seg,'canny');%边缘检测算法
    layout_edge = imfilter(layout_edge, ones(3));
%     figure;imshow(layout_edge,[])
    img_edge = img;
    img_edge(:,:,1) = img_edge(:,:,1)+uint8(layout_edge*255) ;
    imshow(img_edge,[])
    %%
    %keyboard
    % save
    save([save_path d(i).name], 'im_depth');

    end
end