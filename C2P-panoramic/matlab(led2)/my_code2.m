clc;clear;close all;
% path
addpath(genpath('./affine_fit'));
addpath(genpath('./plane_line_intersect'));
addpath(genpath('./geom3d/'));
addpath(genpath('./panoContext_code/'));
addpath(genpath('./tools/'));
% data_path = '../result_PC/';
data_path = '/home/ps/data/Q/matlab/data/data_depth/';
% img_path = '/home/ps/data/Z/LayoutNetv2-master/data/layoutnet_dataset/test/pano/';
% img_path = '/home/ps/data/Z/LayoutNetv2-master/data/layoutnet_dataset/test/standard/'
img_path = '/home/ps/data/Q/matlab/data/data_depth/'
d = dir([data_path, '*.txt']);
t = dir([img_path, '*.png']);
save_path = '/home/stu1/data/Q/vilizer/matlab/result_gen_depth/12-15';
im_h = 512;
im_w = 1024;
gt_h = 512;%512; smaller size for rendering speed
gt_w = 1024;%1024;
c_h = 1;

for i = 1:numel(d)
%     disp(i);
    i=1

    %load prediction
%     load([data_path d(i).name]); %可以知道墙角点8个的2D坐标
    data = fileread([data_path d(i).name]);
    data = jsondecode(data);
    img = imread([img_path t(i).name]);
    xyz_gt = zeros(data.layoutPoints.num,3);
    uv = zeros(data.layoutPoints.num,2);
    radio = data.layoutHeight/1.6;
%     for num1 =1 :data.layoutPoints.num
%          uv(num1,:) = data.layoutPoints.points(num1).coords';
%     end
    for num1 =1 :data.layoutPoints.num
        xyz_gt(num1, :) = data.layoutPoints.points(num1).xyz';
%         xyz(2*num1-1,:) =xyz_gt;
%         xyz(2*num1,:) = [xyz_gt(1,1),-1/radio,xyz_gt(1,3)];
    end
    % get depth
    % 重新居中（abcd——xyz）
    %xyz是8个三维点
    xyz(:,2) = xyz(:,2) - c_h;%设置约束相对位置是1：|V1-V2|=1,V1V2两个墙角点
    xyz = [xyz(:,1) xyz(:,3) xyz(:,2)];
    
    % 检查相机中心是否超出布局边界，如果超出，则设置为零
    %in= 1判断点要么严格在内部，要么在其顶点指定的区域的边上；
    %检查点 (0,0) 是否在区域内
    in = inpolygon(0,0,xyz(1:2:end,1),xyz(1:2:end,2));
    if ~in
        im_depth = zeros(im_h, im_w);
        save([save_path d(i).name], 'im_depth');
        continue
    end

    % 天花板
    %n:垂直于平面的法向量
    %V:位移向量。V的列构成平面的标准正交基
    %p:属于平面的点——最近点
    [n1,v1,p1] = affine_fit(xyz(1:2:end,:));%用上面四个点计算最适合的一组天花板采样点平面(该平面法线距离的最小平方)
    % 地面
    [n2,v2,p2] = affine_fit(xyz(2:2:end,:));%用下面四个点计算最适合的一组地面采样点平面(该平面法线距离的最小平方)
    n = zeros(length(xyz)/2,3,1);
    v = zeros(length(xyz)/2,3,2);
    p = zeros(length(xyz)/2,1,3);
    for n_p = 1:2:length(xyz)
        a = mod(n_p,length(xyz));
        b = mod(n_p+1,length(xyz));
        c = mod(n_p+2,length(xyz));
        d = mod(n_p+3,length(xyz));
        if a == 0
            a = length(xyz);
        end
        if b == 0
            b = length(xyz);
        end
        if c == 0
            c = length(xyz);
        end
        if d == 0
            d = length(xyz);
        end
        id = (n_p+1)/2;
        [n(id,:,:),v(id,:,:),p(id,:,:)] = affine_fit(xyz([a,b,c,d],:));
    end
    %     % 第一个垂直面
%     [n3,v3,p3] = affine_fit(xyz([1,2,3,4],:));
%     % 第二个垂直面
%     [n4,v4,p4] = affine_fit(xyz([3,4,5,6],:));
%     % 第三个垂直面
%     [n5,v5,p5] = affine_fit(xyz([5,6,7,8],:));
%     % 第四个垂直面
%     [n6,v6,p6] = affine_fit(xyz([7,8,1,2],:));
    n_all = [n1';n2'];
    p_all = [p1;p2];
    for n_a = 3 : size(n,1)
        n_all(n_a,:) = n(n_a-2,:)';
        p_all(n_a,:) = p(n_a-2,:);
    end
%     n_all = [n1';n2';n3';n4';n5';n6'];
%     p_all =[p1;p2;p3;p4;p5;p6];
%     n_all = [n1';n2'];
%     p_all =[p1;p2];
    poly = zeros(data.layoutPoints.num,9);
    for j = 1:2:data.layoutPoints.num-2 %j=1;3;5;每一次预测两个面
        poly_t= xyz(j:j+2,:);
        poly(j,:) = poly_t(:);
        poly_t= xyz(j+1:j+3,:);
        poly(j+1,:) = poly_t(:);
    end%把xyz的每三行重新拼成一行（6种）
    %每三个点确定一个平面
    poly_t= xyz([j+2,j+3,1],:);
    poly(end-1,:) = poly_t(:);
    poly_t= xyz([j+3,1,2],:);
    poly(end,:) = poly_t(:);
    
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
    
    for j = 1:size(xyz_im,1)%循环每个像素点，用（0,0,0）和每个像素点构成的线段
        check  = zeros(2+data.layoutPoints.num/2,1);
        I = zeros(2+data.layoutPoints.num/2,3);
        %plane_line_intersect计算平面和直线的交点
        %输入:一个平面：平面由法向量和属于平面的任一点表示；线段由中心和像素xyz坐标构成
        %输出:
        % I是相交的点
        % Check是一个指示器:
        % 0:不相交
        % 1:平面与线段相交于唯一点I 
        % 2:线段位于平面上
        % 3:交点位于线段之外
        % n_all:法向量
        % p_all:点
        for ch = 1 : size(n_all,1)
          [aa,bb] = plane_line_intersect(n_all(ch,:),p_all(ch,:),cen,xyz_im(j,:)*1000);
          I(ch,:) = aa;
          check(ch,:) = bb;
        end
%         [I1,check1]=plane_line_intersect(n_all(1,:),p_all(1,:),cen,xyz_im(j,:)*1000);
%         [I2,check2]=plane_line_intersect(n_all(2,:),p_all(2,:),cen,xyz_im(j,:)*1000);
%         [I3,check3]=plane_line_intersect(n_all(3,:),p_all(3,:),cen,xyz_im(j,:)*1000);
%         [I4,check4]=plane_line_intersect(n_all(4,:),p_all(4,:),cen,xyz_im(j,:)*1000);
%         [I5,check5]=plane_line_intersect(n_all(5,:),p_all(5,:),cen,xyz_im(j,:)*1000);
%         [I6,check6]=plane_line_intersect(n_all(6,:),p_all(6,:),cen,xyz_im(j,:)*1000);
% 
%         I(1,:) = I1; check(1) = check1;
%         I(2,:) = I2; check(2) = check2;
%         I(3,:) = I3; check(3) = check3;
%         I(4,:) = I4; check(4) = check4;
%         I(5,:) = I5; check(5) = check5;
%         I(6,:) = I6; check(6) = check6;
        ray = [cen xyz_im(j,:)]; %光线是原点和方向向量组成
        for k = 1:2:size(poly,1)
            %计算3D射线和3D多边形（输入光线和多边形3个顶点的坐标）   ploy is plane
            % inside 1:平面与线段相交于唯一点I
            % 2:线段位于平面上
            % 3:交点位于线段之外
            [inter1, inside1]= intersectRayPolygon3d(ray, reshape(poly(k,:),3,3));
            % inter是包含交点坐标，如果光线不相交，则为[NaN NaN NaN]。
            [inter2, inside2]= intersectRayPolygon3d(ray, reshape(poly(k+1,:),3,3));
%             [I1,check1]=plane_line_intersect(n_all(k,:),p_all(k,:),cen,xyz_im(j,:)*100);
            if sum(isnan(inter1))==0
                check((k+1)/2+2)= 1;
                I((k+1)/2+2,:)= inter1;
            elseif sum(isnan(inter2))==0
                check((k+1)/2+2)= 1;
                I((k+1)/2+2,:)= inter2;
            end
        end
        id = find(check == 1);%1=平面与线段相交于唯一点
        
        if isempty(id)
           im_depth(im_cor(j,2), im_cor(j,1)) = im_depth(im_cor(j-1,2), im_cor(j-1,1));
           continue
        end
        dist = I(id,:);%I 是光线与墙面的交点
        dist = sqrt(sum(dist.*dist,2));%欧氏距离
        [dep, idx] = min(dist);
        im_depth(im_cor(j,2), im_cor(j,1)) = dep;%深度图
        im_seg(im_cor(j,2), im_cor(j,1)) = id(idx);%分割图
    end
     %imshow(im_depth,[]);
    %% 显示
    %im_seg = imresize(im_seg,[512,1024],'nearest')
    layout_edge = edge(im_seg,'canny');%边缘检测算法
    layout_edge = imfilter(layout_edge, ones(3));%filter
    %figure;imshow(layout_edge,[])
    img_edge = img;
    img_edge(:,:,1) = img_edge(:,:,1)+uint8(layout_edge*255) ;
    imshow(img_edge,[])
    %%
    %keyboard
    % save
    save([save_path d(i).name], 'im_depth');

    end
end