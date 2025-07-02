clear;clc;
%% 导入mat文件和路径
load /home/ps/data/L/lili_mat/datasets/mp/testing/testing/testing.mat;%导入测试mat文件
seg_path = '/home/ps/data/L/lili_mat/datasets/mp/testing/testing/layout_seg_order/';
layout_depth_o_path =  '/home/ps/data/L/lili_mat/layout_depth_output/';
img_path = '/home/ps/data/L/lili_mat/datasets/mp/testing/testing/image_order/';

dataset = data;%将data mat赋值给dataset
count=0; %代表发生错误的张数
h=112;
w=112;

%%给mat文件中每张图片添加分辨率
for j = 1:length(dataset)
    sz(1)=1024;
    sz(2)=1280;
end

result = {};%用来存放结果的
sz = [1024,1280]; %图像分辨率

%建立xy坐标系
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1); %
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
%% 遍历每张图片
for i = 1:length(dataset)%
    
    i=3;
    try
    dataset(i).resolution=sz;%给data1.mat文件中添加分辨率
    %% 读取真实值
    img = imread([img_path num2str(i,'%04d') '.jpg']);%读取图片（路径+图片名）
    gt_seg = imread([seg_path num2str(i,'%04d') '.png']);%读取图片（路径+图片名）
    %导入预测结果
    load(['/home/ps/data/L/lili_mat/predict_param/predict_param_cluster(12-6-newcrf-onlypointdepth)/output1/' num2str(i-1) '.mat']);
    load(['/home/ps/data/L/lili_mat/predict_param/predict_param_cluster(12-6-newcrf-onlypointdepth)/output2/' num2str(i-1) '.mat']);
    load(['/home/ps/data/L/lili_mat/predict_param/predict_param_cluster(12-6-newcrf-onlypointdepth)/output3/' num2str(i-1) '.mat']);
%     load(['/home/ps/data/L/lili_mat(cluster2)/predict_param/predict_param_cluster(10-13-2)/output1/' num2str(i-1) '.mat']);
%     load(['/home/ps/data/L/lili_mat(cluster2)/predict_param/predict_param_cluster(10-13-2)/output2/' num2str(i-1) '.mat']);
%     load(['/home/ps/data/L/lili_mat(cluster2)/predict_param/predict_param_cluster(10-13-2)/output3/' num2str(i-1) '.mat']);
    loc_inner_map1 = squeeze(output1(1,1,:,:));%预测的1通道为内点坐标
    loc_inner_map2 = double(sigmoid(loc_inner_map1));

    loc_bordr_map1 = squeeze(output1(1,2,:,:));%预测的2通道为边界点坐标
    loc_bordr_map2 = double(sigmoid(loc_bordr_map1));
%     
%     loc_inner_map3 = double(loc_inner_map2 > 0.1);%0.06  0.1   0.2
%     loc_bordr_map3 = double(loc_bordr_map2 > 0.4);%0.1   0.2   0.4
    
    loc_bordr_map3 = gaussian_thresholding_nms(loc_bordr_map2, 0.4, 5);
    loc_inner_map3 = gaussian_thresholding_nms(loc_inner_map2, 0.316, 5);
    loc_bordr_map = double(loc_bordr_map3 > 0);
    loc_inner_map = double(loc_inner_map3 > 0);
    %% 求内点坐标
%     conn_comp = bwconncomp(loc_inner_map3);%查找连通区域
%     regions = regionprops(conn_comp, 'PixelList');%连通区域像素坐标
%     num_regions = conn_comp.NumObjects;%连通区域数量
%     keypoints = zeros(num_regions, 2);
%     loc_inner_map = zeros(h,w);
%     for ii = 1:num_regions%遍历每个连通区域
%         aa = fliplr(regions(ii).PixelList);%取得每个连通区域的坐标
%         region_pixels = zeros(size(aa,1),1);
%         for aaa = 1:size(aa,1)%遍历每个坐标，获得每个坐标处的像素值
%             region_pixels(aaa) = loc_inner_map2(aa(aaa,1),aa(aaa,2));
%         end
%         [~, max_idx] = max(region_pixels);%求最大像素值的索引
%         row_i = aa(max_idx,1);
%         col_i = aa(max_idx,2);
%         loc_inner_map(row_i,col_i)=1;
%     end
    %%去除边缘的点
    one_intrix = zeros(size(loc_inner_map));
    one_intrix(2:end-1,2:end-1)=1;
    loc_inner_map  = one_intrix .*loc_inner_map;
    %% 求边界点坐标
%     conn_comp2 = bwconncomp(loc_bordr_map3);%查找连通区域
%     regions2 = regionprops(conn_comp2, 'PixelList');%连通区域像素坐标
%     num_regions2 = conn_comp2.NumObjects;%连通区域数量
%     keypoints2 = zeros(num_regions2, 2);
%     loc_bordr_map = zeros(h,w);
%     for ii2 = 1:num_regions2%遍历每个连通区域
%         aa2 = fliplr(regions2(ii2).PixelList);%取得每个连通区域的坐标
%         region_pixels2 = zeros(size(aa2,1),1);
%         for aaa2 = 1:size(aa2,1)%遍历每个坐标，获得每个坐标处的像素值
%             region_pixels2(aaa2) = loc_bordr_map2(aa2(aaa2,1),aa2(aaa2,2));
%         end
%         [~, max_idx2] = max(region_pixels2);%求最大像素值的索引
%         row_b = aa2(max_idx2,1);
%         col_b = aa2(max_idx2,2);
%         loc_bordr_map(row_b,col_b)=1;
%     end

    %%
    predict_lay_depth = squeeze(output2(1,1,:,:));%预测的1通道为layoutdepth
%     pred_push = squeeze(output2(1,2,:,:));  %
%     pred_pull = squeeze(output2(1,3,:,:));
%     pred_param1 = permute(squeeze(output2(1,2:5,:,:)),[2,3,1]);  %预测的5-8通道为中心点的pqrs参数
    %pred_embedding =   permute(squeeze(output3(1,1:2,:,:)),[2,3,1]);
    [index_inn_cory1, index_inn_corx1] = find(loc_inner_map==1); %
    [index_bor_cory1, index_bor_corx1] = find(loc_bordr_map==1); %

    if isempty(index_inn_cory1) && isempty(index_inn_corx1)
        % 内点坐标为空，直接使用边界点坐标作为合并结果
        inner_and_border_indexes = [index_bor_cory1, index_bor_corx1];
    elseif isempty(index_bor_cory1) && isempty(index_bor_corx1)
        % 边界点坐标为空，直接使用内点坐标作为合并结果
        inner_and_border_indexes = [index_inn_cory1, index_inn_corx1];
    else
        % 内点和边界点坐标均不为空，进行合并操作
        inner_and_border_indexes = cat(1, [index_inn_cory1, index_inn_corx1], [index_bor_cory1, index_bor_corx1]);
    end

%     %在1280*1024图中找到位置
%     index_inn_cory = round((index_inn_cory1-1) ./ (h-1) .* (sz(1)-1))+1 ;
%     index_inn_corx = round((index_inn_corx1-1) ./ (w-1) .* (sz(2)-1))+1;
% 
%     index_bor_cory = round((index_bor_cory1-1) ./ (h-1) .* (sz(1)-1))+1;
%     index_bor_corx = round((index_bor_corx1-1) ./ (w-1) .* (sz(2)-1))+1;
%    
%     %图像坐标——>归一化坐标
%     index_bu = (index_bor_corx-1) ./ (sz(2)-1); % 算出的边界点坐标的u
%     index_bv = (index_bor_cory-1) ./ (sz(1)-1);% 算出的边界点坐标的v

%     if ~isempty(index_inn_corx)
% 
%         index_iu = (index_inn_corx-1) ./ (sz(2)-1); % 算出的边界点坐标的u
%         index_iv = (index_inn_cory-1) ./ (sz(1)-1);% 算出的边界点坐标的v
%    
%     end


    %% 创建一个图片四个角点位置的矩阵
%     edge_point_u = [0;sz(2)-1;0;sz(2)-1];
%     edge_point_v = [0;0;sz(1)-1;sz(1)-1];
%     %归一化后的角点坐标
%     edge_point_u_norm = edge_point_u/(sz(2)-1);
%     edge_point_v_norm = edge_point_v/(sz(1)-1);
    edge_point_u1 = [1;w;1;w];
    edge_point_v1 = [1;1;h;h];
    %归一化后的角点坐标
    edge_point_u_norm = (edge_point_u1-1)./(w-1);
    edge_point_v_norm = (edge_point_v1-1)./(h-1);
    
    
    %% 取得每个点处对应的深度值倒数
    index_bz = [];
    index_ez = [];
    index_iz = [];
    for hi = 1:size(index_bor_cory1)
        index_bz1 = predict_lay_depth(index_bor_cory1(hi),index_bor_corx1(hi));
        index_bz = [index_bz;index_bz1];
    end
    
    for li = 1:size(edge_point_v1)
        index_ez1 = predict_lay_depth(edge_point_v1(li),edge_point_u1(li));
        index_ez = [index_ez;index_ez1];
    end
    if ~isempty(index_inn_corx1)
        for di = 1:size(index_inn_cory1)
            index_iz1 = predict_lay_depth(index_inn_cory1(di),index_inn_corx1(di));
            index_iz = [index_iz;index_iz1];
    
        end
    end
    %% 聚类
    pred_embedding = squeeze(output3(1,1:2,:,:));
    pred_embedding = reshape(pred_embedding,2,[]);%2*12996
%     emb_vis(:,:,1) = imresize(squeeze(output3(1,1,:,:)),[112,140]);
%     emb_vis(:,:,2) = imresize(squeeze(output3(1,1,:,:)),[112,140]);
%     emb_vis(:,:,3) = imresize(squeeze(output3(1,2,:,:)),[112,140]);
% 
% %     emb_vis = (emb_vis + 1)/2;
% %     emb_vis(:,:,1) = ones(112,140)/2;
% %     emb_vis = ycbcr2rgb(emb_vis);
%     imshow(imresize(emb_vis,[320,400]),[]);
    %%
    msd = [0.2, 0.3, 0.4, 0.5];     % try different bandWidths for meanshift clustering
    for pm = 1:length(msd)
        
        [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(pred_embedding,msd(pm));%将预测出的参数进行聚类
        gen_seg = reshape(data2cluster,[h,w]); %将一列转换成一幅图
        faces = unique(gen_seg);%墙面数
        
        select_region = [];
        filled_map = [];
        visiable_area = zeros(sz);
        rc = 0;
        count_point = 0;
        B=[];
        for f = 1:length(faces)%遍历每个面
           
            region = gen_seg==f;%找到标签为1的，令其区域为1，其余为0；找到标签为2的，令其区域为2，其余为0。。。
            pct_pix = sum(region(:))/numel(gen_seg);%计算"gen_seg"图像中属于当前面的像素所占的百分比
            if pct_pix < 0.01%如果pct_pix的值小于0.01，则跳出当前循环迭代，说明没有构成一个面
                continue
            end
            [v, u] = ind2sub([h,w],find(region));%
            border_pix = sum(u==1|u==w|v==1|v==h) / sum(region(:));%"border_pix"计算边界像素在当前面区域中的比例，即边界像素数除以当前面区域的总像素数
            l = bwlabel(region);%bwlabel函数对二进制图像"region"进行标记（labeling），生成一个标记图像"L"，其中每个连通区域被赋予一个唯一的整数标签。
            ni = [];
            for li = 1:max(l(:))
                ni(li) = sum(l(:)==li);%计算标记li的个数
            end
            new_l = l==find(ni==max(ni),1);%返回X中所有1元素的索引
            perim = regionprops(new_l,'Perimeter');%regionprops函数来计算二进制图像"new_l"中标记区域的属性。属性名称"Perimeter"表示计算每个标记区域的周长。
            ppp = perim.Perimeter / sqrt(sum(region(:)));%计算了周长与当前面区域像素数量的平方根的比值
            
            if ppp>6.5 && border_pix < 0.05    % remove unqualified clusters
                continue
            end

            % 膨胀操作
            se = strel('square', 10);
            dilated_img = imdilate(new_l, se);
            
            % 腐蚀操作
            eroded_img = imerode(new_l, se);
            %%
%             % 结合膨胀和侵蚀结果
%             combinedImage = dilated_img & ~eroded_img;
%             % 结合原始边界
%             combinedImageWithBoundary = combinedImage | new_l;
%             combinedImageWithBoundary = im2uint8(combinedImageWithBoundary);
% 
% % 显示结果
% figure;
% % subplot(2, 2, 1), imshow(originalImage), title('原始图像');
% % subplot(2, 2, 2), imshow(binaryImage), title('二值图像');
% % subplot(2, 2, 3), imshow(dilatedImage), title('膨胀后的图像');
%  imshow(combinedImageWithBoundary), title('结合结果（膨胀边界和侵蚀边界之间的区域）');
%             
%             

            
            %%
            
            coplane_point_y = [];
            coplane_point_x = [];
            
            %% 判断内点和边界点坐标是否属于像素值为1的区域
            for poi = 1:size(inner_and_border_indexes, 1)
                point_y = inner_and_border_indexes(poi,1);
                point_x = inner_and_border_indexes(poi,2);
                if dilated_img(point_y,point_x) == 1 && eroded_img(point_y,point_x) == 0
                    %disp('该点属于像素值为1的区域');
                    count_point = count_point + 1;
                    coplane_point_y = [coplane_point_y;point_y];
                    coplane_point_x = [coplane_point_x;point_x];
                end
            end
            %% 判断角点坐标是否属于像素值为1的区域
            for poi_ed = 1:4 %(4个角点)
                poi_ed_u = edge_point_u1(poi_ed);
                poi_ed_v = edge_point_v1(poi_ed);
%                 if dilated_img(poi_ed_v,poi_ed_u) == 1 && eroded_img(poi_ed_v,poi_ed_u) == 1
                if dilated_img(poi_ed_v,poi_ed_u) ~= 0
                  %disp('该点属于像素值为1的区域');
                   coplane_point_y = [coplane_point_y;poi_ed_v];
                   coplane_point_x = [coplane_point_x;poi_ed_u];
                end
            end
            %将共面的点进行归一化
            coplane_point_y_norm = (coplane_point_y-1)./(h-1);
            coplane_point_x_norm = (coplane_point_x-1)./(w-1);
            len = size(coplane_point_y);
            A = [coplane_point_x_norm coplane_point_y_norm ones(len)];%创建矩阵AB=Y中的A
            D = [];
                   
            for gg=1:length(coplane_point_y)
                D1 = predict_lay_depth(coplane_point_y(gg),coplane_point_x(gg));
                D =[D;D1];
            end  
            
            %%
            center_depth = zeros(1,1);
%             if len(1)>=3 
%                 
%                 B1=inv(A'*A)*A'*D(:);%最小二乘公式
%                 B = [B;B1'];
            if len(1)==2 || len(1)==1 || len(1)>=3
                conn_new_l = bwconncomp(new_l);%查找连通区域
                centroid = regionprops(conn_new_l, 'Centroid');%连通区域像素坐标
                center = centroid.Centroid;
                center_v_norm = (round(center(1,2))-1)./(h-1);
                center_u_norm = (round(center(1,1))-1)./(w-1);
                center_depth(:) = predict_lay_depth(round(center(1,2)),round(center(1,1)));
                D = [D;center_depth];
                A = [A;[center_u_norm,center_v_norm,ones(1,1)]];
                B2=inv(A'*A)*A'*D(:);%最小二乘公式
                B = [B;B2'];
            elseif len(1)==0
                continue;
            end
            [B_row,B_col] = size(B); 
            if ~any(all(repmat(B(B_row,:),B_row-1,1)==B(1:B_row-1,:),2))%判断当前的参数是否与之前的参数相等，如果不等，则认为当前参数构成一个面
                rc = rc+1;
            end
%             [~,ia,~] = unique(B,'row','stable');
%             B = B(setdiff(1:size(B,1),ia),:);
            
            
            %%
            
            filled_map(:,:,rc) = 1./ (B(rc,1)*coord_x+B(rc,2)*coord_y+B(rc,3) + 1e-10);   
     
            visiable_area = visiable_area + imresize(new_l,sz,'nearest') * rc;
       
        end

        filled_map(filled_map<0) = inf;
        
        [~, sort_map] = sort(filled_map,3);  % extract visible regions according to the predicted depth maps, only for cuboid rooms
        m = ones(1,rc);
        layout_seg = sort_map(:,:,1);
        
        for n = 1:rc
            tmp_label = mode(layout_seg(visiable_area==n));%mode，输出最多的标签，最主要的标签和当前的标签是否一致
            if tmp_label ~= n && mode(visiable_area(layout_seg==n)) ~= n
                m(n) = m(n)  + 1;
                tmp_layer = sort_map(:,:,m(n));
                layout_seg(layout_seg==tmp_label) = tmp_layer(layout_seg==tmp_label);
            end
        end
       
        score(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]);
        tmp_map{pm} = filled_map;
        tmp_seg{pm} = layout_seg;
        tmp_rc{pm} = rc;
    end
    best_pm = find(score==max(score),1);
    filled_map = tmp_map{best_pm};
    layout_seg = tmp_seg{best_pm};
    rc = tmp_rc{best_pm};

    layout_edge = edge(layout_seg,'canny');
    layout_edge = imfilter(layout_edge, ones(3));
%     figure;imshow(layout_edge,[])
    img_edge = img;
    img_edge(:,:,1) = img_edge(:,:,1)+uint8(layout_edge*255) ;
    imshow(img_edge,[])
    
    layout_depth = zeros(size(layout_seg));
    for n = 1:rc
        tmp_depth_layer = filled_map(:,:,n);
        layout_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);  % corresponding depth map for layout
    end
    %%%%%
    new_img_name = sprintf('%04d',i);
    layout_dep_o = uint16(layout_depth*4000);
    imwrite(layout_dep_o,[layout_depth_o_path,num2str(new_img_name),'.png']);
    %%%%%
    [model, point] = gen_param_point(layout_depth, layout_seg);  % calculate coordinates of layout corners
    result{i}.point = point(:,1:3);
    result{i}.layout = layout_seg;
    catch ME
        disp("发生异常：");
        diary('cuowu_cluster_focal.txt');
        diary on;
        i
        diary off;
        count = count + 1;
        result{i}.point = result{i-1}.point;
        result{i}.layout = result{i-1}.layout;
        continue
    end 
end
 %% 计算误差值
save result(12-8-newcrf-onlypointdepth) result
%load result(8-29)

[ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc_mat(result, dataset);
  
    