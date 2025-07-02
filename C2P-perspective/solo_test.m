clear;clc;

load /home/ps/data/W/matterport_layout/testing/testing/testing.mat
% load /Users/WQ0627/Desktop/solo/matterport_layout/testing/testing/testing.mat

dataset = data;

for j = 1:length(dataset)
    sz(1)=1024;
    sz(2)=1280;   
end

seg_path = '/home/ps/data/W/matterport_layout/testing/testing/layout_seg/';
layout_depth_path =  '/home/ps/data/W/matterport_layout/testing/testing/layout_depth_order/';
img_path = '/home/ps/data/W/matterport_layout/testing/testing/image/';
% seg_path = '/Users/WQ0627/Desktop/solo/matterport_layout/testing/testing/layout_seg/';
% layout_depth_path =  '/Users/WQ0627/Desktop/solo/matterport_layout/testing/testing/layout_depth_o/';
% img_path = '/Users/WQ0627/Desktop/solo/matterport_layout/testing/testing/image/';

result = {};
sz = [1024,1280]; %图像大小
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1); 
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
%coord_hor = coord_x(1,:); 
grid_x = 14;
grid_y = 14;

threshold_conf = 0.8;
threshold_cos = 0.3;

threshold_conf_wall = 0.8;
threshold_cos_wall = 0.6;

for i = 2:length(dataset)
   
    i
    try

    img = imread([img_path num2str(i,'%04d') '.jpg']);
    seg = imread([seg_path num2str(i,'%04d') '.png']);

%     load(['/home/ps/data/W/solo/predict_param(fbl)/param_(10-9)/output0/' num2str(i-1) '.mat']);
%     load(['/home/ps/data/W/solo/predict_param(fbl)/param_(10-9)/output1/' num2str(i-1) '.mat']);
    load(['/home/ps/data/W/solo_3/predict_param(crfs)/param_(12.28.2)/output0/' num2str(i-1) '.mat']);
    load(['/home/ps/data/W/solo_3/predict_param(crfs)/param_(12.28.2)/output1/' num2str(i-1) '.mat']);
    dataset(i).resolution=sz;
   
    
    param_mat = permute(squeeze(output1),[2,3,1]);   
    loc_mat0 = permute(squeeze(output0),[2,3,1]);   % 3x14x14 ---> 14x14x3
    rel_loc = permute(squeeze(output0(:,1:2,:,:)),[2,3,1]);  %偏差
%     loc_rc = squeeze(output0(:,3,:,:));
%     loc_rc2 = sigmoid(loc_rc);
    loc_ceil1 = squeeze(output0(:,4,:,:));
%     loc_ceil2  = sigmoid(loc_ceil1);
    loc_floor1 = squeeze(output0(:,5,:,:));
%     loc_floor2 = sigmoid(loc_floor1);
    loc_wall1 = squeeze(output0(:,6,:,:));
%     loc_wall2  = sigmoid(loc_wall1);

%     loc_ceil = double(loc_ceil2> 0.9);
%     loc_floor = double(loc_floor2 > 0.9);
%     loc_wall = double(loc_wall2 > 0.9);
    loc_ceil = nms_ceil(loc_ceil1 , param_mat, threshold_conf_wall, threshold_cos_wall);
    loc_floor = nms_floor(loc_floor1 , param_mat, threshold_conf_wall, threshold_cos_wall);
    loc_wall = nms_wall(loc_wall1 , param_mat, threshold_conf_wall, threshold_cos_wall);
    
    loc_mat = nms_v1(loc_mat0(:,:,3) , param_mat, threshold_conf, threshold_cos);
    

    [row,col] = find(loc_mat==1); %从 loc_mat 中找到 1 的位置坐标
    model = [];   
    ceil_param = [];
    ceil_loc = [];  
    wall_param = [];
    wall_loc = [];
    floor_param = [];
    floor_loc = [];
        
    all_f = zeros(sz(1),sz(2),length(row));
    invd_mat = zeros(length(row),length(row));
    for f = 1:length(row)
        tmp_param = squeeze(param_mat(row(f),col(f),:));
        all_f(:,:,f) = (tmp_param(1) * coord_x + tmp_param(2) * coord_y + tmp_param(3)) * tmp_param(4);
        for g = 1:length(row)
            tmp_offset = rel_loc(row(g),col(g),:);
            tmp_coord(1) = (col(g) - tmp_offset(1)) / (grid_x - 1);
            tmp_coord(2) = (row(g) - tmp_offset(2)) / (grid_y - 1);%guiyihuahou 
            invd_mat(f,g) = (tmp_param(1) * tmp_coord(1) + tmp_param(2) * tmp_coord(2) + tmp_param(3)) * tmp_param(4); 
        end
    end

    [~,max_invd_mat1]= max(invd_mat, [], 1);
    [~,max_invd_mat2]= max(invd_mat, [], 2);
    if (isequal(max_invd_mat1, (1:length(row))))||(isequal(max_invd_mat2, (1:length(row))'))


         [f_value,f_lable] = sort(all_f,3);
         layout_inv_depth = f_value(:,:,end);
         layout_seg = f_lable(:,:,end);
         layout = 1 ./ layout_inv_depth;

         point = gen_param_point(layout, layout_seg);   

    else
        
         [row1,col1] = find(loc_ceil==1);
         [row2,col2] = find(loc_floor==1);
         [row3,col3] = find(loc_wall==1);
         for f = 1:length(row3)
             wall_loc = [wall_loc;[row3(f),col3(f)]];
             wall_param = [wall_param;squeeze(param_mat(row3(f),col3(f),:))'];
         end
         for ff = 1:length(row1)
              ceil_loc = [ceil_loc;[row1(ff),col1(ff)]];
              ceil_param = [ceil_param;squeeze(param_mat(row1(ff),col1(ff),:))'];
         end
         for fff = 1:length(row2)
              floor_loc = [floor_loc;[row2(fff),col2(fff)]];
              floor_param = [floor_param;squeeze(param_mat(row2(fff),col2(fff),:))'];
         end

        ceil1 = [ceil_loc ceil_param];
        floor  = [floor_loc floor_param];
        if  isempty(ceil1) && ~ isempty(floor)
            ceil = floor;
        elseif  ~ isempty(ceil1) && isempty(floor)
            ceil = ceil1; 
        elseif ~ isempty(ceil1) && ~ isempty(floor)
            ceil = cat(1, ceil1, floor);
        end
        wall = [wall_loc wall_param];
    
        if size(wall,1) >= 1 %墙面数量＞1时
            
            wall = sortrows(wall,2); %将墙面根据第2列元素进行排序
            wall_left = (wall(1,3) * coord_x + wall(1,4) * coord_y + wall(1,5)) * wall(1,6);
%             left_centx = round((wall(1,2)-1) / (grid_x-1) * (sz(2)-1))+1;
%             left_centy = round((wall(1,1)-1) / (grid_y-1) * (sz(1)-1))+1;
            layout_inv_depth = wall_left;
            layout_seg = ones(sz); %给定一个大小为sz的全1矩阵
            count = 1;
            model(count).face = 1; %表示这是第一个墙面
            model(count).params = [wall(1,3), wall(1,4), wall(1,5)]* wall(1,6);
            

            for m = 2:size(wall,1) %处理第一个墙面外的其他面
%                    left_centx = round((wall(1,2)-1) / (grid_x-1) * (sz(2)-1))+1;
%                    left_centy = round((wall(1,1)-1) / (grid_y-1) * (sz(1)-1))+1;
              
                   left_centx = round((wall(m-1,2)-1) / (grid_x-1) * (sz(2)-1))+1;
                   left_centy = round((wall(m-1,1)-1) / (grid_y-1) * (sz(1)-1))+1;
                   tmp_pid = (wall(m,3) * coord_x + wall(m,4) * coord_y + wall(m,5)) * wall(m,6);
                   seg1 = tmp_pid >= wall_left;
                   seg2 = tmp_pid < wall_left;
                   left_inv_dep = wall_left(left_centy,left_centx);
                   tmp_inv_dep = tmp_pid(left_centy,left_centx);
                   count = count + 1;
                   if left_inv_dep < tmp_inv_dep
                        layout_inv_depth(seg2) = tmp_pid(seg2);
                        layout_seg(seg2) = count;
       
                   else
                        layout_inv_depth(seg1) = tmp_pid(seg1);
                        layout_seg(seg1) = count;
                   end
                   wall_left = tmp_pid;
%                    left_centx = round((wall(m,2)-1) / (grid_x-1) * (sz(2)-1))+1;
%                    left_centy = round((wall(m,1)-1) / (grid_y-1) * (sz(1)-1))+1;
                   model(count).face = count;
                   model(count).params = [wall(m,3), wall(m,4), wall(m,5)]*wall(m,6);
            end
 
            for f = 1:size(ceil,1)
                
                tmp_pid = (ceil(f,3) * coord_x + ceil(f,4) * coord_y + ceil(f,5))* ceil(f,6);
                layout_inv_depth = max(layout_inv_depth, tmp_pid);
                count = count + 1;
                layout_seg(layout_inv_depth == tmp_pid) = count;
                
                model(count).face = count;
                model(count).params = [ceil(f,3), ceil(f,4), ceil(f,5)]*ceil(f,6);
                
            end
        
           
        end
%         layout_inv_depth = double(layout_inv_depth);
%         layout_depth = 1 ./ layout_inv_depth;
        point = gen_param_point2(layout_inv_depth, layout_seg, model);
        
     end
     layout_edge = edge(layout_seg,'canny');
     layout_edge = imfilter(layout_edge,ones(3));
     img_edge = img;
     img_edge(:,:,1) = img_edge(:,:,1)+uint8(layout_edge*255) ;
     imshow(img_edge,[])
    

        result{i}.point = point(:,1:3);
        result{i}.layout = layout_seg;
     catch
        result{i}.point = result{i-1}.point;
        result{i}.layout = result{i-1}.layout;
        continue
     end  

          
end

save result_solo_gt(12.28.2) result

[ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc(result, dataset);

