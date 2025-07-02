clear;clc;

load /home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/test.mat
seg_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/layout_seg/';
layout_depth_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/layout_depth/';
img_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/image/';


t = dir([img_path, '*.jpg']);
gg = dir([layout_depth_path, '*.png']);
ss = dir([seg_path, '*.png']);

data = strtrim(string(data));

result = [];
sz = [480,640];
grid_x = 16;
grid_y = grid_x;
out_sz = [grid_y, grid_x]*8;
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);

threshold_ceil = 0.75;
threshold_conf = 0.6;
threshold_conf_mod = 0.3;
threshold_dice = 0.55;
threshold_fit = 0.1;
score_add_stitch = 0;

cmx = repmat(0:out_sz(2)-1,[out_sz(1),1])/(out_sz(2)-1);
cmy = repmat(0:out_sz(1)-1,[out_sz(2),1])'/(out_sz(1)-1);
uv1 = ones(out_sz(1),out_sz(2),3);
uv1(:,:,1) = cmx;
uv1(:,:,2) = cmy;
uv1 = reshape(uv1,[out_sz(1)*out_sz(2),3]);

for i = 1:length(data)
    
    i =365
    
    image_name = char(data(i));
    img = imread([img_path image_name '.jpg']);
    seg = imread([seg_path image_name '.png']);
    depth_gt = imread([layout_depth_path image_name '.png']);
    
    
    load(['/home/ubuntu/work/regiongrow/predict_param/interior0(6-4)/' image_name '.mat']);
    
    
    try
        pred_inv_depth = squeeze(output);
        pred_inv_depth = max(pred_inv_depth,0);
        pred_depth = 1 ./ (pred_inv_depth + 1e-10);
        valid = (pred_depth > 0.1) & (pred_depth < 30);

%         max_value = max(max(pred_depth));
%         min_value = min(min(pred_depth));
%         pred_depth1 = (pred_depth - min_value)./(max_value - min_value); 
%         pred_depth1 = imresize(pred_depth1,sz);
% 
%         color_min = 0; % 自定义颜色范围的最小值
%         color_max = 1.2; % 自定义颜色范围的最大值
%         colormap jet; % 使用 parula 颜色映射
%         imagesc(pred_depth1);
%         caxis([color_min, color_max]); axis off;
% %         set(gcf, 'Position', [100, 100, 1024, 1280]);%colorbar;
%         path1 = '/home/ubuntu/zmj/result_inter0/depth1/';
%         filname = [path1 num2str(i,'%04d') '.png'];
%         print(gcf,filname,'-dpng','-r300');
        
        
%         imshow(pred_depth1,[]);
%         colormap('turbo');
%         path1 = '/home/ubuntu/zmj/result_inter0/pred_depth/';
%         filname = [path1 num2str(i,'%04d') '.png'];
%         print(gcf,filname,'-dpng','-r300');
        
        plane = squeeze(output3);
        
%         weight = imresize(plane,sz);
%         weight(weight<0.0001)=0;
%         weight = weight*10;
%         wm = max(max(weight));
%         beishu = 255/wm;
%         
%         weight = weight*beishu;
%         vv = round(max(max(weight)));
% 
%         imshow(weight,[]);
%         colormap(hot(vv)); %colorbar
%         path2 = '/home/ubuntu/zmj/result_inter0/weight/';
%         filname = [path2 num2str(i,'%04d') '.png'];
%         print(gcf,filname,'-dpng','-r300');

        
        mask_raw = permute(squeeze(output2),[2,3,1]);
        mask_u = mask_raw(:,:,1:grid_x);
        mask_v = mask_raw(:,:,grid_x+1:end);
        
        loc_mat = 1 ./ ( 1 + exp(-squeeze(output1)));
        [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf, threshold_dice);
        [row,col] = find(plane_cent==1);
        
        if length(row) < 2
            [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf_mod, threshold_dice);
            [row,col] = find(plane_cent==1);
        end
        
        model = [];
        tmp_seg_mask = zeros(sz(1),sz(2),length(row));
        inv_pd = zeros(sz(1),sz(2),length(row));
        mask2cent = zeros(length(row),2);
        for f = 1:length(row)
            tmp_seg = mask_u(:,:,col(f)) .* mask_v(:,:,row(f));
            tmp_mask = tmp_seg .* plane .* valid;
            tmp_seg_mask(:,:,f) = imresize(tmp_seg,sz);
            cent_x = loc_coord(row(f), col(f), 1);
            ub_x = (col(f) + 0.5 - 1) / (grid_x - 1);
            lb_x = (col(f) - 0.5 - 1) / (grid_x - 1);
            mask2cent(f,1) = min(max(lb_x, cent_x), ub_x);
            cent_y = loc_coord(row(f), col(f), 2);
            ub_y = (row(f) + 0.5 - 1) / (grid_y - 1);
            lb_y = (row(f) - 0.5 - 1) / (grid_y - 1);
            mask2cent(f,2) = min(max(lb_y, cent_y), ub_y);
            
            vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]);
            sorted = sort(vw,'descend');
            thre = min(threshold_fit, sorted(grid_x * grid_x * 8));
            vw(vw<thre) = 0;
            x = uv1(vw>0,:);
            y = pred_inv_depth(:);
            y = y(vw>0,:);
            w = vw(vw>0);
            w = diag(w);
            tmp_param = (x' * w * x) \ x' * w * y;
            tmp_param = tmp_param';
            
            inv_pd(:,:,f) = tmp_param(1) * coord_x + tmp_param(2) * coord_y + tmp_param(3);
            model(f).face = f;
            model(f).params = tmp_param;
        end
        
        [~, seg_mask] = max(tmp_seg_mask,[],3);
        
        
%         r = zeros(size(seg_mask));
%         g = zeros(size(seg_mask));
%         b = zeros(size(seg_mask));
% 
%         cl =1;%赤红色
%         r(seg_mask ==cl) = 220;
%         g(seg_mask ==cl) = 20;
%         b(seg_mask ==cl) = 60;
% 
%         cl =2;%春天绿色
%         r(seg_mask ==cl) = 0;
%         g(seg_mask ==cl) = 255;
%         b(seg_mask ==cl) = 127;
% 
%         cl =3;%紫色
%         r(seg_mask ==cl) = 238;
%         g(seg_mask ==cl) = 130;
%         b(seg_mask ==cl) = 238;
% 
%         cl =4; %道奇蓝
%         r(seg_mask ==cl) = 30;
%         g(seg_mask ==cl) = 144;
%         b(seg_mask ==cl) = 255;
% 
%         cl =5;%深橙色
%         r(seg_mask ==cl) = 255;
%         g(seg_mask ==cl) = 140;
%         b(seg_mask ==cl) = 0;
% 
%         cl =6;%沙棕色
%         r(seg_mask ==cl) = 255;
%         g(seg_mask ==cl) = 215;
%         b(seg_mask ==cl) = 255;
% 
% 
%         cl =7; %金色
%         r(seg_mask ==cl) = 255;
%         g(seg_mask ==cl) = 215;
%         b(seg_mask ==cl) = 0;
% 
%         cl =8; %草绿色
%         r(seg_mask ==cl) = 0;
%         g(seg_mask ==cl) = 128;
%         b(seg_mask ==cl) = 0;
% 
%         cl =9; %紫罗兰
%         r(seg_mask ==cl) = 138;
%         g(seg_mask ==cl) = 43;
%         b(seg_mask ==cl) = 226;
%         
%         c = uint8(cat(3,r,g,b));
%         imshow(c,[]);
%         path = '/home/ubuntu/zmj/result_inter0/seg/';
%         imwrite(c,[path num2str(i,'%04d') '.png']);
        
        
        [~, sort_map] = sort(inv_pd,3);
        layout_seg = sort_map(:,:,end);
        
        invd_mat = zeros(length(row),length(row));
        for p = 1:length(row)
            for q = 1:length(row)
                invd_mat(p,q) = model(q).params(1) * mask2cent(p,1) + model(q).params(2) * mask2cent(p,2) + model(q).params(3);
            end
        end
     
        [~, md] = max(invd_mat);
        %         if isequal(md, 1:length(row))
        layout_inv_depth = zeros(size(layout_seg));
        for n = 1:length(row)
            tmp_depth_layer = inv_pd(:,:,n);
            layout_inv_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
        end
        score_max = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (1-isequal(md, 1:length(row))) * 0.2;
        layout_inv_depth_max = layout_inv_depth;
        layout_seg_max = layout_seg;
        
        %         else
        m = ones(1,length(row))*length(row);
        for n = 1:length(row)
            tmp_label = layout_seg(round(mask2cent(n,2)*(sz(1)-1))+1, round(mask2cent(n,1)*(sz(2)-1))+1);
            if tmp_label ~= n && mode(seg_mask(layout_seg==n)) ~= n
                m(n) = m(n) - 1;
                tmp_layer = sort_map(:,:,m(n));
                layout_seg(layout_seg==tmp_label) = tmp_layer(layout_seg==tmp_label);
            end
        end
        layout_inv_depth = zeros(size(layout_seg));
        for n = 1:length(row)
            tmp_depth_layer = inv_pd(:,:,n);
            layout_inv_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
        end
        score_layer = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.5;
        if score_layer > score_max
            layout_inv_depth_layer = layout_inv_depth;
            layout_seg_layer = layout_seg;
        else
            score_layer = score_max;
            layout_inv_depth_layer = layout_inv_depth_max;
            layout_seg_layer = layout_seg_max;
        end
        
%             depth_Z = 1./layout_inv_depth;
%             max_value1 = max(max(depth_Z));
%             min_value1 = min(min(depth_Z));
%             weight_preddepth = (depth_Z - min_value1)./(max_value1 - min_value1); 
%             
% % %             设置自定义的颜色范围
%             color_min = 0; % 自定义颜色范围的最小值
%             color_max = 1.2; % 自定义颜色范围的最大值
%             colormap jet; % 使用 parula 颜色映射
%             imagesc(weight_preddepth);
%             caxis([color_min, color_max]); axis off;
% %             set(gcf, 'Position', [100, 100, 1024, 1280]);%colorbar;
%             path1 = '/home/ubuntu/zmj/result_inter0/weight_depth1/';
%             filname = [path1 num2str(i,'%04d') '.png'];
%             print(gcf,filname,'-dpng','-r300');
%         
        
        
        model_new = [];
        ceil_inf = [];
        wall_inf = [];
        for n = 1:length(row)
            tp = model(n).params;
            tp = tp / norm(tp);
            if abs(tp(2)) > threshold_ceil
                ceil_inf = [ceil_inf; [mask2cent(n,:) model(n).params]];
            else
                wall_inf = [wall_inf; [mask2cent(n,:) model(n).params]];
            end
        end
        
        count = 0;
        if size(wall_inf,1) > 0
            wall_inf = sortrows(wall_inf,1);
            wall_left = wall_inf(1,3) * coord_x + wall_inf(1,4) * coord_y + wall_inf(1,5);
            left_cent = wall_inf(1,1:2);
            layout_inv_depth = wall_left;
            layout_seg = ones(sz);
            count = count + 1;
            model_new(count).face = 1;
            model_new(count).params = wall_inf(1,3:5);
            for n = 2:size(wall_inf,1)
                tmp_pid = wall_inf(n,3) * coord_x + wall_inf(n,4) * coord_y + wall_inf(n,5);
                seg1 = tmp_pid >= wall_left;
                seg2 = tmp_pid < wall_left;
                left_cent_l = wall_inf(n-1,3) * left_cent(1) + wall_inf(n-1,4) * left_cent(2) + wall_inf(n-1,5);
                left_cent_r = wall_inf(n,3) * left_cent(1)  + wall_inf(n,4) * left_cent(2) + wall_inf(n,5);
                count = count + 1;
                if left_cent_l > left_cent_r
                    layout_inv_depth(seg1) = tmp_pid(seg1);
                    layout_seg(seg1) = count;
                else
                    layout_inv_depth(seg2) = tmp_pid(seg2);
                    layout_seg(seg2) = count;
                end
                wall_left = tmp_pid;
                left_cent = wall_inf(n,1:2);
                model_new(count).face = count;
                model_new(count).params = wall_inf(n,3:5);
            end
        end
        
        for n = 1:size(ceil_inf,1)
            tmp_pid = ceil_inf(n,3) * coord_x + ceil_inf(n,4) * coord_y + ceil_inf(n,5);
            layout_inv_depth = max(layout_inv_depth, tmp_pid);
            count = count + 1;
            layout_seg(layout_inv_depth == tmp_pid) = count;
            model_new(count).face = count;
            model_new(count).params = ceil_inf(n,3:5);
        end
        score_stitch = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.5 + score_add_stitch - (numel(unique(layout_seg))~=length(row)) * 0.3;
        layout_inv_depth_stitch = layout_inv_depth;
        layout_seg_stitch = layout_seg;
        
        if score_layer >= score_stitch
            score = score_layer;
            layout_seg = layout_seg_layer;
            layout_inv_depth = layout_inv_depth_layer;
        else
            score = score_stitch;
            layout_seg = layout_seg_stitch;
            layout_inv_depth = layout_inv_depth_stitch;
            model = model_new;
        end
        %         end
        
    catch
        score = 0;
    end
    
    try
        
        pred_inv_depth = fliplr(squeeze(output_flip));
        pred_inv_depth = max(pred_inv_depth,0);
        pred_depth = 1 ./ (pred_inv_depth + 1e-10);
        valid = (pred_depth > 0.1) & (pred_depth < 30);
        
        
        plane = fliplr(squeeze(output3_flip));
        
        mask_raw = permute(squeeze(output2_flip),[2,3,1]);
        mask_u = mask_raw(:,:,grid_x:-1:1);
        mask_v = mask_raw(:,:,grid_x+1:end);
        
        loc_mat = fliplr(1 ./ ( 1 + exp(-squeeze(output1_flip))));
        [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf, threshold_dice);
        [row,col] = find(plane_cent==1);
        
        if length(row) < 2
            [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf_mod, threshold_dice);
            [row,col] = find(plane_cent==1);
        end
        
        model_flip = [];
        tmp_seg_mask = zeros(sz(1),sz(2),length(row));
        inv_pd = zeros(sz(1),sz(2),length(row));
        mask2cent = zeros(length(row),2);
        for f = 1:length(row)
            tmp_seg = fliplr(mask_u(:,:,col(f)) .* mask_v(:,:,row(f)));
            tmp_mask = tmp_seg .* plane .* valid;
            tmp_seg_mask(:,:,f) = imresize(tmp_seg,sz);
            cent_x = loc_coord(row(f), col(f), 1);
            ub_x = (col(f) + 0.5 - 1) / (grid_x - 1);
            lb_x = (col(f) - 0.5 - 1) / (grid_x - 1);
            mask2cent(f,1) = min(max(lb_x, cent_x), ub_x);
            cent_y = loc_coord(row(f), col(f), 2);
            ub_y = (row(f) + 0.5 - 1) / (grid_y - 1);
            lb_y = (row(f) - 0.5 - 1) / (grid_y - 1);
            mask2cent(f,2) = min(max(lb_y, cent_y), ub_y);
            
            vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]);
            sorted = sort(vw,'descend');
            thre = min(threshold_fit, sorted(grid_x * grid_x * 8));
            vw(vw<thre) = 0;
            x = uv1(vw>0,:);
            y = pred_inv_depth(:);
            y = y(vw>0,:);
            w = vw(vw>0);
            w = diag(w);
            tmp_param = (x' * w * x) \ x' * w * y;
            tmp_param = tmp_param';
            
            inv_pd(:,:,f) = tmp_param(1) * coord_x + tmp_param(2) * coord_y + tmp_param(3);
            model_flip(f).face = f;
            model_flip(f).params = tmp_param;
        end
        
        [~, seg_mask_flip] = max(tmp_seg_mask,[],3);
        
        [~, sort_map] = sort(inv_pd,3);
        layout_seg_flip = sort_map(:,:,end);
        
        invd_mat = zeros(length(row),length(row));
        for p = 1:length(row)
            for q = 1:length(row)
                invd_mat(p,q) = model_flip(q).params(1) * mask2cent(p,1) + model_flip(q).params(2) * mask2cent(p,2) + model_flip(q).params(3);
            end
        end
        
        [~, md] = max(invd_mat);
        %         if isequal(md, 1:length(row))
        layout_inv_depth_flip = zeros(size(layout_seg_flip));
        for n = 1:length(row)
            tmp_depth_layer = inv_pd(:,:,n);
            layout_inv_depth_flip(layout_seg_flip==n) = tmp_depth_layer(layout_seg_flip==n);
        end
        %             score_flip = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1);
        score_flip_max = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1) - (1-isequal(md, 1:length(row))) * 0.2;
        layout_inv_depth_flip_max = layout_inv_depth_flip;
        layout_seg_flip_max = layout_seg_flip;
        
        %         else
        m = ones(1,length(row))*length(row);
        for n = 1:length(row)
            tmp_label = layout_seg_flip(round(mask2cent(n,2)*(sz(1)-1))+1, round(mask2cent(n,1)*(sz(2)-1))+1);
            if tmp_label ~= n && mode(seg_mask_flip(layout_seg_flip==n)) ~= n
                m(n) = m(n) - 1;
                tmp_layer = sort_map(:,:,m(n));
                layout_seg_flip(layout_seg_flip==tmp_label) = tmp_layer(layout_seg_flip==tmp_label);
            end
        end
        layout_inv_depth_flip = zeros(size(layout_seg_flip));
        for n = 1:length(row)
            tmp_depth_layer = inv_pd(:,:,n);
            layout_inv_depth_flip(layout_seg_flip==n) = tmp_depth_layer(layout_seg_flip==n);
        end
        
        score_flip_layer = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1) - (min(layout_inv_depth_flip(:)) < 0) * 0.5;
        layout_inv_depth_flip_layer = layout_inv_depth_flip;
        layout_seg_flip_layer = layout_seg_flip;
        
        if score_flip_layer > score_flip_max
            layout_inv_depth_flip_layer = layout_inv_depth_flip;
            layout_seg_flip_layer = layout_seg_flip;
        else
            score_flip_layer = score_flip_max;
            layout_inv_depth_flip_layer = layout_inv_depth_flip_max;
            layout_seg_flip_layer = layout_seg_flip_max;
        end
        
        model_flip_new = [];
        ceil_inf = [];
        wall_inf = [];
        for n = 1:length(row)
            tp = model_flip(n).params;
            tp = tp / norm(tp);
            if abs(tp(2)) > threshold_ceil
                ceil_inf = [ceil_inf; [mask2cent(n,:) model_flip(n).params]];
            else
                wall_inf = [wall_inf; [mask2cent(n,:) model_flip(n).params]];
            end
        end
        
        count = 0;
        if size(wall_inf,1) > 0
            wall_inf = sortrows(wall_inf,1);
            wall_left = wall_inf(1,3) * coord_x + wall_inf(1,4) * coord_y + wall_inf(1,5);
            left_cent = wall_inf(1,1:2);
            layout_inv_depth_flip = wall_left;
            layout_seg_flip = ones(sz);
            count = count + 1;
            model_flip_new(count).face = 1;
            model_flip_new(count).params = wall_inf(1,3:5);
            for n = 2:size(wall_inf,1)
                tmp_pid = wall_inf(n,3) * coord_x + wall_inf(n,4) * coord_y + wall_inf(n,5);
                seg1 = tmp_pid >= wall_left;
                seg2 = tmp_pid < wall_left;
                left_cent_l = wall_inf(n-1,3) * left_cent(1) + wall_inf(n-1,4) * left_cent(2) + wall_inf(n-1,5);
                left_cent_r = wall_inf(n,3) * left_cent(1)  + wall_inf(n,4) * left_cent(2) + wall_inf(n,5);
                count = count + 1;
                if left_cent_l > left_cent_r
                    layout_inv_depth_flip(seg1) = tmp_pid(seg1);
                    layout_seg_flip(seg1) = count;
                else
                    layout_inv_depth_flip(seg2) = tmp_pid(seg2);
                    layout_seg_flip(seg2) = count;
                end
                wall_left = tmp_pid;
                left_cent = wall_inf(n,1:2);
                model_flip_new(count).face = count;
                model_flip_new(count).params = wall_inf(n,3:5);
            end
        end
        
        for n = 1:size(ceil_inf,1)
            tmp_pid = ceil_inf(n,3) * coord_x + ceil_inf(n,4) * coord_y + ceil_inf(n,5);
            layout_inv_depth_flip = max(layout_inv_depth_flip, tmp_pid);
            count = count + 1;
            layout_seg_flip(layout_inv_depth_flip == tmp_pid) = count;
            model_flip_new(count).face = count;
            model_flip_new(count).params = ceil_inf(n,3:5);
        end
        score_flip_stitch = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1) - (min(layout_inv_depth_flip(:)) < 0) * 0.5 + score_add_stitch - (numel(unique(layout_seg_flip))~=length(row)) * 0.3;
        layout_inv_depth_flip_stitch = layout_inv_depth_flip;
        layout_seg_flip_stitch = layout_seg_flip;
        
        if score_flip_layer >= score_flip_stitch
            score_flip = score_flip_layer;
            layout_seg_flip = layout_seg_flip_layer;
            layout_inv_depth_flip = layout_inv_depth_flip_layer;
        else
            score_flip = score_flip_stitch;
            layout_seg_flip = layout_seg_flip_stitch;
            layout_inv_depth_flip = layout_inv_depth_flip_stitch;
            model_flip = model_flip_new;
        end
        %         end
        
    catch
        score_flip = 0;
    end
    
    if score_flip > score
        layout_inv_depth = layout_inv_depth_flip;
        layout_seg = layout_seg_flip;
        model = model_flip;
    end
    
    layout_depth = 1./layout_inv_depth;
% 
%     max_value1 = max(max(layout_depth));
%     min_value1 = min(min(layout_depth));
%     pinjie_preddepth = (layout_depth - min_value1)./(max_value1 - min_value1); 
%     
%     color_min = 0; % 自定义颜色范围的最小值
%     color_max = 1.2; % 自定义颜色范围的最大值
%     colormap jet; % 使用 parula 颜色映射
%     imagesc(pinjie_preddepth);
%     caxis([color_min, color_max]); axis off;
% %         set(gcf, 'Position', [100, 100, 1024, 1280]);%colorbar;
%     path_pinjie_preddepth = '/home/ubuntu/zmj/result_inter0/pinjie_depth/';
%     filname = [path_pinjie_preddepth num2str(i,'%04d') '.png'];
%     print(gcf,filname,'-dpng','-r300');
%     
%     layout_seg = imresize(layout_seg,[320,400]);
%     layout_edge = edge(layout_seg,'canny');
%     se = strel('disk',4);
%     layout_edge = imdilate(layout_edge,se);  
%     img = imresize(img,[320,400]);
%     img_edge = img;
%     img_edge(:,:,2) = img_edge(:,:,2)+uint8(layout_edge*255) ;
%     seg = imresize(seg,[320,400]);
%     gt_edge = edge(seg,'canny');
%     gt_edge = imdilate(gt_edge,se);
%     img_edge(:,:,1) = img_edge(:,:,1)+uint8(gt_edge*255);
%     imshow(img_edge,[]);
%     path = '/home/ubuntu/zmj/result_inter0/layout/';
%     imwrite(img_edge,[path num2str(i,'%04d') '.png']);
    
    
    
    
    try
        point = gen_param_point2(layout_inv_depth, layout_seg, model);
    catch
        [model, point] = gen_param_point(layout_depth, layout_seg);
    end
    point_ex(1,:) = [0, 0, layout_depth(1,1)];
    point_ex(2,:) = [0, 480, layout_depth(480,1)];
    point_ex(3,:) = [640, 0, layout_depth(1,640)];
    point_ex(4,:) = [640, 480, layout_depth(480,640)];
    point = [point; point_ex];
    
    result{i}.point = point;
    result{i}.layout = layout_seg;
    
    %     catch
    %         result{i}.point = result{i-1}.point;
    %         result{i}.layout = result{i-1}.layout;
    %     end
    
end


% save result_interior result
