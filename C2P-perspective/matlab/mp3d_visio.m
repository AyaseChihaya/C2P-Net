clear;clc;

% load gt_result

% load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing.mat
% seg_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/layout_seg/';
% layout_depth_path =  '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/layout_depth/';
img_path = '/home/ps/data/Z/Pano_room1/dataset_image/image/';
data_path = '/home/ps/data/Z/Pano_room1/predict_param/test_image(9-12)/';
d = dir([data_path, '*.mat']);

t = dir([img_path, '*.png']);
% g = dir([layout_depth_path, '*.png']);
% seg = dir([seg_path, '*.png']);

result = [];
sz = [1024,1280];
grid_x = 16;
grid_y = grid_x;
out_sz = [128, 128];
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1); 
% coord_hor = coord_x(1,:);
threshold_ceil = 0.75;
threshold_conf = 0.4;
threshold_dice = 0.5;
threshold_fit = 0.1;
thre_sort = 8;
score_add_stitch = 0;
cmx = repmat(0:out_sz(2)-1,[out_sz(1),1])/(out_sz(2)-1);
cmy = repmat(0:out_sz(1)-1,[out_sz(2),1])'/(out_sz(1)-1);

% cmx = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
% cmy = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);

uv1 = ones(out_sz(1),out_sz(2),3);
uv1(:,:,1) = cmx;
uv1(:,:,2) = cmy;
uv1 = reshape(uv1,[out_sz(1)*out_sz(2),3]);

% for i = 1:200
for i = 1:length(d)
    
    i 
    
    img = imread([img_path t(i).name]);
%     seg = imread([seg_path data(i).layout_seg]);
%     depth_gt = imread([layout_depth_path data(i).layout_depth]);
    load([data_path d(i).name]);
    
%     load(['/home/ps/data/Z/Pano_room1/predict_param/test_image(9-12)/' image_name '.mat']);
    
    
   try
        pred_inv_depth = squeeze(output);
        pred_inv_depth = max(pred_inv_depth,0);
        %         valid = pred_inv_depth > 0;
        pred_depth = 1 ./ (pred_inv_depth + 1e-10);
        valid = (pred_depth > 0.1) & (pred_depth < 30);

        plane = squeeze(output3); 
        weight = imresize(plane,sz);
 
        mask_raw = permute(squeeze(output2),[2,3,1]);
        mask_u = mask_raw(:,:,1:grid_x);
        mask_v = mask_raw(:,:,grid_x+1:end);
        
        loc_mat = 1 ./ ( 1 + exp(-squeeze(output1)));
        [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf, threshold_dice);
        [row,col] = find(plane_cent==1);
        
        conf = sort(loc_mat(:),'descend');
        conf_ind = 1;
        while length(row) <2
            conf_ind = conf_ind + 1;
            threshold_conf_mod = conf(conf_ind);
            [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf_mod, threshold_dice);
            [row,col] = find(plane_cent==1);
        end
        
        model = [];
        tmp_seg_mask = zeros(sz(1),sz(2),length(row));
        inv_pd = zeros(sz(1),sz(2),length(row));
        mask2cent = zeros(length(row),2);
        for f = 1:length(row)
            tmp_seg = mask_u(:,:,col(f)) .* mask_v(:,:,row(f));
%             imwrite(tmp_seg,'1.png')
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
            thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
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
        

        
        
        [~, sort_map] = sort(inv_pd,3);
        
        
        layout_seg = sort_map(:,:,end); 
        invd_mat = zeros(length(row),length(row));
        for p = 1:length(row)
            for q = 1:length(row)
                invd_mat(p,q) = model(q).params(1) * mask2cent(p,1) + model(q).params(2) * mask2cent(p,2) + model(q).params(3);
            end
        end
        
        [~, md] = max(invd_mat);
        if isequal(md, 1:length(row))
            layout_inv_depth = zeros(size(layout_seg));
            for n = 1:length(row)
                tmp_depth_layer = inv_pd(:,:,n);
                layout_inv_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
            end
            score = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1);
        else
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
            score_layer = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.1;
            layout_inv_depth_layer = layout_inv_depth;
            layout_seg_layer = layout_seg;
 
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
            [gx,gy] = gradient(layout_inv_depth);
            grad_abs = abs(gx) + abs(gy);
%             grad_stitch(i) = max(grad_abs(:))>0.005;
            score_stitch = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.1 + score_add_stitch - (max(grad_abs(:))>0.005);
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
        end
        
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
        
        conf = sort(loc_mat(:),'descend');
        conf_ind = 1;
        while length(row) <2
            conf_ind = conf_ind + 1;
            threshold_conf_mod = conf(conf_ind);
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
            thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
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
        if isequal(md, 1:length(row))
            layout_inv_depth_flip = zeros(size(layout_seg_flip));
            for n = 1:length(row)
                tmp_depth_layer = inv_pd(:,:,n);
                layout_inv_depth_flip(layout_seg_flip==n) = tmp_depth_layer(layout_seg_flip==n);
            end
            score_flip = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1);
        else
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
            
            score_flip_layer = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1) - (min(layout_inv_depth_flip(:)) < 0) * 0.1;
            layout_inv_depth_flip_layer = layout_inv_depth_flip;
            layout_seg_flip_layer = layout_seg_flip;
            
            
            
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
            [gx,gy] = gradient(layout_inv_depth_flip);
            grad_abs = abs(gx) + abs(gy);
            %             grad_stitch(i) = max(grad_abs(:))>0.005;
            score_flip_stitch = pixelwiseAccuracy(seg_mask_flip, layout_seg_flip, sz) - (length(unique(layout_seg_flip))==1) - (min(layout_inv_depth_flip(:)) < 0) * 0.1 + score_add_stitch - (max(grad_abs(:))>0.005);
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
        end
        
    catch
        score_flip = 0;
    end
    
    if score_flip > score
        layout_inv_depth = layout_inv_depth_flip;
        layout_seg = layout_seg_flip;
        model = model_flip;
    end
    
    layout_depth = 1./layout_inv_depth;



%     try
%         try
%             point = gen_param_point2(layout_inv_depth, layout_seg, model);
%         catch
%             [model, point] = gen_param_point(layout_depth, layout_seg);
%             
%         end
%         result{i}.point = point;
%         result{i}.layout = layout_seg;
%         
%     catch
%         result{i}.point = result{i-1}.point;
%         result{i}.layout = result{i-1}.layout;
%     end
    
end
    
    
% save result_stitch result
