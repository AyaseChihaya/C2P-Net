clear;
clc;

%%%img_path;
%%%data_path;

score_add_stitch = 0;
threshold_fit = 0.1;
grid_x = 16;
grid_y = 16;
thre_sort = 4;
out_sz = [128, 256];
threshold_conf = 0.4;
threshold_dice = 0.5;


h=128;
w=256;

for j = 1:length(d)
    sz(1)=128;
    sz(2)=256;
end

for i = 1:length(d)
    i
    

    scale = 0.5;
    img = imread([img_path t(i).name]);
    scale1 = 0.25;
    img = imresize(img,scale1); 

    coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
    coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
    theta = pi * (2 * coord_x - 1);
    phi = pi * coord_y;

    sin_theta_cos_phi = sin(phi) .* cos(theta);
    sin_theta_sin_phi = sin(phi) .* sin(theta);
    cos_theta = cos(phi);

    X = sin_theta_cos_phi;
    Y = sin_theta_sin_phi;
    Z = cos_theta;

    smallSize = [size(img,1),size(img,2)];
    uv1 = zeros(smallSize(1),smallSize(2),3);
    uv1(:,:,1) = sin_theta_cos_phi;
    uv1(:,:,2) = sin_theta_sin_phi;
    uv1(:,:,3) = cos_theta; 
    uv1 = reshape(uv1,[smallSize(1)*smallSize(2),3]);

    try
        depth_map = squeeze(inv_output_depth(1,1,:,:));
        valid = (depth_map > 0.1) & (depth_map < 30);
        layout_depth = 1./depth_map;
        layoutseg = round(squeeze(segmentation(1,:,:,:)));
        layout_map = sum(layoutseg,1);
        layout_map = squeeze(layout_map);
        weight = squeeze(weight(1,1,:,:));
    
    
        loc_mat = (1 ./ ( 1 + exp(-squeeze(center_map))))';
        [plane_cent, loc_coord] = nms_v2(loc_mat, mask_raw,coord_x,coord_y, threshold_conf, threshold_dice);
        [row,col] = find(plane_cent==1);

        wall_center = col;
        wall_point = zeros(length(wall_center),2);
        for w = 1:length(wall_center)
            wall_point(w,:) = [(wall_center(w)-1)*4,65];
        end
        if wall_point(1,1)==0
            wall_point(1,1)=32;
        end
        ceil_point = [128,13];   
        floor_point = [128,115];
        
        wall_y = wall_point(:,2);
        wall_x = wall_point(:,1);
        ceil_y = ceil_point(:,2);
        ceil_x = ceil_point(:,1);
        floor_y = floor_point(:,2);
        floor_x = floor_point(:,1);
    
        tmp_seg_mask = zeros(sz(1),sz(2),length(wall_center)+2);
        inv_pd = zeros(sz(1),sz(2),length(wall_center)+2);  
        wall_param = zeros(length(wall_center)-1,3);
        for p = 1:length(wall_center)-1
            if p == 1
                first_seg = layoutseg(wall_center(p),:,:); 
                last_seg = layoutseg(wall_center(length(wall_center)),:,:);
                tmp_seg = [first_seg;last_seg];
                tmp_seg = sum(tmp_seg,1);
                tmp_seg = double(squeeze(tmp_seg(1,:,:)));
                tmp_mask = tmp_seg .* weight .* valid;
                tmp_seg_mask(:,:,p) = imresize(tmp_seg,sz);
                vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]);
                sorted = sort(vw,'descend');
                thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
                vw(vw<thre) = 0;
                x = uv1(vw>0,:);
                y = layout_depth(:);
                y = y(vw>0,:);
                w = vw(vw>0);
                w = diag(w);
                wall_param_solo = (x' * w * x) \ x' * w * y;
                wall_param_solo = wall_param_solo';
                wall_param(p,:) = wall_param_solo;% pqr
                inv_pd(:,:,p) = wall_param_solo(1) * X + wall_param_solo(2) * Y + wall_param_solo(3) * Z;      
            else
                tmp_seg = layoutseg(wall_center(p),:,:); 
                tmp_seg = double(squeeze(tmp_seg(1,:,:)));
                tmp_mask = tmp_seg .* weight .* valid;
                tmp_seg_mask(:,:,p) = imresize(tmp_seg,sz);
                vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]); 
                sorted = sort(vw,'descend');
                thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
                vw(vw<thre) = 0;
                x = uv1(vw>0,:);
                y = layout_depth(:);
                y = y(vw>0,:);
                w = vw(vw>0);
                w = diag(w);
                wall_param_solo = (x' * w * x) \ x' * w * y;% pqr
                wall_param_solo = wall_param_solo';
                wall_param(p,:) = wall_param_solo;% pqr
                inv_pd(:,:,p) = wall_param_solo(1) * X + wall_param_solo(2) * Y + wall_param_solo(3) * Z;      
            end
        end    
        first_param = wall_param(1,:);
        wall_param = [wall_param;first_param];
    
        ceiltmp_seg = layoutseg(65,:,:); 
        ceiltmp_seg = double(squeeze(ceiltmp_seg(1,:,:)));
        ceiltmp_mask = ceiltmp_seg .* weight .* valid;
        tmp_seg_mask(:,:,length(wall_center)+1) = imresize(ceiltmp_seg,sz);
        ceilvw = reshape(ceiltmp_seg,[out_sz(1)*out_sz(2),1]);    
        ceilsorted = sort(ceilvw,'descend');
        thre_ceil = min(threshold_fit, ceilsorted(grid_x * grid_x * thre_sort));
        ceilvw(ceilvw<thre_ceil) = 0;
        ceilx = uv1(ceilvw>0,:);
        ceily = layout_depth(:);
        ceily = ceily(ceilvw>0,:);
        ceilw = weight(ceilvw>0);
        ceilw = diag(ceilw);
        ceil_param = (ceilx' * ceilw * ceilx) \ ceilx' * ceilw * ceily;
        ceil_param = ceil_param';
        inv_pd(:,:,length(wall_center)+1) = ceil_param(1) * X + ceil_param(2) * Y + ceil_param(3) * Z;      
    
        floortmp_seg = layoutseg(66,:,:); 
        floortmp_seg = double(squeeze(floortmp_seg(1,:,:)));
        floortmp_mask = floortmp_seg .* weight .* valid;
        tmp_seg_mask(:,:,length(wall_center)+2) = imresize(floortmp_seg,sz);
        floorvw = reshape(floortmp_seg,[out_sz(1)*out_sz(2),1]);     
        floorsorted = sort(floorvw,'descend');
        thre_floor = min(threshold_fit, floorsorted(grid_x * grid_x * thre_sort));
        floorvw(floorvw<thre_floor) = 0;
        floorx = uv1(floorvw>0,:);
        floory = layout_depth(:);
        floory = floory(floorvw>0,:);
        floorw = weight(floorvw>0);
        floorw = diag(floorw);
        floor_param = (floorx' * floorw * floorx) \ floorx' * floorw * floory;
        floor_param = floor_param';
        inv_pd(:,:,length(wall_center)+2) = floor_param(1) * X + floor_param(2) * Y + floor_param(3) * Z;      
    
        [~, seg_mask] = max(tmp_seg_mask,[],3);       
        [~, sort_map] = sort(inv_pd,3);
        layout_seg = sort_map(:,:,end);

        ceil_loc = [ceil_y,ceil_x];
        floor_loc = [floor_y,floor_x];
        wall_loc = [wall_y,wall_x];
        ceil = [ceil_loc ceil_param];
        floor  = [floor_loc floor_param];
        wall = [wall_loc wall_param];

        count = 2;
        layout_seg = ones(sz);
        if size(wall,1) >= 1 
            wall_left = (wall(1,3) * X + wall(1,4) * Y + wall(1,5) * Z);
            layout_inv_depth = wall_left;%1/Z 
            for m = 2:size(wall,1) 
                left_centx = round((wall(m-1,2)-1))+1;
                left_centy = round((wall(m-1,1)-1))+1;
                tmp_pid = (wall(m,3) * X + wall(m,4) * Y + wall(m,5) * Z);
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
            end
            for f = 1:size(ceil,1) %ceil  floor         
                tmp_pid_ceil = (ceil(f,3) * X + ceil(f,4) * Y + ceil(f,5) * Z);
                layout_inv_depth = max(layout_inv_depth, tmp_pid_ceil);
                layout_seg(layout_inv_depth == tmp_pid_ceil) = 1;       
            end
            for f = 1:size(floor,1) %ceil  floor         
                tmp_pid_floor = (floor(f,3) * X + floor(f,4) * Y + floor(f,5) * Z);
                layout_inv_depth = max(layout_inv_depth, tmp_pid_floor);
                layout_seg(layout_inv_depth == tmp_pid_floor) = 2;       
            end
        end
        [gx,gy] = gradient(layout_inv_depth);
        grad_abs = abs(gx) + abs(gy);
        score_stitch = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.1 + score_add_stitch - (max(grad_abs(:))>1);
        layout_inv_depth_stitch = layout_inv_depth;
        layout_seg_stitch = layout_seg;
        if score_layer >= score_stitch
            score = score_layer;
            layout_seg = layout_seg_layer;
            layout_inv_depth = layout_inv_depth_layer;
        else
            score = score_stitch;
            layout_seg = layout_seg_layer;
            layout_inv_depth = layout_inv_depth_layer;
        end
    catch
        score = 0;
    end


    try
        depth_map = squeeze(inv_output_depth_flip(1,1,:,:));
        valid = (depth_map > 0.1) & (depth_map < 30);
        layout_depth = 1./depth_map;
        layoutseg = round(squeeze(segmentation_flip(1,:,:,:)));
        layout_map = sum(layoutseg,1);
        layout_map = squeeze(layout_map);
        weight = squeeze(weight_flip(1,1,:,:));
        
        
        loc_mat = (1 ./ ( 1 + exp(-squeeze(center_map_flip))))';
        [plane_cent, loc_coord] = nms_v2(loc_mat, mask_raw,coord_x,coord_y, threshold_conf, threshold_dice);
        [row,col] = find(plane_cent==1);
        wall_center = col;

        wall_center = sortrows(index_inn_corx,2);
        wall_point = zeros(length(wall_center),2);
        for w = 1:length(wall_center)
            wall_point(w,:) = [(wall_center(w)-1)*4,65];
        end
        if wall_point(1,1)==0
            wall_point(1,1)=32;
        end
        ceil_point = [128,13];   
        floor_point = [128,115];
        
        wall_y = wall_point(:,2);
        wall_x = wall_point(:,1);
        ceil_y = ceil_point(:,2);
        ceil_x = ceil_point(:,1);
        floor_y = floor_point(:,2);
        floor_x = floor_point(:,1);
    
        tmp_seg_mask = zeros(sz(1),sz(2),length(wall_center)+2);
        inv_pd = zeros(sz(1),sz(2),length(wall_center)+2);  
        wall_param = zeros(length(wall_center)-1,3);
        for p = 1:length(wall_center)-1
            if p == 1
                first_seg = layoutseg(wall_center(p),:,:); 
                last_seg = layoutseg(wall_center(length(wall_center)),:,:);
                tmp_seg = [first_seg;last_seg];
                tmp_seg = sum(tmp_seg,1);
                tmp_seg = double(squeeze(tmp_seg(1,:,:)));
                tmp_mask = tmp_seg .* weight .* valid;
                tmp_seg_mask(:,:,p) = imresize(tmp_seg,sz);
                vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]);
                sorted = sort(vw,'descend');
                thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
                vw(vw<thre) = 0;
                x = uv1(vw>0,:);
                y = layout_depth(:);
                y = y(vw>0,:);
                w = vw(vw>0);
                w = diag(w);
                wall_param_solo = (x' * w * x) \ x' * w * y;
                wall_param_solo = wall_param_solo';
                wall_param(p,:) = wall_param_solo;% pqr
                inv_pd(:,:,p) = wall_param_solo(1) * X + wall_param_solo(2) * Y + wall_param_solo(3) * Z;      
            else
                tmp_seg = layoutseg(wall_center(p),:,:); 
                tmp_seg = double(squeeze(tmp_seg(1,:,:)));
                tmp_mask = tmp_seg .* weight .* valid;
                tmp_seg_mask(:,:,p) = imresize(tmp_seg,sz);
                vw = reshape(tmp_mask,[out_sz(1)*out_sz(2),1]); 
                sorted = sort(vw,'descend');
                thre = min(threshold_fit, sorted(grid_x * grid_x * thre_sort));
                vw(vw<thre) = 0;
                x = uv1(vw>0,:);
                y = layout_depth(:);
                y = y(vw>0,:);
                w = vw(vw>0);
                w = diag(w);
                wall_param_solo = (x' * w * x) \ x' * w * y;% pqr
                wall_param_solo = wall_param_solo';
                wall_param(p,:) = wall_param_solo;% pqr
                inv_pd(:,:,p) = wall_param_solo(1) * X + wall_param_solo(2) * Y + wall_param_solo(3) * Z;      
            end
        end    
        first_param = wall_param(1,:);
        wall_param = [wall_param;first_param];
    
        ceiltmp_seg = layoutseg(65,:,:);
        ceiltmp_seg = double(squeeze(ceiltmp_seg(1,:,:)));
        ceiltmp_mask = ceiltmp_seg .* weight .* valid;
        tmp_seg_mask(:,:,length(wall_center)+1) = imresize(ceiltmp_seg,sz);
        ceilvw = reshape(ceiltmp_seg,[out_sz(1)*out_sz(2),1]);    
        ceilsorted = sort(ceilvw,'descend');
        thre_ceil = min(threshold_fit, ceilsorted(grid_x * grid_x * thre_sort));
        ceilvw(ceilvw<thre_ceil) = 0;
        ceilx = uv1(ceilvw>0,:);
        ceily = layout_depth(:);
        ceily = ceily(ceilvw>0,:);
        ceilw = weight(ceilvw>0);
        ceilw = diag(ceilw);
        ceil_param = (ceilx' * ceilw * ceilx) \ ceilx' * ceilw * ceily;
        ceil_param = ceil_param';
        inv_pd(:,:,length(wall_center)+1) = ceil_param(1) * X + ceil_param(2) * Y + ceil_param(3) * Z;      
    
        floortmp_seg = layoutseg(66,:,:); 
        floortmp_seg = double(squeeze(floortmp_seg(1,:,:)));
        floortmp_mask = floortmp_seg .* weight .* valid;
        tmp_seg_mask(:,:,length(wall_center)+2) = imresize(floortmp_seg,sz);
        floorvw = reshape(floortmp_seg,[out_sz(1)*out_sz(2),1]);     
        floorsorted = sort(floorvw,'descend');
        thre_floor = min(threshold_fit, floorsorted(grid_x * grid_x * thre_sort));
        floorvw(floorvw<thre_floor) = 0;
        floorx = uv1(floorvw>0,:);
        floory = layout_depth(:);
        floory = floory(floorvw>0,:);
        floorw = weight(floorvw>0);
        floorw = diag(floorw);
        floor_param = (floorx' * floorw * floorx) \ floorx' * floorw * floory;
        floor_param = floor_param';
        inv_pd(:,:,length(wall_center)+2) = floor_param(1) * X + floor_param(2) * Y + floor_param(3) * Z;      
    
        [~, seg_mask] = max(tmp_seg_mask,[],3);   
        [~, sort_map] = sort(inv_pd,3);
        layout_seg = sort_map(:,:,end);
    
        ceil_loc = [ceil_y,ceil_x];
        floor_loc = [floor_y,floor_x];
        wall_loc = [wall_y,wall_x]; 
        ceil = [ceil_loc ceil_param];
        floor  = [floor_loc floor_param];
        wall = [wall_loc wall_param];
        count = 2;
        layout_seg = ones(sz); 
        if size(wall,1) >= 1 
            wall_left = (wall(1,3) * X + wall(1,4) * Y + wall(1,5) * Z);
            layout_inv_depth = wall_left;
            for m = 2:size(wall,1) 
                left_centx = round((wall(m-1,2)-1))+1;
                left_centy = round((wall(m-1,1)-1))+1;
                tmp_pid = (wall(m,3) * X + wall(m,4) * Y + wall(m,5) * Z);
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
            end
            for f = 1:size(ceil,1) %ceil  floor         
                tmp_pid_ceil = (ceil(f,3) * X + ceil(f,4) * Y + ceil(f,5) * Z);
                layout_inv_depth = max(layout_inv_depth, tmp_pid_ceil);
                layout_seg(layout_inv_depth == tmp_pid_ceil) = 1;       
            end
            for f = 1:size(floor,1) %ceil  floor         
                tmp_pid_floor = (floor(f,3) * X + floor(f,4) * Y + floor(f,5) * Z);
                layout_inv_depth = max(layout_inv_depth, tmp_pid_floor);
                layout_seg(layout_inv_depth == tmp_pid_floor) = 2;       
            end
        end
        [gx,gy] = gradient(layout_inv_depth);
        grad_abs = abs(gx) + abs(gy);
        score_stitch = pixelwiseAccuracy(seg_mask, layout_seg, sz) - (length(unique(layout_seg))==1) - (min(layout_inv_depth(:)) < 0) * 0.1 + score_add_stitch - (max(grad_abs(:))>1);
        layout_inv_depth_stitch = layout_inv_depth;
        layout_seg_stitch = layout_seg;
        if score_layer >= score_stitch
            score_flip = score_layer;
            layout_seg = layout_seg_layer;
            layout_inv_depth = layout_inv_depth_layer;
        else
            score_flip = score_stitch;
            layout_seg = layout_seg_layer;
            layout_inv_depth = layout_inv_depth_layer;
        end
    catch
        score_flip = 0;
    end
 
    if score_flip > score
        layout_inv_depth = layout_inv_depth_flip;
        layout_seg = layout_seg_flip;
    end

end



  
