clear;clc;

% load gt_result

load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing.mat
seg_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/layout_seg/';
layout_depth_path =  '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/layout_depth/';
img_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/image/';

result = [];
sz = [1024,1280];
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
coord_hor = coord_x(1,:);
grid_x = 14;
grid_y = 14;
threshold_ceil = 0.75;
threshold_conf = 0.8;
threshold_cos = 0.3;

cmx = repmat(0:112-1,[112,1])/(112-1);
cmy = repmat(0:112-1,[112,1])'/(112-1);

for i = 1:length(data)
    
    i
    
    %     img = imread([img_path data(i).image]);
    %     figure,imshow(img)
    
    load(['/home/ubuntu/work/regiongrow/predict_param/model_rc_nyu_v0/' num2str(i-1) '.mat']);
    
    
    try
        param_mat = permute(squeeze(output),[2,3,1]);
        %     param_mat_flip = permute(squeeze(output_flip),[2,3,1]);
        %     p_flip = -fliplr(param_mat_flip(:,:,1));
        %     q_flip = fliplr(param_mat_flip(:,:,2));
        %     r_flip = fliplr(param_mat_flip(:,:,1) .* cmx + param_mat_flip(:,:,2) .* cmy + param_mat_flip(:,:,3)) - p_flip .* cmx - q_flip .* cmy;
        %     s_flip = fliplr(param_mat_flip(:,:,4));
        
        loc_mat = 1 ./ ( 1 + exp(-squeeze(output1)));
        loc_mat = nms_v1(loc_mat, param_mat, threshold_conf, threshold_cos);
        
%         loc_mat = squeeze(gt_rc(i,:,:,:));
        %     param_mat = permute(squeeze(gt_param(i,:,:,:)),[2,3,1]);
        %     gen_depth = 1./ ((param_mat(:,:,1) .* cmx + param_mat(:,:,2) .* cmy + param_mat(:,:,3)) .* param_mat(:,:,4));
        %     imshow(gen_depth / 10)
        
        
        [row,col] = find(loc_mat==1);
        
        mask_raw = permute(squeeze(output2),[2,3,1]);
        mask_u = mask_raw(:,:,1:14);
        mask_v = mask_raw(:,:,15:end);
%         mask_raw_flip = permute(squeeze(output2_flip),[2,3,1]);
%         mask_u_flip = mask_raw_flip(:,:,1:14);
%         mask_v_flip = mask_raw_flip(:,:,15:end);
        
        
        model = [];
        ceil_param = [];
        ceil_loc = [];
        wall_param = [];
        wall_loc = [];
        for f = 1:length(row)
            tmp_mask_u = mask_u(:,:,col(f));
            tmp_mask_v = mask_v(:,:,row(f));
            tmp_mask = tmp_mask_u .* tmp_mask_v;
%             tmp_mask_u_flip = mask_u_flip(:,:,grid_x + 1 - col(f));
%             tmp_mask_v_flip = mask_v_flip(:,:,row(f));
%             tmp_mask_flip = fliplr(tmp_mask_u_flip .* tmp_mask_v_flip);
%             tmp_mask = tmp_mask .* tmp_mask_flip;
%             tmp_mask = (tmp_mask>0.5);
            
%             tmp_p = param_mat(:,:,1);
%             tmp_p = tmp_p(tmp_mask);
%             tmp_q = param_mat(:,:,2);
%             tmp_q = tmp_q(tmp_mask);
%             tmp_r = param_mat(:,:,3);
%             tmp_r = tmp_r(tmp_mask);
%             tmp_s = param_mat(:,:,4);
%             tmp_s = tmp_s(tmp_mask);
%             tmp_param(1,1) = trimmean(tmp_p,5);
%             tmp_param(2,1) = trimmean(tmp_q,5);
%             tmp_param(3,1) = trimmean(tmp_r,5);
%             tmp_param(4,1) = trimmean(tmp_s,5);
            
            
            tmp_param = squeeze(sum(param_mat .* repmat(tmp_mask,[1,1,4]),[1,2]) ./ sum(repmat(tmp_mask,[1,1,4]),[1,2]));
            
%             tmp_q = tmp_param(2);
            if abs(tmp_param(2)) > threshold_ceil
                ceil_loc = [ceil_loc;[row(f),col(f)]];
                ceil_param = [ceil_param;tmp_param'];
            else
                wall_loc = [wall_loc;[row(f),col(f)]];
                wall_param = [wall_param;tmp_param'];
            end
        end
        
        ceil = [ceil_loc ceil_param];
        wall = [wall_loc wall_param];
        
        
        if size(wall,1) > 1
            
            wall = sortrows(wall,2);
            wall_left = (wall(1,3) * coord_x + wall(1,4) * coord_y + wall(1,5))*wall(1,6);
            %         left_centx = round((wall(1,2)-1) / (grid_x-1) * (sz(2)-1))+1;
            %         left_centy = round((wall(1,1)-1) / (grid_y-1) * (sz(1)-1))+1;
            layout_inv_depth = wall_left;
            layout_seg = ones(sz);
            count = 1;
            model(count).face = 1;
            model(count).params = [wall(1,3), wall(1,4), wall(1,5)]*wall(1,6);
            for f = 2:size(wall,1)
                
                tmp_pid = (wall(f,3) * coord_x + wall(f,4) * coord_y + wall(f,5))*wall(f,6);
                seg1 = tmp_pid >= wall_left;
                seg2 = tmp_pid < wall_left;
                seg1_x = mean(coord_hor(seg1(sz(1)/2,:)));
                seg2_x = mean(coord_hor(seg2(sz(1)/2,:)));
                count = count + 1;
                if seg1_x < seg2_x
                    layout_inv_depth(seg2) = tmp_pid(seg2);
                    layout_seg(seg2) = count;
                else
                    layout_inv_depth(seg1) = tmp_pid(seg1);
                    layout_seg(seg1) = count;
                end
                
                wall_left = tmp_pid;
                %             left_centx = round((wall(f,2)-1) / (grid_x-1) * (sz(2)-1))+1;
                %             left_centy = round((wall(f,1)-1) / (grid_y-1) * (sz(1)-1))+1;
                
                model(count).face = count;
                model(count).params = [wall(f,3), wall(f,4), wall(f,5)]*wall(f,6);
                
            end
            
            for f = 1:size(ceil,1)
                
                tmp_pid = (ceil(f,3) * coord_x + ceil(f,4) * coord_y + ceil(f,5))*ceil(f,6);
                layout_inv_depth = max(layout_inv_depth, tmp_pid);
                count = count + 1;
                layout_seg(layout_inv_depth == tmp_pid) = count;
                
                model(count).face = count;
                model(count).params = [ceil(f,3), ceil(f,4), ceil(f,5)]*ceil(f,6);
                
            end
            
        else
            
            faces = [ceil_param; wall_param];
            all_pid = zeros(sz(1),sz(2),size(faces,1));
            for f = 1:size(faces,1)
                all_pid(:,:,f) = (faces(f,1) * coord_x + faces(f,2) * coord_y + faces(f,3))*faces(f,4);
            end
            
            [pid_value, pid_label] = sort(all_pid,3);
            layout_inv_depth = pid_value(:,:,end);
            layout_seg = pid_label(:,:,end);
            vis_face = unique(layout_seg);
            count1 = 0;
            for f = 1:size(faces,1)
                if ismember(f, vis_face)
                    count1 = count1 + 1;
                    model(count1).face = f;
                    model(count1).params = [faces(f,1), faces(f,2), faces(f,3)]*faces(f,4);
                end
            end
            
        end
        
        
        %         layout_depth = 1 ./ layout_inv_depth;
        %         imshow(1 ./ layout_inv_depth, [])
        point = gen_param_point2(layout_inv_depth, layout_seg, model);
        result{i}.point = point;
        result{i}.layout = layout_seg;
        
    catch
        
        try
            
            param_mat_flip = permute(squeeze(output_flip),[2,3,1]);
            p_flip = -fliplr(param_mat_flip(:,:,1));
            q_flip = fliplr(param_mat_flip(:,:,2));
            r_flip = fliplr(param_mat_flip(:,:,1) .* cmx + param_mat_flip(:,:,2) .* cmy + param_mat_flip(:,:,3)) - p_flip .* cmx - q_flip .* cmy;
            s_flip = fliplr(param_mat_flip(:,:,4));
            param_mat(:,:,1) = p_flip;
            param_mat(:,:,2) = q_flip;
            param_mat(:,:,3) = r_flip;
            param_mat(:,:,4) = s_flip;
            
            
            loc_mat = fliplr(1 ./ ( 1 + exp(-squeeze(output1_flip))));
            loc_mat = nms_v1(loc_mat, param_mat, threshold_conf, threshold_cos);
            
            
            [row,col] = find(loc_mat==1);
            
            %             mask_raw = permute(squeeze(output2),[2,3,1]);
            %             mask_u = mask_raw(:,:,1:14);
            %             mask_v = mask_raw(:,:,15:end);
            mask_raw_flip = permute(squeeze(output2_flip),[2,3,1]);
            mask_u_flip = mask_raw_flip(:,:,1:14);
            mask_v_flip = mask_raw_flip(:,:,15:end);
            
            
            model = [];
            ceil_param = [];
            ceil_loc = [];
            wall_param = [];
            wall_loc = [];
            for f = 1:length(row)
                %                 tmp_mask_u = mask_u(:,:,col(f));
                %                 tmp_mask_v = mask_v(:,:,row(f));
                %                 tmp_mask = tmp_mask_u .* tmp_mask_v;
                tmp_mask_u_flip = mask_u_flip(:,:,grid_x + 1 - col(f));
                tmp_mask_v_flip = mask_v_flip(:,:,row(f));
                tmp_mask = fliplr(tmp_mask_u_flip .* tmp_mask_v_flip);
                %                 tmp_mask = tmp_mask .* tmp_mask_flip;
                
%                 tmp_mask = (tmp_mask>0.5);
% 
%                 tmp_p = param_mat(:,:,1);
%                 tmp_p = tmp_p(tmp_mask);
%                 tmp_q = param_mat(:,:,2);
%                 tmp_q = tmp_q(tmp_mask);
%                 tmp_r = param_mat(:,:,3);
%                 tmp_r = tmp_r(tmp_mask);
%                 tmp_s = param_mat(:,:,4);
%                 tmp_s = tmp_s(tmp_mask);
%                 tmp_param(1,1) = trimmean(tmp_p,5);
%                 tmp_param(2,1) = trimmean(tmp_q,5);
%                 tmp_param(3,1) = trimmean(tmp_r,5);
%                 tmp_param(4,1) = trimmean(tmp_s,5);

                
                tmp_param = squeeze(sum(param_mat .* repmat(tmp_mask,[1,1,4]),[1,2]) ./ sum(repmat(tmp_mask,[1,1,4]),[1,2]));
                
%                 tmp_q = tmp_param(2);
                if abs(tmp_param(2)) > threshold_ceil
                    ceil_loc = [ceil_loc;[row(f),col(f)]];
                    ceil_param = [ceil_param;tmp_param'];
                else
                    wall_loc = [wall_loc;[row(f),col(f)]];
                    wall_param = [wall_param;tmp_param'];
                end
            end
            
            ceil = [ceil_loc ceil_param];
            wall = [wall_loc wall_param];
            
            
            if size(wall,1) > 1
                
                wall = sortrows(wall,2);
                wall_left = (wall(1,3) * coord_x + wall(1,4) * coord_y + wall(1,5))*wall(1,6);
                %         left_centx = round((wall(1,2)-1) / (grid_x-1) * (sz(2)-1))+1;
                %         left_centy = round((wall(1,1)-1) / (grid_y-1) * (sz(1)-1))+1;
                layout_inv_depth = wall_left;
                layout_seg = ones(sz);
                count = 1;
                model(count).face = 1;
                model(count).params = [wall(1,3), wall(1,4), wall(1,5)]*wall(1,6);
                for f = 2:size(wall,1)
                    
                    tmp_pid = (wall(f,3) * coord_x + wall(f,4) * coord_y + wall(f,5))*wall(f,6);
                    seg1 = tmp_pid >= wall_left;
                    seg2 = tmp_pid < wall_left;
                    seg1_x = mean(coord_hor(seg1(sz(1)/2,:)));
                    seg2_x = mean(coord_hor(seg2(sz(1)/2,:)));
                    count = count + 1;
                    if seg1_x < seg2_x
                        layout_inv_depth(seg2) = tmp_pid(seg2);
                        layout_seg(seg2) = count;
                    else
                        layout_inv_depth(seg1) = tmp_pid(seg1);
                        layout_seg(seg1) = count;
                    end
                    
                    wall_left = tmp_pid;
                    %             left_centx = round((wall(f,2)-1) / (grid_x-1) * (sz(2)-1))+1;
                    %             left_centy = round((wall(f,1)-1) / (grid_y-1) * (sz(1)-1))+1;
                    
                    model(count).face = count;
                    model(count).params = [wall(f,3), wall(f,4), wall(f,5)]*wall(f,6);
                    
                end
                
                for f = 1:size(ceil,1)
                    
                    tmp_pid = (ceil(f,3) * coord_x + ceil(f,4) * coord_y + ceil(f,5))*ceil(f,6);
                    layout_inv_depth = max(layout_inv_depth, tmp_pid);
                    count = count + 1;
                    layout_seg(layout_inv_depth == tmp_pid) = count;
                    
                    model(count).face = count;
                    model(count).params = [ceil(f,3), ceil(f,4), ceil(f,5)]*ceil(f,6);
                    
                end
                
            else
                
                faces = [ceil_param; wall_param];
                all_pid = zeros(sz(1),sz(2),size(faces,1));
                for f = 1:size(faces,1)
                    all_pid(:,:,f) = (faces(f,1) * coord_x + faces(f,2) * coord_y + faces(f,3))*faces(f,4);
                end
                
                [pid_value, pid_label] = sort(all_pid,3);
                layout_inv_depth = pid_value(:,:,end);
                layout_seg = pid_label(:,:,end);
                vis_face = unique(layout_seg);
                count1 = 0;
                for f = 1:size(faces,1)
                    if ismember(f, vis_face)
                        count1 = count1 + 1;
                        model(count1).face = f;
                        model(count1).params = [faces(f,1), faces(f,2), faces(f,3)]*faces(f,4);
                    end
                end
                
            end
            
            
            %         layout_depth = 1 ./ layout_inv_depth;
            %         imshow(1 ./ layout_inv_depth, [])
            point = gen_param_point2(layout_inv_depth, layout_seg, model);
            result{i}.point = point;
            result{i}.layout = layout_seg;
            
        catch
            result{i}.point = result{i-1}.point;
            result{i}.layout = result{i-1}.layout;
        end
    end
end


save result_solo_gt result
