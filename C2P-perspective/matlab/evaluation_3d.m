% clear;clc;


load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing.mat
% load result_solo_gt

seg_path = '/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing/layout_seg/';


% result = result(1:200);
% validation = validation(1:384);

h=1024;
w=1280;
[bl_x, bl_y] = meshgrid(1:1:w, h:-1:1);

allPtError = zeros(length(result),1);
allPxError = zeros(length(result),1);
all3dPtError = zeros(length(result),1);

for i = 1:length(result)
%     for i = 1:200
%     
    i
    
    pred_2d = result{i}.point;
    pred_2d = pred_2d(:,1:2);
    gt_2d = data(i).point;
    gt_2d = gt_2d(:,1:2);
    allPtError(i) = cornerError(pred_2d, gt_2d, [1024,1280]);
    
    pred_3d = result{i}.point;
    pred_3d(:,2) = h - pred_3d(:,2);
    gt_3d = data(i).point;
    gt_3d(:,2) = h - gt_3d(:,2);
    
    
    scale(i,1) = mean(pred_3d(:,3));
    scale(i,2) = mean(gt_3d(:,3));
    scale(i,3) = mean(pred_3d(:,3)) / mean(gt_3d(:,3));
    
    %         pred_3d(:,3) = pred_3d(:,3) / scale(i,3);
    
    
    %         mean(scale(:,3))
    %         xx = sortrows(scale,1);
    %         mean(xx(1:980,3))
    %         mean(xx(981:end,3))
    %         xx(981,1)
    %         if scale(i,3)<3.2367
    %             pred_3d(:,3) = pred_3d(:,3) / 1.0149;
    %         else
    %             pred_3d(:,3) = pred_3d(:,3) / 1.0539;
    %         end
    %         pred_3d(:,3) = pred_3d(:,3) / 1.01;
    
    %         x = scale(:,1);
    %         y = scale(:,2);
    %         cftool
    
    %         yy(i) = 0.9252 + 0.168 / scale(i,1);
    %         pred_3d(:,3) = pred_3d(:,3) * yy(i);
    
    intrinsics_matrix = data(i).intrinsics_matrix;
    
    fx = intrinsics_matrix(1,1);
    fy = intrinsics_matrix(2,2);
    cx = intrinsics_matrix(1,3);
    cy = intrinsics_matrix(2,3);
    
    p_xyz = [];
    for pi = 1:size(pred_3d,1)
        u = pred_3d(pi,1);
        v = pred_3d(pi,2);
        z = pred_3d(pi,3);
        
        x = (u-cx)*z/fx;
        y = (v-cy)*z/fy;
        
        p_xyz(pi,:) = [x y z];
    end
    
    g_xyz = [];
    for pi = 1:size(gt_3d,1)
        u = gt_3d(pi,1);
        v = gt_3d(pi,2);
        z = gt_3d(pi,3);
        
        x = (u-cx)*z/fx;
        y = (v-cy)*z/fy;
        
        g_xyz(pi,:) = [x y z];
    end
    
%     md = 0;
%     for j = 1:size(g_xyz,1)-1
%         for k = j+1:size(g_xyz,1)
%             td = sqrt(sum((g_xyz(j,:) - g_xyz(k,:)).^2));
%             md = max(md,td);
%         end
%     end
    
    
    %         all3dPtError(i) = cornerError(p_xyz, g_xyz, md+0.5);
    all3dPtError(i) = cornerError(p_xyz, g_xyz, max(gt_3d(:,3)));
    
    gt_seg = imread([seg_path data(i).layout_seg]);
    allPxError(i) = 1 - pixelwiseAccuracy(result{i}.layout, gt_seg, [1024,1280]);
    
end
meanPtError = mean(allPtError);
meanPxError = mean(allPxError);
mean3derror = mean(all3dPtError);

% meanPtError = sum(allPtError)/i;
% meanPxError = sum(allPxError)/i;
% mean3derror = sum(all3dPtError)/i;





