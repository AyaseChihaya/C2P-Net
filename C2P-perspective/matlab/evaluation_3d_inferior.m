% clear;clc;

load /home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/test.mat
seg_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/layout_seg/';
point_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/layout_keypoint_mat/';
param_path = '/home/ubuntu/work/geolayout/dataset/InteriorNet-Layout/test/intrinsics_matrix_mat/';

data = strtrim(string(data));

h = 480;
w = 640;
h1 = 192;
w1 = 256;

[bl_x, bl_y] = meshgrid(1:1:w, h:-1:1);

allPtError = zeros(length(result),1);
allPxError = zeros(length(result),1);
all3dPtError = zeros(length(result),1);

for i = 1:length(result)
    
    i
    
    tmp_name = char(data(i));
    gt_point = load([point_path tmp_name '.mat']).point;
    gt_point = gt_point(any(gt_point,2),:);

    pred_point = single(result{i}.point);
    pred_point(:,1) = pred_point(:,1) * 639 / 640;
    pred_point(:,2) = pred_point(:,2) * 479 / 480;
    
    pred_2d = pred_point(:,1:2);
    gt_2d = gt_point(:,1:2);
    allPtError(i) = cornerError(pred_2d, gt_2d, [480,640]);
    
    pred_3d = pred_point;
    pred_3d(:,2) = h - pred_3d(:,2);
    gt_3d = gt_point;
    gt_3d(:,2) = h - gt_3d(:,2);
    
    
    scale(i,1) = mean(pred_3d(:,3));
    scale(i,2) = mean(gt_3d(:,3));
    scale(i,3) = mean(pred_3d(:,3)) / mean(gt_3d(:,3));
    
    %         pred_3d(:,3) = pred_3d(:,3) / scale(i,3);
    
    
    intrinsics_matrix = double(load([param_path tmp_name '.mat']).point);
    
    
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
    

%     m = size(g_xyz, 1);
%     n = size(p_xyz, 1);
%     
%     gt_repeat = repmat(g_xyz, 1, n);
%     gt_repeat = reshape(gt_repeat, n * m, 3);
%     pred_repeat = repmat(p_xyz, m, 1);
%     
%     distance = vecnorm(gt_repeat - pred_repeat, 2, 2);
%     distance = reshape(distance, m, n);
%     
%     [error, ~] = min(distance, [], 2);
%     error_mean = mean(error);
%     
%     
%     all3dPtError(i) = error_mean;
%     
%     gt_seg = imread([seg_path tmp_name '.png']);
%     
%     allPxError(i) = 1 - pixelwiseAccuracy(result{i}.layout, gt_seg, [480,640]);
    
    all3dPtError(i) = cornerError(p_xyz, g_xyz, max(gt_3d(:,3)));
    
    gt_seg = imread([seg_path tmp_name '.png']);
    allPxError(i) = 1 - pixelwiseAccuracy(result{i}.layout, gt_seg, [480,640]);
    
    
end
meanPtError = mean(allPtError);
meanPxError = mean(allPxError);
mean3derror = mean(all3dPtError);


mse = zeros(1,length(data));
rmse = zeros(1,length(data));
for i = 1:length(data)
    
    i
    pred = imread(['/home/ubuntu/work/regiongrow/result_inferior/' num2str(i,'%04d') '_layout.png']);
    gt = imread(['/home/ubuntu/work/geolayout/dataset/test_depth_gt_interior/' num2str(i,'%04d') '.png']);
    
    pred = double(pred) / 4000;
    gt = double(gt) / 4000;
    
    pred = imresize(pred,0.5);
    gt = imresize(gt,0.5);
    
%     rmse(i) = sqrt(mean((pred(:) - gt(:)).^2)); 
    mse(i) = mean((pred(:) - gt(:)).^2); 
    rmse(i) = sqrt(mse(i));
    
end

meanrmse = mean(rmse);

% mean(mse)
% mean(rmse)
    


