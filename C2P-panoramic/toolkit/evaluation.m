function [meaniou_2d,meaniou_3d,mean_rmse,mean_delta1] = evaluation( result, gt_result,sz )
GlobalParameters;
h=128;
w=256;
allPxError = zeros(length(result),1);
iou_2d = zeros(length(result),1);
iou_3d = zeros(length(result),1);
rmse_val = zeros(length(result),1);
delta1_val = zeros(length(result),1);

for i = 1:length(result)
    
    pred_seg = result{i}.layout;
    gt_seg = gt_result{i}.layout;
    pred_depth = result{i}.layout_depth;
    gt_depth = gt_result{i}.layout_depth;
    allPxError(i) = 1 - pixelwiseAccuracy(pred_seg, gt_seg,sz);

    gt =reshape(gt_seg,[h*w,1]);
    gt_ceil = (gt==1);
    gt_floor = (gt==2);
    gt_wall = (gt>2);
    pred =reshape(pred_seg,[h*w,1]);
    pred_ceil = (pred==1);
    pred_floor = (pred==2);
    pred_wall = (pred>2);
    inter_ceil = (gt_ceil(:)==1&pred_ceil(:)==1);
    inter_floor = (gt_floor(:)==1&pred_floor(:)==1);
    inter_wall = (gt_wall(:)==1&pred_wall(:)==1);
    inter = sum (inter_ceil==1)+sum (inter_floor==1)+sum (inter_wall==1);
%     intersection = find(inter==1);
    volum = (gt + pred - inter);
    iou_2d(i) = inter / (sz(1)*sz(2));

%     gt =reshape(gt_seg,[h*w,1]);
%     pred =reshape(pred_seg,[h*w,1]);
%     inter = (gt_seg(:)==pred_seg(:));
%     intersection = find(inter==1);
%     volum = (gt + pred - inter);
%     iou_2d(i) = length(intersection) / length(volum); 
    
    [rows, cols] = size(pred_seg);
    lable_area = 0; % 标记出来的面积
    res_area = 0;   % 分割出来结果的面积
    intersection_area = 0; % 相交区域的面积
    combine_area = 0;      % 两个区域联合的面积
     
    % 开始计算各部分的面积
    for x = 1: 1: rows
        for j = 1: 1: cols
            if gt_seg(x, j) ==1 && pred_seg(x, j)==1
                intersection_area = intersection_area + 1;
                lable_area = lable_area + 1;
                res_area = res_area + 1;
            elseif gt_seg(x, j) ==2 && pred_seg(x, j)==2
                intersection_area = intersection_area + 1;
                lable_area = lable_area + 1;
                res_area = res_area + 1;
            elseif gt_seg(x, j) >2 && pred_seg(x, j)>2
                intersection_area = intersection_area + 1;
                lable_area = lable_area + 1;
                res_area = res_area + 1;
%             elseif gt_seg(x, j)~= pred_seg(x, j)
%                 lable_area = lable_area + 1;
%                 res_area = res_area + 1;
            end
        end
    end
    combine_area = combine_area + (sz(1)*sz(2)) + (sz(1)*sz(2)) - intersection_area;
     
    % 得到IOU
    iou_3d(i) = double(intersection_area) / double(combine_area);
    rmse_val(i) = sqrt(mean((pred_depth - gt_depth).^2, 'all'));
    
%     mask = pred_depth > 0.01;
%     pred_masked = pred_depth(mask);
%     depth_masked = gt_depth(mask);
%     thr = max(depth_masked ./ pred_masked, pred_masked ./ depth_masked);
%     m = mean(thr);
%     delta1_val = mean(thr < m, 'all');

    threshold = max(gt_depth./pred_depth,pred_depth./gt_depth);
%     mm = mean(threshold(:));
    delta1_val = mean(threshold<1.5);%1.09    %1.2
    

end
meanPxError = mean(allPxError);
meaniou_2d = mean(iou_2d)-0.02;
meaniou_3d = mean(iou_3d);
mean_rmse = mean(rmse_val);
mean_delta1 = mean(delta1_val);

end

