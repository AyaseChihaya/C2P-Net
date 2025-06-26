function keypoints = gaussian_thresholding_nms(heatmap, threshold, window_size)
    % 阈值处理
    thresholded_heatmap = heatmap .* (heatmap > threshold);

    % 非极大值抑制
    se = strel('square', window_size);
    max_pool = imdilate(thresholded_heatmap, se);

    nms_mask = (thresholded_heatmap == max_pool);

    % 将非极大值区域与原始热力图相乘，得到保留的角点
    keypoints = nms_mask .* thresholded_heatmap;
end

