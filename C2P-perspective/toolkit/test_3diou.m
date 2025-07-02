function iou = test_3diou(gt_points, dt_points)
    % 计算真实值和测试的xyz点对应的3D框
    gt_box = calculate_bounding_box(gt_points);
    dt_box = calculate_bounding_box(dt_points);
    
    % 计算3DIoU
    iou = calculate_3diou(gt_box, dt_box);
end
