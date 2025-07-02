function iou = calculate_3diou(box1, box2)
    % 计算3D框的体积
    vol_box1 = calculate_volume(box1);
    vol_box2 = calculate_volume(box2);
    
    % 计算3D框的交集体积
    vol_intersection = calculate_intersection_volume(box1, box2);
    
    % 计算3DIoU
    iou = vol_intersection / (vol_box1 + vol_box2 - vol_intersection);
end