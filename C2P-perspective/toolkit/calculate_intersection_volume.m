function intersection_volume = calculate_intersection_volume(box1, box2)
    % 计算3D框的交集体积
    min_x = max(box1(1), box2(1));
    min_y = max(box1(2), box2(2));
    min_z = max(box1(3), box2(3));
    max_x = min(box1(4), box2(4));
    max_y = min(box1(5), box2(5));
    max_z = min(box1(6), box2(6));
    
    dx = max(0, max_x - min_x);
    dy = max(0, max_y - min_y);
    dz = max(0, max_z - min_z);
    
    intersection_volume = dx * dy * dz;
end
