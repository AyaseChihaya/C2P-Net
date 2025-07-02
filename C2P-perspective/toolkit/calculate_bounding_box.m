function box = calculate_bounding_box(points)
    % 计算点集的包围盒
    min_x = min(points(:, 1));
    min_y = min(points(:, 2));
    min_z = min(points(:, 3));
    max_x = max(points(:, 1));
    max_y = max(points(:, 2));
    max_z = max(points(:, 3));
    
    box = [min_x, min_y, min_z, max_x, max_y, max_z];
end