function volume = calculate_volume(box)
    % 计算3D框的体积
    dx = box(4) - box(1);
    dy = box(5) - box(2);
    dz = box(6) - box(3);
    volume = dx * dy * dz;
end