function [xyz] = uvtoxyz(uv)

    u = uv(:,1)/1024;
    v = uv(:,2)/512;
    % 定义UV坐标原点（0, 0）对应的经度和纬度
    origin_lat = 0;  % 原点纬度
    origin_lon = 0;  % 原点经度
        
    % 定义经纬度范围
    lat_range = [-90, 90];  % 纬度范围
    lon_range = [-180, 180];  % 经度范围
        
    % 将UV坐标映射到经纬度范围内
    lat = (v - 0.5) * (lat_range(2) - lat_range(1)) + origin_lat;
    lon = (u - 0.5) * (lon_range(2) - lon_range(1)) + origin_lon;
        
    % 将经纬度转换为XYZ坐标
    radius = 1;  % 球体半径
    x = radius * cos(lat) * cos(lon);
    y = radius * cos(lat) * sin(lon);
    z = radius * sin(lat);
    xyz = [x y z];

end
