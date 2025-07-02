function xyz = lonlat2xyz(lonlat, plan_y)
    lon = lonlat(:, 1);
    lat = lonlat(:, 2);
    x = cos(lat) .* sin(lon);
    y = sin(lat);
    z = cos(lat) .* cos(lon);
    xyz = [x, y, z];

    if nargin > 1 && ~isempty(plan_y)
        xyz = xyz .* (plan_y ./ xyz(:, 2));
    end
end