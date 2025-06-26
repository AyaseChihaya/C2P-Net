function lonlat = uv2lonlat(uv, axis)
    if nargin < 2
        lon = (uv(:, 1) - 0.5) * 2 * pi;
        lat = (uv(:, 2) - 0.5) * pi;
    elseif axis == 0
        lon = (uv - 0.5) * 2 * pi;
        lonlat = lon;
        return;
    elseif axis == 1
        lat = (uv - 0.5) * pi;
        lonlat = lat;
        return;
    else
        error('axis error');
    end
    
    lonlat = cat(3, lon, lat);
end