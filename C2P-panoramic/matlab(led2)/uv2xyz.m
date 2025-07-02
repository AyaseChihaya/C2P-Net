function xyz = uv2xyz(uv, plan_y, spherical)
    lonlat = uv2lonlat(uv);
    xyz = lonlat2xyz(lonlat);
    
    if nargin > 2 && spherical
        % Projection onto the sphere
        return;
    end
    
    if nargin < 2 || isempty(plan_y)
        plan_y = 1;
    end
    
    % Projection onto the specified plane
    xyz = xyz .* (plan_y ./ xyz(:, 2));
end