function out = nms_cent(loc, cent_x, cent_y, grid_cent_x, grid_cent_y, weight, threshold_conf, threshold_dist)

loc(loc<threshold_conf) = 0;
[h,w] = size(loc);

for i = 1:h
    for j = 1:w
        cent_conf = loc(i,j);
        if cent_conf == 0
            continue
        end
        local_row = repmat((i-1:i+1),[3,1]);
        local_row = local_row(:);
        local_col = repmat((j-1:j+1),[1,3])';
        local_coord = [local_row local_col];
        local_coord(5,:) = [];
        local_coord(local_coord(:,1)<1,:) = [];
        local_coord(local_coord(:,1)>h,:) = [];
        local_coord(local_coord(:,2)<1,:) = [];
        local_coord(local_coord(:,2)>h,:) = [];
        for k = 1:size(local_coord,1)
            tmp_conf = loc(local_coord(k,1),local_coord(k,2));
            if tmp_conf < cent_conf
                continue
            end
            cent_index = grid_cent_x==j & grid_cent_y==i;
            cent_mean_x = sum(cent_x(cent_index) .* weight(cent_index)) / sum(weight(cent_index)) * (w-1) + 1;
            cent_mean_y = sum(cent_y(cent_index) .* weight(cent_index)) / sum(weight(cent_index)) * (h-1) + 1;
            tmp_index = grid_cent_x==local_coord(k,2) & grid_cent_y==local_coord(k,1);
            tmp_mean_x = sum(cent_x(tmp_index) .* weight(tmp_index)) / sum(weight(tmp_index)) * (w-1) + 1;
            tmp_mean_y = sum(cent_y(tmp_index) .* weight(tmp_index)) / sum(weight(tmp_index)) * (h-1) + 1;
            dist = sqrt((cent_mean_x - tmp_mean_x) ^ 2 + (cent_mean_y - tmp_mean_y) ^ 2);
            if dist < threshold_dist
                loc(i,j) = 0;
                break
            end
        end
    end
end

out = loc > 0;

end



