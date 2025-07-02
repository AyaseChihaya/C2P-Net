function [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf, threshold_dice)

loc_mat(loc_mat<threshold_conf) = 0;
[h,w] = size(loc_mat);
loc_coord = zeros(h,w,2);

for i = 1:h
    for j = 1:w
        if loc_mat(i,j) == 0
            continue
        end
        cent_conf = loc_mat(i,j);
        cent_mask = mask_u(:,:,j) .* mask_v(:,:,i);
        cent_mask_bi = cent_mask > 0.5;
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
            tmp_conf = loc_mat(local_coord(k,1),local_coord(k,2));
            if tmp_conf < cent_conf
                continue
            end
            tmp_mask = mask_u(:,:,local_coord(k,2)) .* mask_v(:,:,local_coord(k,1));
            tmp_mask_bi = tmp_mask > 0.5;
            dice_similarity = dice(cent_mask_bi, tmp_mask_bi);
            if dice_similarity > threshold_dice
                loc_mat(i,j) = 0;
                break
            end
        end
        if loc_mat(i,j)
            loc_coord(i,j,1) = sum(cmx .* cent_mask, 'all') / sum(cent_mask, 'all');
            loc_coord(i,j,2) = sum(cmy .* cent_mask, 'all') / sum(cent_mask, 'all');
        end
    end
end

plane_cent = loc_mat > 0;

end



