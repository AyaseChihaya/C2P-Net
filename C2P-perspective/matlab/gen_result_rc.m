function [seg_mask, layout_inv_depth, layout_seg, model] = gen_result_rc(param_mat, plane, mask_u, mask_v, loc_mat)

sz = [1024,1280];
out_sz = [112, 112];
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);
grid_x = 14;
grid_y = 14;
threshold_conf = 0.7;
threshold_conf_mod = 0.3;
threshold_dice = 0.5;

cmx = repmat(0:out_sz(2)-1,[out_sz(1),1])/(out_sz(2)-1);
cmy = repmat(0:out_sz(1)-1,[out_sz(2),1])'/(out_sz(1)-1);

[plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf, threshold_dice);
[row,col] = find(plane_cent==1);

if length(row) < 2
    [plane_cent, loc_coord] = nms_v2(loc_mat, mask_u, mask_v, cmx, cmy, threshold_conf_mod, threshold_dice);
    [row,col] = find(plane_cent==1);
end

model = [];
tmp_seg_mask = zeros(1024,1280,length(row));
inv_pd = zeros(1024,1280,length(row));
mask2cent = zeros(length(row),2);

for f = 1:length(row)
    tmp_seg = mask_u(:,:,col(f)) .* mask_v(:,:,row(f));
    tmp_mask = tmp_seg .* plane;
    tmp_seg_mask(:,:,f) = imresize(tmp_seg,sz);
    cent_x = loc_coord(row(f), col(f), 1);
    ub_x = (col(f) + 0.5 - 1) / (grid_x - 1);
    lb_x = (col(f) - 0.5 - 1) / (grid_x - 1);
    mask2cent(f,1) = min(max(lb_x, cent_x), ub_x);
    cent_y = loc_coord(row(f), col(f), 2);
    ub_y = (row(f) + 0.5 - 1) / (grid_y - 1);
    lb_y = (row(f) - 0.5 - 1) / (grid_y - 1);
    mask2cent(f,2) = min(max(lb_y, cent_y), ub_y);
    tmp_param = squeeze(sum(param_mat .* repmat(tmp_mask,[1,1,4]),[1,2]) ./ sum(repmat(tmp_mask,[1,1,4]),[1,2]));
    inv_pd(:,:,f) = (tmp_param(1) * coord_x + tmp_param(2) * coord_y + tmp_param(3)) * tmp_param(4);
    model(f).face = f;
    model(f).params = [tmp_param(1),tmp_param(2),tmp_param(3)] * tmp_param(4);
end

[~, seg_mask] = max(tmp_seg_mask,[],3);

[~, sort_map] = sort(inv_pd,3);
layout_seg = sort_map(:,:,end);

invd_mat = zeros(length(row),length(row));
for p = 1:length(row)
    for q = 1:length(row)
        invd_mat(p,q) = model(q).params(1) * mask2cent(p,1) + model(q).params(2) * mask2cent(p,2) + model(q).params(3);
    end
end

[~, md] = max(invd_mat);
if isequal(md, 1:length(row))
    layout_inv_depth = zeros(size(layout_seg));
    for n = 1:length(row)
        tmp_depth_layer = inv_pd(:,:,n);
        layout_inv_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
    end
else
    m = ones(1,length(row))*length(row);
    for n = 1:length(row)
        tmp_label = layout_seg(round(mask2cent(n,2)*(sz(1)-1))+1, round(mask2cent(n,1)*(sz(2)-1))+1);
        if tmp_label ~= n && mode(seg_mask(layout_seg==n)) ~= n
            m(n) = m(n) - 1;
            tmp_layer = sort_map(:,:,m(n));
            layout_seg(layout_seg==tmp_label) = tmp_layer(layout_seg==tmp_label);
        end
    end
    layout_inv_depth = zeros(size(layout_seg));
    for n = 1:length(row)
        tmp_depth_layer = inv_pd(:,:,n);
        layout_inv_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
    end
end

