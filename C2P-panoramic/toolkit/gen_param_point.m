function [model,point] = gen_param_point(layout_depth, seg)

inv_depth = 1 ./ layout_depth;

[h,w] = size(layout_depth);
faces = unique(seg);

coord_x = repmat(0:w-1,[h,1])/(w-1);
coord_y = repmat(0:h-1,[w,1])'/(h-1);

model = [];
for f = 1:length(faces)
    
    region = seg==faces(f);
    vis_idx = find(region);
    
    [V, U] = ind2sub([h,w],vis_idx);
    U = (U-1) / (w-1);
    V = (V-1) / (h-1);
    D = 1./ layout_depth(vis_idx);
    fit_points = [U V D];
    
    myfun = @(x, UV) (x(1)*UV(:,1)+x(2)*UV(:,2)+x(3));
    
    
    tmp_model = nlinfit(fit_points(:,1:2),fit_points(:,3),myfun,randn(1,3)*1e-8);
    
    dist = abs((tmp_model(1)*coord_x+tmp_model(2)*coord_y+tmp_model(3)) ./ inv_depth - 1) .* region;
    dist = sum(dist(:)) / sum(region(:));
    assert(dist<0.03)
    
    model(f).face = faces(f);
    model(f).params = tmp_model;
    
    
end


border_point = [];
count_bp = 0;
combs = nchoosek(1:length(faces),2);
for j = 1:size(combs,1)
    input_planes = [model(combs(j,1)).params; model(combs(j,2)).params];
    [la,lb,lc] = get_intersaction_line(input_planes);
    %     ips = lineToBorderPoints([la,lb,lc],[1,1]);
    %     if ips(1)==-1
    %         continue
    %     end
    ips = [];
    p1 = [-lc/la, 0];
    if p1(1)>=0 && p1(1)<=1
        ips = [ips;p1];
    end
    p2 = [-(lb+lc)/la, 1];
    if p2(1)>=0 && p2(1)<=1
        ips = [ips;p2];
    end
    p3 = [0,-lc/lb];
    if p3(2)>=0 && p3(2)<=1
        ips = [ips;p3];
    end
    p4 = [1,-(la+lc)/lb];
    if p4(2)>=0 && p4(2)<=1
        ips = [ips;p4];
    end
    
    ips = unique(ips,'rows');
    
    if ~isempty(ips)
        for jj = 1:size(ips,1)
            d = input_planes(1,1)*ips(jj,1) + input_planes(1,2)*ips(jj,2) + input_planes(1,3);
            %         d2 = input_planes(2,1)*ips(jj,1) + input_planes(2,2)*ips(jj,2) + input_planes(2,3);
            [row,col] = uv2rc(ips(jj,:).*[w,h]);
            [np1,np2] = neighbor_pixels(row,col,h,w,4);
            if abs(d/inv_depth(row,col) - 1) < 0.03 && prod(ismember([model(combs(j,1)).face, model(combs(j,2)).face],unique(seg(np1,np2))))
                count_bp = count_bp+1;
                border_point(count_bp,:) = [ips(jj,:).*[w,h],d];
            end
        end
    end
end


inner_point = [];
count_ip = 0;
if length(faces)>=3
    combs = nchoosek(1:length(faces),3);
    for j = 1:size(combs,1)
        input_planes = [model(combs(j,1)).params; model(combs(j,2)).params; model(combs(j,3)).params];
        [u,v,d] = get_intersaction_point(input_planes);
        if u<0 || u>1 || v<0 || v>1
            continue
        end
%         u = u * w;
%         v = v * h;
        [row,col] = uv2rc([u,v].*[w,h]);
        [np1,np2] = neighbor_pixels(row,col,h,w,4);
        if abs(d/inv_depth(row,col) - 1) < 0.03 && prod(ismember([model(combs(j,1)).face, model(combs(j,2)).face, model(combs(j,3)).face],unique(seg(np1,np2))))
            count_ip = count_ip+1;
            inner_point(count_ip,:) = [[u,v].*[w,h],d];
        end
    end
end

point = [border_point; inner_point];
point(:,3) = 1 ./ point(:,3);









