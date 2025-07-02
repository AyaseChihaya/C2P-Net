clear;clc;

load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing.mat


dataset = data;


result = [];
sz = [1024,1280];
coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);

h=112;
w=140;
cmx = repmat(0:w-1,[h,1])/(w-1);
cmy = repmat(0:h-1,[w,1])'/(h-1);

for i = 1:100
% for i = 1:length(dataset)


    i


    load(['/home/ubuntu/work/regiongrow/predict_param/model_regioncent_v0/' num2str(i-1) '.mat']);

    p = squeeze(param(1,1,:,:,1));
    q = squeeze(param(1,2,:,:,1));
    r = squeeze(param(1,3,:,:,1));
    s = squeeze(param(1,4,:,:,1));
    
    mask = squeeze(param(1,5,:,:,1));
    mask(mask>1) = 1;
       
    centx = squeeze(param(1,6,:,:,1));
    centy = squeeze(param(1,7,:,:,1));
    
    gen_depth = 1./ ((p.*cmx+q.*cmy+r) .* s + 1e-10);
    
    emb = [];
    emb(1,:,:) = centx;
    emb(2,:,:) = centy;
    emb = reshape(emb,2,[]);

    score = [];
    tmp_map = [];
    tmp_seg = [];
    tmp_rc = [];

%     msd = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6];
    msd = [0.15, 0.2, 0.25, 0.4];
    try
    for pm = 1:length(msd)

        [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(emb,msd(pm));
%         [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(emb,0.15);
        gen_seg = reshape(data2cluster,[h,w]);
        if numel(unique(gen_seg)) > 100
            continue
        end
%         imshow(gen_seg,[])

        faces = unique(gen_seg);

        select_region = [];
        filled_map = [];
        visiable_area = zeros(sz);
        rc = 0;
%         mod_param = [];
        model = [];
        for f = 1:length(faces)

            region = gen_seg==f;
            pct_pix = sum(region(:))/numel(gen_seg);


            if pct_pix < 0.01
                continue
            end

            [v, u] = ind2sub([h,w],find(region));
            border_pix = sum(u==1|u==w|v==1|v==h) / sum(region(:));

            l = bwlabel(region);
            ni = [];
            for li = 1:max(l(:))
                ni(li) = sum(l(:)==li);
            end
            new_l = l==find(ni==max(ni),1);
            perim = regionprops(new_l,'Perimeter');
            ppp = perim.Perimeter / sqrt(sum(region(:)));

            if ppp>6.5 && border_pix < 0.05
                continue
            end


            ma = trimmean(p(new_l),30);
            mb = trimmean(q(new_l),30);
            mc = trimmean(r(new_l),30);
            ms = trimmean(s(new_l),30);
            
            weight_mat = new_l .* mask;
            ma = sum(p(:) .* weight_mat(:)) / sum(weight_mat(:));
            mb = sum(q(:) .* weight_mat(:)) / sum(weight_mat(:));
            mc = sum(r(:) .* weight_mat(:)) / sum(weight_mat(:));
            ms = sum(s(:) .* weight_mat(:)) / sum(weight_mat(:));
            
            rc = rc+1;

            visiable_area = visiable_area + imresize(new_l,sz,'nearest') * rc;
            filled_map(:,:,rc) = 1./ ((ma*coord_x+mb*coord_y+mc) * ms + 1e-10);
           
            model(rc).face = rc;
            model(rc).params = [ma* ms,mb* ms,mc* ms];
%             model(rc).params = [ma,mb,mc];

        end


        filled_map(filled_map<0) = inf;

        [~, sort_map] = sort(filled_map,3);
        m = ones(1,rc);
        layout_seg = sort_map(:,:,1);

        if length(unique(layout_seg))==1
            continue
        end

        
        for n = 1:rc
            tmp_label = mode(layout_seg(visiable_area==n));
            if tmp_label ~= n && mode(visiable_area(layout_seg==n)) ~= n
                m(n) = m(n)  + 1;
                tmp_layer = sort_map(:,:,m(n));
                layout_seg(layout_seg==tmp_label) = tmp_layer(layout_seg==tmp_label);
            end
        end


        score(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]);
        tmp_map{pm} = filled_map;
        tmp_seg{pm} = layout_seg;
        tmp_rc{pm} = rc;
        tmp_model{pm} = model;
    end

    best_pm = find(score==max(score),1);
    filled_map = tmp_map{best_pm};
    layout_seg = tmp_seg{best_pm};
    rc = tmp_rc{best_pm};
    model = tmp_model{best_pm};
    

    layout_depth = zeros(size(layout_seg));
    for n = 1:rc
        tmp_depth_layer = filled_map(:,:,n);
        layout_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
    end

    
%         [model,point] = gen_param_point(layout_depth, layout_seg);
        point = gen_param_point1(layout_depth, layout_seg, model);

        result{i}.point = point;
        result{i}.layout = layout_seg;
    catch
        result{i}.point = result{i-1}.point;
        result{i}.layout = result{i-1}.layout;
    end


end


save result_regioncent_mp3d_v0 result

% [ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc(result, validation);



