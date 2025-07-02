clear all;
clc;

load /home/ubuntu/work/geolayout/dataset/Matterport3D_Layout/testing.mat
% load layout_pool


dataset = data;

mu = 1;
pos = 0;
neg = 0;
thre = 0.74;
sz = [1024,1280];
% mm = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4];
mm = [0.5, 0.6, 0.7];


for nn = 1:length(mm)
    
msd = mm(nn);
% msd = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4];
% msd = [0.15, 0.2 ,0.3, 0.4, 0.5, 0.6];
% msd = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];

final_score = [];
result = [];
tic
% for i = 57:57
% for i = 1:20
for i = 1:length(dataset)
%     for i = 1:100
    % for i = 1001:length(dataset)

    i
    
    load(['/home/ubuntu/work/geolayout/predict_param/pred_progressive_mp3d_v0/' num2str(i-1) '.mat']);
    

    
    coord_x = repmat(0:sz(2)-1,[sz(1),1])/(sz(2)-1);
    coord_y = repmat(0:sz(1)-1,[sz(2),1])'/(sz(1)-1);

    p = squeeze(param(1,1,:,:));
    q = squeeze(param(1,2,:,:));
    r = squeeze(param(1,3,:,:));
    s = squeeze(param(1,4,:,:));

    [h,w] = size(p);
    n_pix = h*w;
    cmx = repmat(0:w-1,[h,1])/(w-1);
    cmy = repmat(0:h-1,[w,1])'/(h-1);

    %     pred_invd = (p .* cmx + q .* cmy + r) .* s;
    pred_invd = squeeze(ref_invd);
    pred_d = 1./pred_invd;

    p_flip = -fliplr(squeeze(param_flip(1,1,:,:)));
    q_flip = fliplr(squeeze(param_flip(1,2,:,:)));
    r_flip = fliplr(squeeze(param_flip(1,1,:,:)).* cmx + squeeze(param_flip(1,2,:,:)).* cmy + squeeze(param_flip(1,3,:,:))) - p_flip .* cmx - q_flip .* cmy;
    s_flip = fliplr(squeeze(param_flip(1,4,:,:)));

    %     pred_invd_flip = (p_flip .* cmx + q_flip .* cmy + r_flip) .* s_flip;
    pred_invd_flip = fliplr(squeeze(ref_invd_flip));
    pred_d_flip = 1./pred_invd_flip;


    %     p = (p + p_flip)/2;
    %     q = (q + q_flip)/2;
    %     r = (r + r_flip)/2;

    data = squeeze(embedding(1,1:2,:,:));
    data = reshape(data,2,[]);
    data_flip = [];
    data_flip(1,:,:) = fliplr(squeeze(embedding_flip(1,1,:,:)));
    data_flip(2,:,:) = fliplr(squeeze(embedding_flip(1,2,:,:)));
%     data_flip(3,:,:) = fliplr(squeeze(embedding_flip(1,3,:,:)));
%     data_flip(4,:,:) = fliplr(squeeze(embedding_flip(1,4,:,:)));
    data_flip = reshape(data_flip,2,[]);

    %     data = squeeze(param(1,1:4,:,:));
    %     data = reshape(data,4,[]);
    %     data_flip = [];
    %     data_flip(1,:,:) = fliplr(squeeze(param_flip(1,1,:,:)));
    %     data_flip(2,:,:) = fliplr(squeeze(param_flip(1,2,:,:)));
    %     data_flip(3,:,:) = fliplr(squeeze(param_flip(1,3,:,:)));
    %     data_flip(4,:,:) = fliplr(squeeze(param_flip(1,4,:,:)));
    %     data_flip = reshape(data_flip,4,[]);

    score = [];
    tmp_map = [];
    tmp_seg = [];
    tmp_rc = [];
    tmp_model = [];
    tmp_depth = [];

    try


        for pm = 1:length(msd)

            [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(data,msd(pm));

            gen_seg = reshape(data2cluster,[h,w]);
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

                rc = rc+1;


                visiable_area = visiable_area + imresize(new_l,sz,'nearest') * rc;
                filled_map(:,:,rc) = 1./ ((ma*coord_x+mb*coord_y+mc)*ms + 1e-10);
                %             mod_param(rc,:) = [ma,mb,mc];
                model(rc).face = rc;
                model(rc).params = [ma,mb,mc];

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


            layout_depth = zeros(size(layout_seg));
            for n = 1:rc
                tmp_depth_layer = filled_map(:,:,n);
                layout_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
            end

            inv_layout_depth = imresize(1./(layout_depth+1e-10),[h,w]);
            dist = sort(abs(pred_invd(:) - inv_layout_depth(:)));
            valid_num = round(0.98 * length(dist));
            loss_depth = mean(dist(1:valid_num));



            %             score(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]);
            score(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]) - mu * loss_depth;
            tmp_map{pm} = filled_map;
            tmp_seg{pm} = layout_seg;
            tmp_rc{pm} = rc;
            tmp_model{pm} = model;
            tmp_depth{pm} = layout_depth;
        end

        best_pm = find(score==max(score),1);
        %     filled_map = tmp_map{best_pm};
        %     layout_seg = tmp_seg{best_pm};
        %     rc = tmp_rc{best_pm};
        %     model = tmp_model{best_pm};
        %
        %
        %     layout_depth = zeros(size(layout_seg));
        %     for n = 1:rc
        %         tmp_depth_layer = filled_map(:,:,n);
        %         layout_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
        %     end




        score_flip = [];
        tmp_map_flip = [];
        tmp_seg_flip = [];
        tmp_rc_flip = [];
        tmp_model_flip = [];
        tmp_depth_flip = [];
        %     msd = [0.15, 0.2, 0.3, 0.4, 0.5];


        for pm = 1:length(msd)

            [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(data_flip,msd(pm));

            gen_seg = reshape(data2cluster,[h,w]);
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


                ma = trimmean(p_flip(new_l),30);
                mb = trimmean(q_flip(new_l),30);
                mc = trimmean(r_flip(new_l),30);
                ms = trimmean(s_flip(new_l),30);

                rc = rc+1;


                visiable_area = visiable_area + imresize(new_l,sz,'nearest') * rc;
                filled_map(:,:,rc) = 1./ ((ma*coord_x+mb*coord_y+mc)*ms + 1e-10);
                %             mod_param(rc,:) = [ma,mb,mc];
                model(rc).face = rc;
                model(rc).params = [ma,mb,mc];

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

            layout_depth = zeros(size(layout_seg));
            for n = 1:rc
                tmp_depth_layer = filled_map(:,:,n);
                layout_depth(layout_seg==n) = tmp_depth_layer(layout_seg==n);
            end

            inv_layout_depth = imresize(1./(layout_depth+1e-10),[h,w]);
            dist = sort(abs(pred_invd_flip(:) - inv_layout_depth(:)));
            %         valid_num = round(0.98 * length(dist));
            loss_depth_flip = mean(dist(1:valid_num));


            score_flip(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]) - mu * loss_depth_flip;
            %         score_flip(pm) = pixelwiseAccuracy(imresize(layout_seg,[h,w],'nearest'), gen_seg, [h,w]);
            tmp_map_flip{pm} = filled_map;
            tmp_seg_flip{pm} = layout_seg;
            tmp_rc_flip{pm} = rc;
            %             tmp_model_flip{pm} = model;
            tmp_depth_flip{pm} = layout_depth;
        end

        best_pm_flip = find(score_flip==max(score_flip),1);

        if score(best_pm) > score_flip(best_pm_flip)
            %         0

            pos = pos + 1;
            filled_map = tmp_map{best_pm};
            layout_seg = tmp_seg{best_pm};
            rc = tmp_rc{best_pm};
            %             model = tmp_model{best_pm};
            layout_depth = tmp_depth{best_pm};

        else
            %         1
            neg = neg + 1;
            filled_map = tmp_map_flip{best_pm_flip};
            layout_seg = tmp_seg_flip{best_pm_flip};
            rc = tmp_rc_flip{best_pm_flip};
            %             model = tmp_model_flip{best_pm_flip};
            layout_depth = tmp_depth_flip{best_pm_flip};

        end

        [model,point] = gen_param_point(layout_depth, layout_seg);
%         final_score(i) = max(score(best_pm),score_flip(best_pm_flip));


%     catch
%         final_score(i) = 0;
%     end
% 
%     if final_score(i) < thre
% 
%         %             if score(best_pm) > score_flip(best_pm_flip)
%         %                 %         0
%         %
%         %                 pos = pos + 1;
%         %                 filled_map = tmp_map{best_pm};
%         %                 layout_seg = tmp_seg{best_pm};
%         %                 rc = tmp_rc{best_pm};
%         %                 %             model = tmp_model{best_pm};
%         %                 layout_depth = tmp_depth{best_pm};
%         %
%         %             else
%         %                 %         1
%         %                 neg = neg + 1;
%         %                 filled_map = tmp_map_flip{best_pm_flip};
%         %                 layout_seg = tmp_seg_flip{best_pm_flip};
%         %                 rc = tmp_rc_flip{best_pm_flip};
%         %                 %             model = tmp_model_flip{best_pm_flip};
%         %                 layout_depth = tmp_depth_flip{best_pm_flip};
%         %
%         %             end
% 
% 
% 
%         %         else
%         sim_score = inf;
%         siw = 0;
%         for lpi = 1:length(layout_pool)
% 
%             tmp_invd = layout_pool(lpi).invd;
%             tmp_invd = imresize(tmp_invd,[h,w]);
%             %             di = tmp_invd - pred_invd;
%             di = 1./tmp_invd - 1./pred_invd;
%             di2 = di.^2;
%             di_flip = 1./tmp_invd - 1./pred_invd_flip;
%             di_flip2 = di_flip.^2;
%             si_loss = trimmean(di2(:),2)+trimmean(di_flip2(:),2);
% 
%             if si_loss < sim_score
%                 sim_score = si_loss;
%                 layout_depth = double(layout_pool(lpi).layout_depth) / 4000;
%                 layout_seg = layout_pool(lpi).layout_seg;
%             end
% 
% 
% 
%             tmp_invd = fliplr(layout_pool(lpi).invd);
%             tmp_invd = imresize(tmp_invd,[h,w]);
%             %             di = tmp_invd - pred_invd;
%             di = 1./tmp_invd - 1./pred_invd;
%             di2 = di.^2;
%             di_flip = 1./tmp_invd - 1./pred_invd_flip;
%             di_flip2 = di_flip.^2;
%             si_loss = trimmean(di2(:),2)+trimmean(di_flip2(:),2);
%             %             ft = sum(di2(:)) / n_pix;
%             %             st = siw * sum(di(:))^2 / (n_pix^2);
%             %             si_loss = ft - st;
%             if si_loss < sim_score
%                 sim_score = si_loss;
%                 layout_depth = fliplr(double(layout_pool(lpi).layout_depth) / 4000);
%                 %                 layout_depth = layout_depth * trimmean(tmp_invd(:),5) / trimmean(pred_invd(:),5);
%                 layout_seg = fliplr(layout_pool(lpi).layout_seg);
%             end
%         end
% 
%         [model,point] = gen_param_point(layout_depth, layout_seg);
%     end

%         point = gen_param_point1(layout_depth, layout_seg, model);

    result{i}.point = point;
    result{i}.layout = layout_seg;
   
    catch
            
            result{i}.point = result{i-1}.point;
            result{i}.layout = result{i-1}.layout;
            
    end
        
    
%         tmp = uint16(layout_depth * 4000);
%         imwrite(tmp,['/home/zwd/work/geolayout/code/result/depth_mp3d/' num2str(i,'%04d') '_layout.png'])


    %     tmp1 = uint16(4000./(imresize(pred_invd,sz)+1e-10));
    %     imwrite(tmp1,['/home/zwd/work/geolayout/code/result/depth_raw_mp3d/' num2str(i,'%04d') '_layout.png'])

end

rt = toc

pos = pos / i;
neg = neg / i;

% fname = ['result_mp3d_msd_' num2str(msd)];

save(['result_mp3d_msd_' num2str(msd) '.mat'],'result','rt')



end


% save(['result_z0_mp3d_v' num2str(mm) '.mat'],'result')
% end


% load('result_mp3d_01.mat')
% xx = result;
% clear result
% load('result_mp3d_02.mat')
% for i = 1:1000
%     result{i} = xx{i};
% end
% clear xx
% save result_mp3d_01 result


