function [ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc( result, data )
%EVALUATIONFUNC EVALUATE PIXELWISE ACCURACY AND CORNERWISE ACCURACY
%   result: prediciton
%   data: ground truth
GlobalParameters;
allPtError = zeros(length(result),1);
allPxError = zeros(length(result),1);

for i = 1:length(result)
    if isfield(data(i), 'point')
        allPtError(i) = cornerError(result{i}.point, data(i).point(:,1:3), data(i).resolution);
    else
        allPtError(i) = nan;
    end
%     layout_path = sprintf(LAYOUT_PATTERN,data(i).image);
    layout_path = ['/home/ps/data/Z/zz_attention/dataset/matterport_layout/testing/testing(test)/layout_seg/' num2str(i,'%04d') '.png'];
    if exist(layout_path, 'file')
        %load(layout_path);
        layout=imread(layout_path);
        allPxError(i) = 1 - pixelwiseAccuracy(result{i}.layout, layout, data(i).resolution);
    else
        allPxError(i) = nan;
    end
end
meanPtError = mean(allPtError);
meanPxError = mean(allPxError);

end

