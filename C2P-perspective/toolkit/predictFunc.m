function [ result ] = predictFunc( data )
%ALGORITHM PERFORM PREDICTION FOR A BATCH OF DATA
%   data can be testing or validation

GlobalParameters;
result = cell(length(data),1);
for i = 1:length(data)
    img = imread(sprintf(IMAGE_PATTERN, data(i).image));
    % a fake method
    result{i} = algorithmFunc(img);
    % a cheating method
%     tmp = data(i);
%     tmp.layout = getSegmentation(data(i));
%     result{i} = tmp;
end

end

