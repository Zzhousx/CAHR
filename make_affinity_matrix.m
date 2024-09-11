function [affinity_matrix, distance_matrix] = make_affinity_matrix(data, metric)
% Construct the weight matrix (a graph) for data.
% Inputs:
%   data - a data feature matrix with each row being an instance (data point), each column representing a feature
%   metric - the metric used to measure the distance between two instances
% Outputs:
%   affinity_matrix - a matrix representing the similarity between data points
%   distance_matrix - a matrix representing the distance (dissimilarity) between data points
% data=data+eps;
[N, M] = size(data);
if strcmp(metric, 'cosine')
    distance_matrix = pdist2(data, data, 'cosine');
    1
elseif strcmp(metric, 'original')
    distance_matrix = data;
    2
elseif strcmp(metric, 'euclidean')
    distance_matrix = pdist2(data, data, 'squaredeuclidean');
%       distance_matrix = L2_distance_1(data', data');
%     distance_matrix = pdist2(data, data, cosine);
else
    error('unknown metric')
end 


sigma = mean(mean(distance_matrix));
% sigma = median(dist);
affinity_matrix = exp(-distance_matrix/(2*sigma));

for i=1:N
    affinity_matrix(i, i) = 0;
end

end