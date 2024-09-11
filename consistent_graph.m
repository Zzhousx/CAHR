function [con_graph, alpha] = consistent_graph(W)
% Learn a consistent graph from multiple graphs.
% Inputs:
%   W - weight matrix of a graph
%   knn_idx - common kNN index for all views
% Optional Inputs:
%   tol, tol2 - the tolerance that determines convergence of algorithm
% Outputs:
%   con_graph - weight matrix of the learned unified graph
%   E - a cell matrix containing the inconsistent part of all views
%   A - a cell matrix containing the consistent part of all views
% See "Youwei Liang, Dong Huang, and Chang-Dong Wang. Consistency Meets 
% Inconsistency: A Unified Graph Learning Framework for Multi-view 
% Clustering. 2019 IEEE International Conference on Data Mining(ICDM)."
% for a detailed description of the algorithm.
% Author: Youwei Liang
% 2019/08/31
v = length(W);
if nnz(W{1})/numel(W{1}) < 0.4  % if W contains a large proportion of zeros, use sparse mode
    for i=1:v
        W{i} = sparse(W{i});
    end
    sparse_mode = true;
else
    for i=1:v
        W{i} = full(W{i});
    end
    sparse_mode = false;
end
v = length(W);
n = size(W{1}, 1);
A = cell(v,1);
E = cell(v,1);

zz = 2.^(0:v-1);
ww = 1:2^v-2; % alpha can't be all zeros, so -2
logww = log2(ww);
yy = ww(abs(floor(logww)-logww)>eps);
alpha_zeros_ones = de2bi([0,zz,yy]);
n_eye_coef = -eye(v);

% initialize A{i}, alpha, con_graph
alpha = ones(v,1) / v;
if sparse_mode
    D = sparse(n, n);
else
    D = zeros(n,n);
end
for i=1:v
    D = max(D, W{i});
    A{i} = full(W{i});
end
for i=1:v
    if sparse_mode
        A{i} = sparse(A{i});
    end
    A{i} = min(A{i}, D);
end
    % fix A{i}, update con_graph and alpha
    coef = zeros(v);
    for i=1:v
        for j=i:v
            coef(i,j) = sum(sum(A{i}.*A{j}));
            coef(j,i) = coef(i,j);
        end
    end
    
    % compute coefficient for the linear equation
    H = 2*(diag(diag(coef)) - coef/v);
    one = ones(v, 1);   
    for i=1:1
        mpl = alpha_zeros_ones(i,:);
        coef3 = H .* ~mpl + n_eye_coef .* mpl;
        X = [coef3, one; 1-mpl, 0];
        temp_b = [zeros(v,1); 1];
        if det(X) == 0 % abs(det(X)) <= eps
            fprintf('*************')
            solution = pinv(X)*temp_b;
        else
            solution = X \ temp_b;
        end
        alpha = EProjSimplex_new(solution(1:v));
    end
    con_graph = alpha(1)*A{1};
    for j=2:v
        con_graph = con_graph + alpha(j)*A{j};
    end
    con_graph = con_graph/v;


end
