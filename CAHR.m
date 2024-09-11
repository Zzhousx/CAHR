function [W,obj] = CAHR(X,param)
%% ===================== Parameters =====================
lambda1 = param.lambda1;
lambda2 = param.lambda2;
lambda4 = param.lambda4;
cross_b=1e-6;
self_b=1e1;
NITER = param.NITER;
n = param.n;
v = param.v;
c = param.c;
Eigen_NUM=c;
knn=15;
pn=15;
%% ===================== initialize =====================
W=cell(1,v);
E=cell(1,v);
I_n=eye(n);

Beta = cross_b*ones(v) - diag(cross_b*ones(1,v)) + diag(self_b*ones(1,v));

Z=cell(1,v);
H=cell(1,v);
A=cell(1,v);
q=cell(1,v);
Q=cell(1,v);
K_complement = zeros(n,n);
Xnor=cell(1,v);
w1 = ones(v,1)*(1/v);
%% initialize
affinity = cell(1, v);
original_affinity = cell(1, v);
idx = cell(1, v);
ONE = ones(n);
knn_idx = false(n);

for i=1:v
     W_zhou = make_affinity_matrix(X{i}', 'euclidean');
    original_affinity{i} = W_zhou;
    if knn ~= 0  % not using fully connected graph
        [W_zhou, idx{i}] = kNN(W_zhou, knn);
        [~, tp] = extract_from_idx(ONE, idx{i});
        knn_idx = knn_idx | logical(tp);  % common kNN index for all views
    end
    affinity{i} = W_zhou;
end

if knn ~= 0
    for i=1:v
        for j=1:v
            if j~=i
                [~, tp] = extract_from_idx(original_affinity{i}, idx{j});
                affinity{i} = affinity{i} + (tp + tp')/2;
            end
        end
    end
end
% make knn index symmetric
if knn~=0
    knn_idx = knn_idx | knn_idx';
else
    knn_idx = true(n);
end
for i=1:v
    Z{i}=affinity{i};
end

for vIndex=1:v
    dv = size(X{vIndex},1);
    I_d{vIndex}=eye(dv);
    Q{vIndex} = eye(dv);
    W{vIndex}=rands(dv,c);
    H{vIndex}=zeros(n,n);
    E{vIndex}=zeros(n,n);
    Xnor{vIndex}=X{vIndex};
end
%%
if nnz(Z{1})/numel(Z{1}) < 0.4  % if W contains a large proportion of zeros, use sparse mode
    for i=1:v
        Z{i} = sparse(Z{i});
    end
    sparse_mode = true;
else
    for i=1:v
        Z{i} = full(Z{i});
    end
    sparse_mode = false;
end
b=Beta;
b_coef = b + eye(v);
n = size(Z{1}, 1);
baW = cell(v,1);
special_baW = cell(v,1);
true_baW = cell(v,1);
H = cell(v,1);
B = cell(v,1);
E = cell(v,1);
up_knn_idx = triu(knn_idx);

zz = 2.^(0:v-1);
ww = 1:2^v-2; % alpha can't be all zeros, so -2
logww = log2(ww);
yy = ww(abs(floor(logww)-logww)>eps);
alpha_zeros_ones = de2bi([0,zz,yy]);
n_eye_coef = -eye(v);
% initialize A{i}, alpha, con_graph
Zcon = Z{1};
if sparse_mode
    D = sparse(n, n);
else
    D = zeros(n,n);
end
for i=1:v
    D = max(D, Z{i});
    H{i} = full(Z{i});
end
for i=1:v
    if sparse_mode
        H{i} = sparse(H{i});
    end
    H{i} = min(H{i}, D);
end
%% ===================== updating =====================
for iter = 1:NITER
    %% update Zcon and w1
    for i=1:v
        E{i} = Z{i} - H{i};
    end
    coef = zeros(v);
    coef2 = zeros(v);
    for i=1:v
        for j=i:v
            coef(i,j) = sum(sum(H{i}.*H{j}));
            coef(j,i) = coef(i,j);
            coef2(i,j) = sum(sum(E{i}.*E{j}));
            coef2(j,i) = coef2(i,j);
        end
    end
    coef2 = coef2 .* b;
    % compute coefficient for the linear equation
    H_zhou = 2*(diag(diag(coef)) - coef/v + coef2);
    one = ones(v, 1);
    
    for i=1:1
        mpl = alpha_zeros_ones(i,:);
        coef3 = H_zhou .* ~mpl + n_eye_coef .* mpl;
        X_zhou = [coef3, one; 1-mpl, 0];
        temp_b = [zeros(v,1); 1];
        if det(X_zhou) == 0 % abs(det(X)) <= eps
            solution = pinv(X_zhou)*temp_b;
        else
            solution = X_zhou \ temp_b;
        end
        w1 = EProjSimplex_new(solution(1:v));
    end
    Zcon = w1(1)*H{1};
    for j=2:v
        Zcon = Zcon + w1(j)*H{j};
    end
    Zcon = Zcon/v;   
    %% update H
    alp_coef = w1 * w1';
    coef = alp_coef .* b_coef;
    if sparse_mode
        commom_baW = sparse(n, n);
    else
        commom_baW = zeros(n,n);
    end
    for i=1:v
        baW{i} = cross_b*w1(i)*Z{i};
        special_baW{i} = self_b*w1(i)*Z{i};
        commom_baW = commom_baW + baW{i};
    end

    for i=1:v
        true_baW{i} = commom_baW-baW{i}+special_baW{i};
        temp = full(w1(i)*(Zcon + true_baW{i}));
        B{i} = temp(up_knn_idx);
    end
    right_b = cat(2, B{:})';
    if det(coef) == 0
        solution = (pinv(coef) * right_b)';
    else
        solution = (coef \ right_b)';
    end
    solution(solution<0) = 0;
    for i=1:v
        temp = solution(:,i);
        H{i} = zeros(n, n);
        H{i}(up_knn_idx) = temp;
        H{i} = max(H{i}, H{i}');
        H{i} = min(Z{i}, H{i});
        if sparse_mode
            H{i} = sparse(H{i});
        end
    end
    %% update L_hyper
       if iter == 1
          Weight = constructW_PKN(Zcon, pn);
          Diag_tmp = diag(sum(Weight));
          L = Diag_tmp - Weight;
       else
          param.num_view = pn; 
          HG = gsp_nn_hypergraph(Zcon, param);
          L = HG.L;
       end
    for iterv = 1:v
   % obtain K_complement
        K_complement= K_complement*0;
        for k=1:v
            if (abs(k-iterv)>0) 
            K_complement =  K_complement + Beta(iterv,k)*w1(k)*w1(iterv)*(Z{k}-H{k});
            end
        end
    %% update W
        A{iterv} = lambda2*(Z{iterv}+Z{iterv}'-Z{iterv}'*Z{iterv})-lambda1*L;
        Mat1 = Xnor{iterv}*A{iterv}*Xnor{iterv}'-lambda4*Q{iterv}+eps;
        Mat2 = I_d{iterv}+eps;
        [W{iterv},Gen_Value]=Find_K_Max_Gen_Eigen(Mat1,Mat2,Eigen_NUM);
    %% construct l_21 norm matrix  
        qj{iterv} = sqrt(sum(W{iterv}.*W{iterv},2)+eps);
        q{iterv} = 0.5./qj{iterv};
        Q{iterv} = diag(q{iterv});
    end
% ===================== calculate obj =====================
    NormX=0;
    Term1=0;
    Term2=0;
    Term3=0;
    for objIndex=1:v
       Term1 = Term1 + trace(W{objIndex}'*(Xnor{objIndex}*(lambda2*(I_n-Z{iterv}-Z{iterv}'+Z{iterv}'*Z{iterv})+lambda1*L)*Xnor{objIndex}'+lambda4*Q{objIndex})*W{objIndex});
        for k=1:v
            if (abs(k-iterv)>0) 
            Term2 =  Term2 + Beta(objIndex,k)*w1(k)*w1(objIndex)*trace((Z{objIndex}-H{objIndex})*(Z{k}-H{k})');
            end
        end
        Term3 = Term3 + norm(Zcon-w1(objIndex)*H{objIndex},'fro').^2;
        NormX = NormX + norm(Xnor{objIndex},'fro')^2;
    end
    tempobj=Term1+Term2+Term3;
    r=3;
    HH = bsxfun(@power,tempobj, 1/(1-r));
    alpha1 = bsxfun(@rdivide,HH,sum(HH));
    alpha_r = alpha1.^r;        
    obj(iter) = (alpha_r*tempobj')/NormX;
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end
    fprintf('iteration =  %d:  obj: %.10f; err: %.8f  \n', ...
        iter, obj(iter), err);
    if (abs(err))<1e-7
        if iter > 15
            break;
        end
    end
end










