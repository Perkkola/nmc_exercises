% Call as : R = my reach(A, v, S)
%
% A is a matrix, v is the current node, S is a vector of nodes.
%
function [R,visited] = my_reach(A, v, S, R, visited)
if(nargin == 3)
    R = [];
    visited(1:size(A,2)) = false;
end
visited(v) = true;
edges = find( abs( A(:,v)) > 0);
if( isempty(S))
    R = setdiff(edges,v);
    return;
end
for w=edges(:)'
    if( not(visited(w)))
        if( not(ismember(S,w)) )
            R = [R w];
        else
            [R,visited] = my_reach(A,w,S,R,visited);
        end
    end
end
end

% Construct a fill-in reducing permutation vector for
% A using minimum degree ordering method. (this is a naive example
% implementation)
function p = my_md(A)
n = size(A,1);
p = 1:n;
for i=1:(n-1)
    i
    % try all remaining entries as entry i
    nnzLi = zeros(1,n);
    for j=(i+1):n
        tmp = p; tmp(i) = p(j); tmp(j) = p(i);
        nnzLi(j) = length(unique(my_reach(A(tmp,tmp), i, [1:(i-1)])));
    end
    % choose permutation minimising nnz in column i.
    [~,I] = min(nnzLi((i+1):n));
    I = I(1)+i;
    pi = p(i); p(i) = p(I(1)); p(I) = pi;
end
end

clear; clc;

% Matrix from 1.36
A2 = [20  0  1  1  1  1  0;
    0 20  1  1  0  0  1;
    1  1 20  0  0  0  0;
    1  1  0 20  0  0  0;
    1  0  0  0 20  0  0;
    1  0  0  0  0 20  0;
    0  1  0  0  0  0 20];

n = size(A2,1);
A2s = sparse(A2);   % amd expects a sparse matrix

% my_md. more repetitions to get a more stable timing estimate.
nrep = 2000;

tic;
for k = 1:nrep
    p_md = my_md(A2);
    P_md = eye(n); P_md = P_md(:,p_md);
    L_md = chol(P_md'*A2*P_md,'lower');
    nnz_md = nnz(L_md);
end
t_md = toc / nrep;


% amd
tic;
for k = 1:nrep
    p_amd = amd(A2s);
    P_amd = eye(n); P_amd = P_amd(:,p_amd);
    L_amd = chol(P_amd'*A2*P_amd,'lower');
    nnz_amd = nnz(L_amd);
end
t_amd = toc / nrep;


fprintf('\n%-28s %10s %18s\n', 'Method', 'nnz', 'time');
fprintf('%-28s %10d %18.3e\n', 'my_md', nnz_md,  t_md);
fprintf('%-28s %10d %18.3e\n', 'amd',  nnz_amd, t_amd);

fprintf('\nPermutation vector my_md: '); disp(p_md);
fprintf('Permutation vector amd:   '); disp(p_amd');

