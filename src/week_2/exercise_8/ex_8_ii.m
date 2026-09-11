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

% Cholesky factor of A2
L_orig = chol(A2,'lower');
nnz_orig = nnz(L_orig);

% Cholesky factor of P'*A2*P with my_md permutation
p = my_md(A2);
P = eye(n);
P = P(:,p);

A2_permuted = P'*A2*P;         
L_md = chol(A2_permuted,'lower');
nnz_md = nnz(L_md);

fprintf('Nonzeros in Cholesky factor of A2:     %d\n', nnz_orig);
fprintf('Nonzeros in Cholesky factor of P''*A2*P:  %d\n', nnz_md);

