import numpy as np

def my_reach(A, v, S, R=None, visited=None):
    """
    Direct port of my_reach.m
    R = my_reach(A, v, S)

    A : (n x n) array-like
    v : int, current node (1-indexed, MATLAB-style)
    S : set/list of int, allowed pass-through nodes (1-indexed)
    """
    A = np.asarray(A)
    n = A.shape[1]

    if R is None and visited is None:      # nargin == 3
        R = []
        visited = [False] * n

    visited[v - 1] = True

    # edges = find(abs(A(:,v)) > 0)  -- neighbors of v, 1-indexed, ascending
    edges = [i + 1 for i in range(A.shape[0]) if abs(A[i, v - 1]) > 0]

    if len(S) == 0:
        R = sorted(set(edges) - {v})       # R = setdiff(edges, v)
        return R, visited

    for w in edges:
        if not visited[w - 1]:
            if w not in S:                 # not(ismember(S,w))
                R.append(w)
            else:
                R, visited = my_reach(A, w, S, R, visited)

    return R, visited