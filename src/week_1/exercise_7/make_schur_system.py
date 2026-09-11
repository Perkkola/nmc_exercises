"""Python translation of make_schur_system.m
Translated with Claude.

Creates a block matrix system related to a 2D finite element method:
P1 (linear) elements on a uniform right-triangle triangulation of the unit
square, assembling the Laplacian stiffness matrix, then reordering the
degrees of freedom as [interior nodes ; boundary nodes] so that

    K = [[A11, A12],
         [A21, A22]]

A11 is the interior-interior block (nonsingular, so the Schur complement
S = A22 - A21 @ inv(A11) @ A12 is well defined). The full K is singular
(constants are in its nullspace), which is expected for a pure Neumann /
unconstrained stiffness matrix.
"""

import numpy as np
import scipy.sparse as sp


def make_schur_system(n: int, dense: bool = False):
    """Assemble the FEM block system on an n x n grid of nodes.

    Parameters
    ----------
    n : int
        Number of nodes per direction. Total dofs = n**2, of which
        (n-2)**2 are interior and 4*n-4 are boundary.
    dense : bool
        If True return numpy arrays instead of scipy sparse matrices.

    Returns
    -------
    A11, A12, A21, A22, K
        Blocks and the reordered global matrix K = [[A11, A12], [A21, A22]].
    """
    # ---------------------------------------------------------------- MESH
    # MATLAB meshgrid(1:n, 1:n): Ix[i, j] = j+1, Iy[i, j] = i+1.
    Ix, Iy = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))

    Nx = Ix.shape[0]
    # MATLAB's A(:) is column-major, hence order="F".
    p = np.vstack([Ix.ravel(order="F"), Iy.ravel(order="F")]) / n

    # Drop the last row/column: one lower-left corner per grid cell.
    Ixc = Ix[:-1, :-1].ravel(order="F")
    Iyc = Iy[:-1, :-1].ravel(order="F")

    # Two triangles per cell, split along the (Iy,Ix)--(Iy+1,Ix+1) diagonal.
    tt1 = np.vstack([Iyc + Nx * (Ixc - 1),
                     Iyc + Nx * Ixc,
                     Iyc + 1 + Nx * Ixc])

    tt2 = np.vstack([Iyc + Nx * (Ixc - 1),
                     Iyc + 1 + Nx * (Ixc - 1),
                     Iyc + 1 + Nx * Ixc])

    # 3 x Nt connectivity. Converted from MATLAB 1-based to 0-based indexing.
    t = np.hstack([tt1, tt2]) - 1

    # ------------------------------------------------------- AFFINE MAPPING
    Ax = np.column_stack([p[0, t[1]] - p[0, t[0]],
                          p[0, t[2]] - p[0, t[0]]])

    Ay = np.column_stack([p[1, t[1]] - p[1, t[0]],
                          p[1, t[2]] - p[1, t[0]]])

    # bx, by from the MATLAB source are the element offsets; unused below.
    detA = -Ax[:, 1] * Ay[:, 0] + Ax[:, 0] * Ay[:, 1]

    # Rows of the inverse-transpose Jacobian, one row per element.
    Px = np.column_stack([Ay[:, 1], -Ay[:, 0]]) / detA[:, None]
    Py = np.column_stack([-Ax[:, 1], Ax[:, 0]]) / detA[:, None]

    # ------------------------------------------------------ MATRIX ASSEMBLY
    # The bilinear form only touches the gradients; U and V are kept to mirror
    # the MATLAB signature.
    def bilin(U, V, dU, dV):
        return dU[0] * dV[0] + dU[1] * dV[1]

    # Midpoint quadrature rule on the reference triangle: 3 points, weight 1/6.
    X = np.array([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5]])
    W = np.ones(3) / 6.0

    L = [1.0 - X[0] - X[1], X[0], X[1]]

    dL = [np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]),
          np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
          np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]

    Nt = t.shape[1]
    Ndof = p.shape[1]

    iind = np.zeros(9, dtype=int)
    jind = np.zeros(9, dtype=int)
    kk = np.zeros((9, Nt))

    mind = 0
    for i in range(3):
        Li = L[i]                      # broadcasts against the Nt x 3 blocks
        dLi = (Px @ dL[i], Py @ dL[i])  # each Nt x 3

        for j in range(3):
            Lj = L[j]
            dLj = (Px @ dL[j], Py @ dL[j])

            # Keep track of indices.
            iind[mind] = i
            jind[mind] = j

            kk[mind] = (bilin(Lj, Li, dLj, dLi) @ W) * np.abs(detA)
            mind += 1

    # coo_matrix sums duplicate entries, like MATLAB's sparse().
    K = sp.coo_matrix((kk.ravel(),
                       (t[iind].ravel(), t[jind].ravel())),
                      shape=(Ndof, Ndof)).tocsr()

    # ------------------------------------------------------- BOUNDARY NODES
    # Count how many triangles each node belongs to: 6 for interior nodes.
    counts = np.bincount(t.ravel(), minlength=Ndof)
    bind = np.flatnonzero(counts < 6)   # boundary
    nind = np.flatnonzero(counts == 6)  # interior (MATLAB reuses `iind` here)

    A11 = K[nind, :][:, nind]
    A12 = K[nind, :][:, bind]
    A21 = K[bind, :][:, nind]
    A22 = K[bind, :][:, bind]

    K = sp.bmat([[A11, A12], [A21, A22]], format="csr")

    if dense:
        return (A11.toarray(), A12.toarray(), A21.toarray(),
                A22.toarray(), K.toarray())
    return A11, A12, A21, A22, K


if __name__ == "__main__":
    n = 12
    A11, A12, A21, A22, K = make_schur_system(n, dense=True)

    n_int, n_bnd = A11.shape[0], A22.shape[0]
    print(f"n = {n}: {n_int} interior dofs (expect {(n - 2) ** 2}), "
          f"{n_bnd} boundary dofs (expect {4 * n - 4})")

    print("K symmetric                :", np.allclose(K, K.T))
    print("K @ ones == 0 (const. mode):", np.allclose(K @ np.ones(K.shape[0]), 0))
    print("A11 eigenvalues > 0        :", np.linalg.eigvalsh(A11).min() > 0)

    # On this triangulation the P1 Laplacian reproduces the 5-point stencil.
    row = A11[n_int // 2]
    print("A11 stencil values         :", np.unique(np.round(row[row != 0], 12)))

    S = A22 - A21 @ np.linalg.solve(A11, A12)
    print("Schur complement shape     :", S.shape)
    print("S symmetric                :", np.allclose(S, S.T))
