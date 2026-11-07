"""GF(2) linear algebra: row-reduction and systematic generator matrix
derivation from a binary parity-check matrix H.
"""

import numpy as np


def gf2_rref(H: np.ndarray):
    """Row-reduce `H` to reduced row-echelon form over GF(2).

    Returns (R, pivot_cols):
      R          -- (rank x n) matrix in RREF (all-zero rows dropped)
      pivot_cols -- list of length `rank`, pivot_cols[i] is the pivot
                    column of row i of R
    """
    R = (H.copy() % 2).astype(np.uint8)
    m, n = R.shape
    pivot_cols = []
    pivot_row = 0
    for col in range(n):
        if pivot_row == m:
            break
        nz = np.nonzero(R[pivot_row:, col])[0]
        if nz.size == 0:
            continue
        sel = pivot_row + nz[0]
        if sel != pivot_row:
            R[[pivot_row, sel]] = R[[sel, pivot_row]]
        for r in range(m):
            if r != pivot_row and R[r, col]:
                R[r] ^= R[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
    return R[:pivot_row], pivot_cols


def build_generator(H: np.ndarray):
    """Derive a systematic generator matrix G (k x n) from parity-check
    matrix H (m x n) over GF(2), satisfying (H @ G.T) % 2 == 0.

    Returns (G, info_cols, rank):
      G         -- (k x n) generator matrix, k = n - rank
      info_cols -- length-k list of codeword column indices holding the
                   message bits directly (G[:, info_cols] == I_k)
      rank      -- rank(H) over GF(2)
    """
    n = H.shape[1]
    R, pivot_cols = gf2_rref(H)
    rank = len(pivot_cols)
    pivot_set = set(pivot_cols)
    info_cols = [j for j in range(n) if j not in pivot_set]
    k = len(info_cols)

    G = np.zeros((k, n), dtype=np.uint8)
    for a, j in enumerate(info_cols):
        G[a, j] = 1
    for i, p in enumerate(pivot_cols):
        for a, j in enumerate(info_cols):
            G[a, p] = R[i, j]
    return G, info_cols, rank


def verify_generator(H: np.ndarray, G: np.ndarray) -> bool:
    """Return True iff every row of G is in the null space of H, i.e.
    (H @ G.T) % 2 == 0."""
    return bool(np.all((H.astype(np.uint64) @ G.T.astype(np.uint64)) % 2 == 0))
