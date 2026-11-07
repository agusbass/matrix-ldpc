"""Tanner-graph structural metrics.

  Girth -- the length of the shortest cycle in the bipartite Tanner graph
  (variable nodes <-> check nodes). Short girth (the bipartite minimum is 4)
  means the belief-propagation decoder sees strongly correlated messages
  after only one or two iterations, which is the standard explanation for
  why small/regular QC-LDPC lifts (e.g. lift_size=2 in this project, see
  README "Empirical results") perform poorly.
"""

from collections import deque

import numpy as np


def compute_girth(H: np.ndarray, max_depth: int = 10) -> int | None:
    """Return the girth (shortest cycle length) of the Tanner graph of `H`.

    `H` is an (m x n) 0/1 parity-check matrix. The Tanner graph is bipartite
    with `n` variable nodes and `m` check nodes; a cycle alternates between
    the two node types, so the girth is always even and at least 4.

    BFS is run from every node up to `max_depth` hops; the shortest cycle
    found anywhere is the girth. Returns `None` if no cycle of length
    `<= 2 * max_depth` is found (girth larger than the search depth).
    """
    m, n = H.shape
    # Node ids: variable nodes 0..n-1, check nodes n..n+m-1.
    var_neighbors = [np.flatnonzero(H[:, j]) + n for j in range(n)]
    check_neighbors = [np.flatnonzero(H[i, :]) for i in range(m)]
    adjacency = var_neighbors + check_neighbors

    girth = None
    for root in range(n + m):
        dist = {root: 0}
        parent = {root: -1}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adjacency[u]:
                if v == parent[u]:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if girth is None or cycle_len < girth:
                        girth = cycle_len
        if girth == 4:
            break  # 4 is the minimum possible girth for a simple bipartite graph
    return girth


def count_4_cycles(H: np.ndarray) -> int:
    """Number of length-4 cycles in the Tanner graph of `H`.

    A 4-cycle is a pair of check nodes sharing two or more variable-node
    neighbors (check_i - var_a - check_j - var_b - check_i); each pair of
    shared variables contributes one such cycle. Computed from the check-node
    co-occurrence matrix `H @ H.T`.
    """
    H = H.astype(np.int64)
    shared = (H @ H.T)[np.triu_indices(H.shape[0], k=1)]
    return int(np.sum(shared * (shared - 1) // 2))
