"""Layered min-sum belief-propagation decoder.

Mapping (mapping.html, domain "Error Correcting Codes"):
  Diag Ring (forward traversal)  -> forward layered decoding sweep
  Anti Ring (reverse traversal)  -> reverse layered decoding sweep
  Infinity portal (digital root convergence) -> iteration cap / convergence
                                                  test (syndrome == 0)

This is the standard "layered" (a.k.a. shuffled / serial-schedule) min-sum
algorithm: check-to-variable messages R are updated one check node at a
time and immediately folded back into the running variable LLRs, which
converges faster than flooding schedules. Even iterations sweep checks
0..m-1 (Diag Ring), odd iterations sweep m-1..0 (Anti Ring).
"""

import numpy as np


def decode_min_sum(H: np.ndarray, channel_llr: np.ndarray, max_iter: int = 20):
    """Decode `channel_llr` against parity-check matrix `H`.

    Returns (codeword, iterations, converged):
      codeword   -- (n,) hard-decision bits (uint8)
      iterations -- number of sweeps performed
      converged  -- True iff H @ codeword == 0 (mod 2) was reached
    """
    m, n = H.shape
    check_neighbors = [np.flatnonzero(H[i]) for i in range(m)]

    # check -> variable messages, keyed by (check_index, var_index)
    R = {(i, j): 0.0 for i in range(m) for j in check_neighbors[i]}

    llr = channel_llr.astype(np.float64).copy()

    def syndrome_ok(c):
        return not np.any(np.dot(H, c) % 2)

    for it in range(max_iter):
        order = range(m) if it % 2 == 0 else range(m - 1, -1, -1)
        for i in order:
            neigh = check_neighbors[i]
            q = np.array([llr[j] - R[(i, j)] for j in neigh])
            signs = np.sign(q)
            signs[signs == 0] = 1.0
            mags = np.abs(q)
            total_sign = np.prod(signs)

            for idx, j in enumerate(neigh):
                other_sign = total_sign * signs[idx]
                if mags.size > 1:
                    other_mags = np.delete(mags, idx)
                    min_mag = np.min(other_mags)
                else:
                    min_mag = 0.0
                new_r = other_sign * min_mag
                llr[j] += new_r - R[(i, j)]
                R[(i, j)] = new_r

        c_hat = (llr < 0).astype(np.uint8)
        if syndrome_ok(c_hat):
            return c_hat, it + 1, True

    c_hat = (llr < 0).astype(np.uint8)
    return c_hat, max_iter, syndrome_ok(c_hat)
