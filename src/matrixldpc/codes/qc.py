"""Quasi-cyclic (QC) expansion of the protograph base matrix.

Mapping (mapping.html, domain "Error Correcting Codes"):
  Vortex Fisik ring (1->2->4->8->7->5->1) -> protograph expansion sequence

Each "1" entry in the base matrix becomes a `lift_size x lift_size`
circulant permutation matrix (CPM), shifted by a value drawn cyclically
from the Vortex Fisik ring (mod lift_size). Each "0" entry becomes a
`lift_size x lift_size` zero block.
"""

import numpy as np

# Vortex Fisik ring: 1 -> 2 -> 4 -> 8 -> 7 -> 5 -> 1
FISIK_RING = (1, 2, 4, 8, 7, 5)


def circulant_permutation(shift: int, size: int) -> np.ndarray:
    """Return the `size x size` identity matrix cyclically shifted by
    `shift` columns (a circulant permutation matrix / CPM)."""
    return np.roll(np.eye(size, dtype=np.uint8), shift, axis=1)


def expand_qc(base: np.ndarray, lift_size: int) -> np.ndarray:
    """Expand a 0/1 base (protograph) matrix into a full QC parity-check
    matrix by replacing each entry with a `lift_size x lift_size` block:
      base[i,j] == 1 -> CPM shifted by FISIK_RING[k % 6] % lift_size
      base[i,j] == 0 -> zero block

    The shift sequence advances (k += 1) for every "1" entry, scanning
    the base matrix in row-major order.
    """
    m, n = base.shape
    rows = []
    ring_idx = 0
    for i in range(m):
        row_blocks = []
        for j in range(n):
            if base[i, j]:
                shift = FISIK_RING[ring_idx % len(FISIK_RING)] % lift_size
                row_blocks.append(circulant_permutation(shift, lift_size))
                ring_idx += 1
            else:
                row_blocks.append(np.zeros((lift_size, lift_size), dtype=np.uint8))
        rows.append(np.hstack(row_blocks))
    return np.vstack(rows)
