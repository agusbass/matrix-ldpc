"""Protograph / base-matrix construction.

Mapping (mapping.html, domain "Error Correcting Codes", 10/10):
  Grid %m  -> H(i,j) = 1 iff (i+j) % modulus == 0   (parity-check matrix)
  Value    -> variable node / check node
  Grid->Val -> Tanner graph connection

This is the literal 2D variant of the Matrix Graph formula `(i+j) % modulus`,
applied to a (num_checks x num_vars) base matrix instead of the 11x11x11
3D grid -- the same residue-class connectivity rule, at code-design scale.

3 Layer = multi-rate code (mapping.html): the "Luar" layer (modulus=10)
and "Dalam" layer (modulus=20) are each built as a base matrix and stacked
to form the combined protograph (see codes.layers.build_base_matrix_stack).
"""

import numpy as np

LUAR_MODULUS = 10
DALAM_MODULUS = 20


def build_base_matrix(num_checks: int, num_vars: int, modulus: int) -> np.ndarray:
    """Return a (num_checks x num_vars) 0/1 base matrix where
    base[i, j] = 1 iff (i + j) % modulus == 0.
    """
    rows = np.arange(num_checks).reshape(-1, 1)
    cols = np.arange(num_vars).reshape(1, -1)
    return ((rows + cols) % modulus == 0).astype(np.uint8)


def build_layered_base_matrix(num_vars: int) -> np.ndarray:
    """Stack the Luar (mod 10) and Dalam (mod 20) base matrices on the same
    `num_vars` columns -- the "3 Layer = multi-rate code" mapping.

    `num_vars` must be a multiple of DALAM_MODULUS (20) so that every
    column has exactly one Luar check and one Dalam check (variable
    degree = 2 from the protograph alone).

    Luar block:  num_checks = LUAR_MODULUS  (10)  -> check degree = num_vars / 10
    Dalam block: num_checks = DALAM_MODULUS (20)  -> check degree = num_vars / 20
    """
    if num_vars % DALAM_MODULUS != 0:
        raise ValueError(f"num_vars must be a multiple of {DALAM_MODULUS}")

    luar = build_base_matrix(LUAR_MODULUS, num_vars, LUAR_MODULUS)
    dalam = build_base_matrix(DALAM_MODULUS, num_vars, DALAM_MODULUS)
    return np.vstack([luar, dalam])
