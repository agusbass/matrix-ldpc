import numpy as np
import pytest

from matrixldpc.codes.base_matrix import (
    build_base_matrix,
    build_layered_base_matrix,
    LUAR_MODULUS,
    DALAM_MODULUS,
)


def test_build_base_matrix_formula():
    base = build_base_matrix(num_checks=4, num_vars=6, modulus=3)
    expected = np.array([
        [(i + j) % 3 == 0 for j in range(6)]
        for i in range(4)
    ], dtype=np.uint8)
    assert np.array_equal(base, expected)


def test_layered_base_matrix_shape_and_rejects_bad_size():
    with pytest.raises(ValueError):
        build_layered_base_matrix(15)

    base = build_layered_base_matrix(40)
    assert base.shape == (LUAR_MODULUS + DALAM_MODULUS, 40)


def test_layered_base_matrix_degrees():
    num_vars = 40
    base = build_layered_base_matrix(num_vars)

    luar = base[:LUAR_MODULUS]
    dalam = base[LUAR_MODULUS:]

    # Each Luar check connects to num_vars / 10 variables.
    assert np.all(luar.sum(axis=1) == num_vars // LUAR_MODULUS)
    # Each Dalam check connects to num_vars / 20 variables.
    assert np.all(dalam.sum(axis=1) == num_vars // DALAM_MODULUS)

    # Every variable has exactly one Luar check and one Dalam check
    # (variable degree = 2 from the protograph alone).
    assert np.all(luar.sum(axis=0) == 1)
    assert np.all(dalam.sum(axis=0) == 1)
    assert np.all(base.sum(axis=0) == 2)
