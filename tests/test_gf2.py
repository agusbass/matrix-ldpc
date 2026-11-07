import numpy as np

from matrixldpc.codes.gf2 import gf2_rref, build_generator, verify_generator


def test_gf2_rref_full_rank():
    H = np.array([
        [1, 1, 1, 1],
    ], dtype=np.uint8)
    R, pivots = gf2_rref(H)
    assert R.shape[0] == 1
    assert pivots == [0]


def test_build_generator_with_redundant_row():
    # row2 = row0 XOR row1 -> rank should be 2, not 3.
    H = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ], dtype=np.uint8)

    G, info_cols, rank = build_generator(H)

    assert rank == 2
    assert info_cols == [2]
    assert G.shape == (1, 3)
    assert verify_generator(H, G)


def test_build_generator_random_matrices():
    rng = np.random.default_rng(42)
    for _ in range(10):
        H = rng.integers(0, 2, size=(8, 16), dtype=np.uint8)
        G, info_cols, rank = build_generator(H)

        assert G.shape == (16 - rank, 16)
        assert len(info_cols) == 16 - rank
        assert verify_generator(H, G)

        # G[:, info_cols] must be the identity matrix.
        identity = G[:, info_cols]
        assert np.array_equal(identity, np.eye(len(info_cols), dtype=np.uint8))
