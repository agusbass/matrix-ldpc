import numpy as np

from matrixldpc.codes.qc import circulant_permutation, expand_qc, FISIK_RING


def test_circulant_permutation_is_permutation_matrix():
    for shift in range(4):
        cpm = circulant_permutation(shift, 4)
        assert cpm.shape == (4, 4)
        assert np.all(cpm.sum(axis=0) == 1)
        assert np.all(cpm.sum(axis=1) == 1)


def test_expand_qc_shape_and_block_structure():
    base = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    lift = 4

    H = expand_qc(base, lift)
    assert H.shape == (base.shape[0] * lift, base.shape[1] * lift)

    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            block = H[i * lift:(i + 1) * lift, j * lift:(j + 1) * lift]
            if base[i, j]:
                # Each "1" entry becomes a circulant permutation matrix.
                assert np.all(block.sum(axis=0) == 1)
                assert np.all(block.sum(axis=1) == 1)
            else:
                assert np.all(block == 0)


def test_expand_qc_preserves_row_and_column_weight():
    base = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)
    lift = 5

    H = expand_qc(base, lift)
    base_row_weights = base.sum(axis=1)
    base_col_weights = base.sum(axis=0)

    for i in range(base.shape[0]):
        for r in range(lift):
            assert H[i * lift + r].sum() == base_row_weights[i]

    for j in range(base.shape[1]):
        for c in range(lift):
            assert H[:, j * lift + c].sum() == base_col_weights[j]


def test_fisik_ring_values():
    assert FISIK_RING == (1, 2, 4, 8, 7, 5)
