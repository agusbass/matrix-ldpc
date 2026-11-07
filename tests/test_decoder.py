import numpy as np

from matrixldpc.decoder import decode_min_sum


def test_no_error_converges_immediately():
    # Single parity check on 4 bits: codeword [0,0,0,0] is valid (even weight).
    H = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    llr = np.array([5.0, 5.0, 5.0, 5.0])

    c_hat, iters, converged = decode_min_sum(H, llr, max_iter=5)

    assert converged
    assert iters == 1
    assert np.array_equal(c_hat, np.zeros(4, dtype=np.uint8))


def test_corrects_single_flip_single_check():
    # codeword [0,0,0,0], bit 0 flipped to 1 (low confidence) -> should be
    # corrected back to the all-zero codeword.
    H = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    llr = np.array([-5.0, 5.0, 5.0, 5.0])

    c_hat, iters, converged = decode_min_sum(H, llr, max_iter=5)

    assert converged
    assert np.array_equal(c_hat, np.zeros(4, dtype=np.uint8))


def test_converged_flag_matches_actual_syndrome():
    rng = np.random.default_rng(7)
    H = rng.integers(0, 2, size=(6, 12), dtype=np.uint8)
    llr = rng.normal(0, 2, size=12)

    c_hat, iters, converged = decode_min_sum(H, llr, max_iter=10)

    actual = not np.any(np.dot(H, c_hat) % 2)
    assert converged == actual
    assert 1 <= iters <= 10
    assert c_hat.shape == (12,)
    assert c_hat.dtype == np.uint8
