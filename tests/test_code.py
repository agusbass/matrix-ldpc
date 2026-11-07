import numpy as np
import pytest

from matrixldpc.codes.code import LDPCCode
from matrixldpc.channel import bsc_transmit, bsc_llr


def test_default_code_shapes_and_rate():
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)

    assert code.H.shape == (30 * 4, 40 * 4)  # (10+20) checks x 40 vars, lifted by 4
    assert code.G.shape == (code.k, code.n)
    assert 0 < code.rate < 1


def test_gateway_columns_from_lo_shu():
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)
    # Lo Shu values 1-9 -> 0-indexed columns 0-8 (n=160 > 9, no wraparound).
    assert code.gateway_columns() == list(range(9))


@pytest.fixture(scope="module")
def small_code():
    return LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=2)


def test_encode_produces_valid_codeword(small_code):
    rng = np.random.default_rng(1)
    msg = rng.integers(0, 2, size=small_code.k, dtype=np.uint8)
    cw = small_code.encode(msg)

    assert cw.shape == (small_code.n,)
    assert np.all((small_code.H @ cw) % 2 == 0)
    assert np.array_equal(small_code.extract_message(cw), msg)


def test_encode_rejects_wrong_length(small_code):
    with pytest.raises(ValueError):
        small_code.encode(np.zeros(small_code.k + 1, dtype=np.uint8))


def test_decode_noise_free_roundtrip(small_code):
    rng = np.random.default_rng(2)
    msg = rng.integers(0, 2, size=small_code.k, dtype=np.uint8)
    cw = small_code.encode(msg)

    llr = bsc_llr(cw, p=1e-3)  # very confident, no actual flips
    decoded_cw, decoded_msg, _, converged = small_code.decode(llr, max_iter=10)

    assert converged
    assert np.array_equal(decoded_cw, cw)
    assert np.array_equal(decoded_msg, msg)


def test_decode_corrects_single_bit_error(small_code):
    rng = np.random.default_rng(3)
    msg = rng.integers(0, 2, size=small_code.k, dtype=np.uint8)
    cw = small_code.encode(msg)

    received = cw.copy()
    received[0] ^= 1  # flip one coded bit
    llr = bsc_llr(received, p=0.05)

    decoded_cw, decoded_msg, _, converged = small_code.decode(llr, max_iter=20)

    assert converged
    assert np.array_equal(decoded_cw, cw)
    assert np.array_equal(decoded_msg, msg)
