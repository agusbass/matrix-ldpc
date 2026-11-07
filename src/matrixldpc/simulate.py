"""BER simulation harness over BSC and AWGN channels."""

import numpy as np

from matrixldpc.codes.code import LDPCCode
from matrixldpc.channel import bsc_transmit, bsc_llr, awgn_transmit, awgn_llr


def simulate_bsc(code: LDPCCode, p_values, num_trials: int = 100,
                  max_iter: int = 20, seed: int | None = None) -> dict:
    """Run `num_trials` random codewords through a BSC(p) for each crossover
    probability `p` in `p_values`.

    Returns {p: {"coded_ber", "uncoded_ber", "frame_error_rate", "avg_iterations"}}.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for p in p_values:
        bit_errors = 0
        uncoded_bit_errors = 0
        frame_errors = 0
        total_iters = 0

        for _ in range(num_trials):
            msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
            cw = code.encode(msg)
            received = bsc_transmit(cw, p, rng)
            llr = bsc_llr(received, p)
            _, decoded_msg, iters, _ = code.decode(llr, max_iter)

            bit_errors += int(np.sum(decoded_msg != msg))
            uncoded_bit_errors += int(np.sum(received != cw))
            total_iters += iters
            if not np.array_equal(decoded_msg, msg):
                frame_errors += 1

        results[p] = {
            "coded_ber": bit_errors / (num_trials * code.k),
            "uncoded_ber": uncoded_bit_errors / (num_trials * code.n),
            "frame_error_rate": frame_errors / num_trials,
            "avg_iterations": total_iters / num_trials,
        }
    return results


def simulate_awgn(code: LDPCCode, snr_db_values, num_trials: int = 100,
                   max_iter: int = 20, seed: int | None = None) -> dict:
    """Run `num_trials` random codewords through an AWGN+BPSK channel for
    each Eb/N0 (dB) value in `snr_db_values`.

    Returns {snr_db: {"coded_ber", "uncoded_ber", "frame_error_rate", "avg_iterations"}}.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for snr_db in snr_db_values:
        bit_errors = 0
        uncoded_bit_errors = 0
        frame_errors = 0
        total_iters = 0

        for _ in range(num_trials):
            msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
            cw = code.encode(msg)
            signal, sigma = awgn_transmit(cw, snr_db, rng)
            llr = awgn_llr(signal, sigma)
            _, decoded_msg, iters, _ = code.decode(llr, max_iter)

            received_bits = (signal < 0).astype(np.uint8)
            bit_errors += int(np.sum(decoded_msg != msg))
            uncoded_bit_errors += int(np.sum(received_bits != cw))
            total_iters += iters
            if not np.array_equal(decoded_msg, msg):
                frame_errors += 1

        results[snr_db] = {
            "coded_ber": bit_errors / (num_trials * code.k),
            "uncoded_ber": uncoded_bit_errors / (num_trials * code.n),
            "frame_error_rate": frame_errors / num_trials,
            "avg_iterations": total_iters / num_trials,
        }
    return results
