"""Cross-validate the matrix-ldpc parity-check matrix H against Sionna's
generic LDPC belief-propagation decoder.

`sionna.phy.fec.ldpc.LDPCBPDecoder` works with *arbitrary* parity-check
matrices (it is not specific to the 5G NR base graphs), so it serves as an
independent reference decoder for the QC-LDPC matrix derived from the
matrik-core Matrix Graph (see src/matrixldpc/codes/base_matrix.py + qc.py).

Two checks are performed for the default code (lift_size=4):

1. Sanity check -- noise-free LLRs must decode back to the exact codeword,
   which confirms `code.H` / `code.G` are mutually consistent and that the
   sign convention used to feed LLRs into Sionna is correct.
   (matrix-ldpc LLR convention: LLR = log P(0)/P(1), positive => bit 0;
   Sionna's LDPCBPDecoder expects logits = log P(1)/P(0), i.e. the negated
   value.)

2. AWGN BER/FER sweep -- the same noisy channel LLRs are decoded by both
   this project's layered min-sum decoder (`LDPCCode.decode`) and Sionna's
   flooding min-sum `LDPCBPDecoder`, so the two columns are a like-for-like
   comparison on identical channel realizations.

Run with:
    python examples/sionna_compare.py
"""

import numpy as np
import torch

from matrixldpc.codes.code import LDPCCode
from sionna.phy.fec.ldpc import LDPCBPDecoder

SNR_DB_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
NUM_TRIALS = 500
MAX_ITER = 30


def sanity_check(code: LDPCCode, decoder: LDPCBPDecoder, rng: np.random.Generator) -> None:
    msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
    cw = code.encode(msg)

    llr_ours = np.where(cw == 0, 10.0, -10.0)  # confident, noise-free LLRs
    llr_sionna = torch.as_tensor(-llr_ours, dtype=torch.float32).unsqueeze(0)

    x_hat = decoder(llr_sionna).numpy().astype(np.uint8)[0]
    assert np.array_equal(x_hat, cw), "Sionna decoder failed noise-free roundtrip"
    assert not np.any((code.H @ x_hat) % 2), "Sionna output does not satisfy H @ c == 0 (mod 2)"


def main():
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)
    print(f"H={code.H.shape}  k={code.k}  n={code.n}  rate={code.rate:.3f}")

    rng = np.random.default_rng(123)

    sionna_decoder = LDPCBPDecoder(
        code.H.astype(np.float32), cn_update="minsum", num_iter=MAX_ITER, hard_out=True
    )
    sanity_check(code, sionna_decoder, rng)
    print("Sanity check passed: Sionna BP decoder reproduces noise-free codewords.\n")

    header = (
        f"{'Eb/N0(dB)':>10} {'own_ber':>10} {'sionna_ber':>11} "
        f"{'uncoded_ber':>12} {'own_FER':>9} {'sionna_FER':>11}"
    )
    print(header)

    for snr_db in SNR_DB_VALUES:
        msgs = rng.integers(0, 2, size=(NUM_TRIALS, code.k), dtype=np.uint8)
        cws = (msgs @ code.G) % 2

        x = 1.0 - 2.0 * cws.astype(np.float64)  # BPSK: 0 -> +1, 1 -> -1
        snr_linear = 10 ** (snr_db / 10)
        sigma = np.sqrt(1.0 / (2 * snr_linear))
        noise = rng.normal(0.0, sigma, size=x.shape)
        y = x + noise
        llr_ours = 2.0 * y / (sigma ** 2)  # log P(0)/P(1)

        # Own layered min-sum decoder, one codeword at a time.
        own_bit_errors = 0
        own_frame_errors = 0
        for t in range(NUM_TRIALS):
            _, decoded_msg, _, _ = code.decode(llr_ours[t], max_iter=MAX_ITER)
            own_bit_errors += int(np.sum(decoded_msg != msgs[t]))
            if not np.array_equal(decoded_msg, msgs[t]):
                own_frame_errors += 1

        # Sionna flooding min-sum decoder, whole batch at once.
        llr_sionna = torch.as_tensor(-llr_ours, dtype=torch.float32)
        x_hat = sionna_decoder(llr_sionna).numpy().astype(np.uint8)
        sionna_decoded_msgs = x_hat[:, code.info_cols]
        sionna_bit_errors = int(np.sum(sionna_decoded_msgs != msgs))
        sionna_frame_errors = int(np.sum(np.any(sionna_decoded_msgs != msgs, axis=1)))

        received_bits = (y < 0).astype(np.uint8)
        uncoded_bit_errors = int(np.sum(received_bits != cws))

        own_ber = own_bit_errors / (NUM_TRIALS * code.k)
        sionna_ber = sionna_bit_errors / (NUM_TRIALS * code.k)
        uncoded_ber = uncoded_bit_errors / (NUM_TRIALS * code.n)
        own_fer = own_frame_errors / NUM_TRIALS
        sionna_fer = sionna_frame_errors / NUM_TRIALS

        print(
            f"{snr_db:>10.1f} {own_ber:>10.5f} {sionna_ber:>11.5f} "
            f"{uncoded_ber:>12.5f} {own_fer:>9.3f} {sionna_fer:>11.3f}"
        )


if __name__ == "__main__":
    main()
