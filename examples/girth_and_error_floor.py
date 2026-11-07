"""Tanner-graph girth/4-cycle analysis and a coarse error-floor screen.

Two standard LDPC evaluation techniques the rest of this project does not
cover:

1. Girth and 4-cycle count -- a 4-cycle (two check nodes sharing two
   variable-node neighbors) is the shortest possible cycle in a bipartite
   Tanner graph and the classic cause of poor belief-propagation
   convergence ("trapping sets").

2. Error-floor screen -- BER/FER at progressively higher Eb/N0. A real
   floor study needs 10^7-10^9 simulated bits to resolve BER ~1e-9; with
   this project's pure-Python decoder that is not feasible. This script is
   a coarse screen over a much smaller trial budget: it reports whether the
   coded BER keeps falling with each +2 dB step, or stalls/zeroes out
   within that budget.

Run with:
    python examples/girth_and_error_floor.py
"""

from matrixldpc.codes.code import LDPCCode
from matrixldpc.simulate import simulate_awgn

ERROR_FLOOR_SNR_DB = [2.0, 4.0, 6.0, 8.0, 10.0]
NUM_TRIALS = 1000
MAX_ITER = 30


def main():
    for lift in (2, 4, 8):
        code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=lift)
        girth = code.girth()
        n4 = code.num_4_cycles()
        print(f"--- lift_size={lift}  H={code.H.shape}  k={code.k}  n={code.n}  rate={code.rate:.3f} ---")
        print(f"  girth={girth}  4-cycles={n4}  (4-cycles/check={n4 / code.m:.3f}, 4-cycles/var={n4 / code.n:.3f})")

        results = simulate_awgn(code, snr_db_values=ERROR_FLOOR_SNR_DB,
                                  num_trials=NUM_TRIALS, max_iter=MAX_ITER, seed=123)

        prev_ber = None
        min_resolvable_ber = 1.0 / (NUM_TRIALS * code.k)
        for snr_db, stats in results.items():
            ber = stats["coded_ber"]
            if ber == 0.0:
                trend = f"0 errors (below resolution {min_resolvable_ber:.1e})"
            elif prev_ber is None:
                trend = ""
            elif prev_ber == 0.0:
                trend = "ERRORS REAPPEARED after a 0-error point"
            elif ber < prev_ber:
                trend = "still falling"
            else:
                trend = "FLATTENED (possible floor within this trial budget)"

            print(f"  Eb/N0={snr_db:>5.1f} dB  coded_ber={ber:.6f}  FER={stats['frame_error_rate']:.3f}  "
                  f"avg_iter={stats['avg_iterations']:.1f}  {trend}")
            prev_ber = ber
        print()


if __name__ == "__main__":
    main()
