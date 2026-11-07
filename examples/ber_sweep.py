"""BER/FER sweep over the Binary Symmetric Channel.

This script sweeps the QC-LDPC code derived from the matrik-core Matrix
Graph (see src/matrixldpc/codes/base_matrix.py and qc.py) across several
crossover probabilities, for three lift sizes.

Empirical finding (seed=123, num_trials=200, max_iter=30):

  lift_size=2 (H=60x80,   k=36, rate=0.450):
      coded_ber is WORSE than uncoded_ber at almost every p tested.
      The lifted Tanner graph is too small/regular and has low girth
      (many length-4 cycles), so belief propagation barely helps.

  lift_size=4 (H=120x160, k=56, rate=0.350)  <- LDPCCode.from_matrix_graph default
      coded_ber < uncoded_ber at EVERY p from 0.001 to 0.08.
      This is the configuration the test suite and README rely on as the
      "real" demonstration of coding gain.

  lift_size=8 (H=240x320, k=96, rate=0.300):
      Larger and more consistent coding gain than lift_size=4, at the
      cost of a lower rate and more decoding iterations.

Run with:
    python examples/ber_sweep.py
"""

from matrixldpc.codes.code import LDPCCode
from matrixldpc.simulate import simulate_bsc

P_VALUES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]


def main():
    for lift in (2, 4, 8):
        code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=lift)
        print(f"--- lift_size={lift}  H={code.H.shape}  k={code.k}  n={code.n}  rate={code.rate:.3f} ---")

        results = simulate_bsc(code, p_values=P_VALUES, num_trials=200, max_iter=30, seed=123)
        for p, stats in results.items():
            gain = "gain" if stats["coded_ber"] < stats["uncoded_ber"] else "LOSS"
            print(
                f"  p={p:<6} coded_ber={stats['coded_ber']:.5f}  uncoded_ber={stats['uncoded_ber']:.5f}  "
                f"FER={stats['frame_error_rate']:.3f}  avg_iter={stats['avg_iterations']:.1f}  [{gain}]"
            )


if __name__ == "__main__":
    main()
