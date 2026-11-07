# matrix-ldpc

A QC-LDPC (Quasi-Cyclic Low-Density Parity-Check) encoder, layered min-sum
belief-propagation decoder, and BER/FER simulator whose parity-check matrix,
decoding schedule, and rate structure are derived **literally** from the
[matrik-core](../matrik-core) "Matrix Graph (3D+1D)" architecture.

## What this is (and isn't)

This is an **educational / research** implementation, not a production or
industrial codec.

- The code is real and fully working: encoding, layered min-sum BP decoding,
  BSC/AWGN channel models, BER/FER simulation, and Tanner-graph metrics
  (girth, 4-cycle count) are all implemented and tested (30 passing tests,
  no stubs/TODOs).
- The specific parity-check matrix construction (`(i+j) % modulus` base
  matrices, lifted with circulant shifts taken from the "Vortex Fisik" ring
  `1,2,4,8,7,5`) is **not** an industry-standard code. It does not match the
  5G NR, Wi-Fi, DVB-S2, or SSD LDPC codes, which are the product of decades of
  density-evolution optimization and are patent-encumbered.
- What it *does* demonstrate, with real numbers (see [Empirical results]
  (#empirical-results) below): a working systematic QC-LDPC encoder/decoder
  pair, derived from a non-traditional construction, that produces a genuine,
  measurable, reproducible coding gain over the uncoded channel at low-to-
  moderate error rates.

If you need a production-grade FEC stack, use
[Sionna](https://developer.nvidia.com/sionna) (NVIDIA, GPU-accelerated,
differentiable link-level simulator with standard 5G LDPC/Polar codes) or
[AFF3CT](https://aff3ct.github.io/) (MIT-licensed C++ toolbox for high-
throughput LDPC/Turbo/Polar/RS simulation). See
[Comparison](#comparison-with-real-fec-toolboxes) below.

## Matrix Graph -> LDPC mapping

| Matrix Graph concept (matrik-core)            | LDPC role here                                              | Module |
|------------------------------------------------|--------------------------------------------------------------|--------|
| Grid `(i + j) % modulus`                        | Protograph / base parity-check matrix entries                | `codes/base_matrix.py` |
| Luar layer (mod 10) + Dalam layer (mod 20)      | Two stacked check-matrix blocks -> multi-rate layered code    | `codes/base_matrix.py` (`build_layered_base_matrix`) |
| Vortex Fisik ring `1 -> 2 -> 4 -> 8 -> 7 -> 5`   | Circulant shift sequence for QC lifting (protograph -> H)     | `codes/qc.py` (`expand_qc`) |
| Diag Ring (forward) / Anti Ring (reverse)       | Forward/backward sweep order in the layered BP schedule       | `decoder.py` (`decode_min_sum`, alternating sweep direction per iteration) |
| Lo Shu 3x3 magic square (gateway)               | "Gateway" column set used for puncturing/rate-matching lookup | `codes/code.py` (`LDPCCode.gateway_columns`) |
| Infinity Portal / digital-root convergence      | Iterative decoding to convergence (syndrome check each pass)  | `decoder.py` |

## Architecture

```
src/matrixldpc/
  core/matrix_graph.py   # ported MatrixGraph (3D+1D graph builder)
  codes/base_matrix.py   # (i+j)%modulus protograph, layered Luar+Dalam base matrix
  codes/qc.py            # circulant permutation matrices + QC lifting (Vortex Fisik shifts)
  codes/gf2.py           # GF(2) RREF, systematic generator matrix G = [I | P]
  codes/graph_metrics.py # Tanner graph girth + 4-cycle count
  codes/code.py          # LDPCCode: encode/decode/rate/gateway_columns/girth facade
  decoder.py             # layered min-sum belief propagation
  channel.py             # BSC and AWGN+BPSK channel models, LLR computation
  simulate.py            # BER/FER simulation harness (BSC, AWGN)
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

```python
import numpy as np
from matrixldpc import LDPCCode
from matrixldpc.channel import bsc_transmit, bsc_llr

code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)
print(code.H.shape, code.k, code.n, code.rate)  # (120, 160) 56 160 0.35

rng = np.random.default_rng(0)
msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
codeword = code.encode(msg)

received = bsc_transmit(codeword, p=0.02, rng=rng)
llr = bsc_llr(received, p=0.02)
decoded_cw, decoded_msg, iters, converged = code.decode(llr, max_iter=30)

assert np.array_equal(decoded_msg, msg)
```

## Empirical results

`python examples/ber_sweep.py` runs the BSC simulator across crossover
probabilities `p` for three lift sizes (seed=123, 200 trials/point, max 30
iterations). Reproduced output:

```
--- lift_size=2  H=(60, 80)   k=36  n=80   rate=0.450 ---
  p=0.001  coded_ber=0.00028  uncoded_ber=0.00075  [gain]
  p=0.005  coded_ber=0.00708  uncoded_ber=0.00519  [LOSS]
  p=0.01   coded_ber=0.01514  uncoded_ber=0.01050  [LOSS]
  p=0.02   coded_ber=0.02306  uncoded_ber=0.01900  [LOSS]
  p=0.03   coded_ber=0.03750  uncoded_ber=0.02950  [LOSS]
  p=0.05   coded_ber=0.06889  uncoded_ber=0.05206  [LOSS]
  p=0.08   coded_ber=0.11097  uncoded_ber=0.08088  [LOSS]

--- lift_size=4  H=(120, 160) k=56  n=160  rate=0.350  (default) ---
  p=0.001  coded_ber=0.00045  uncoded_ber=0.00081  [gain]
  p=0.005  coded_ber=0.00313  uncoded_ber=0.00422  [gain]
  p=0.01   coded_ber=0.00982  uncoded_ber=0.01078  [gain]
  p=0.02   coded_ber=0.01812  uncoded_ber=0.02047  [gain]
  p=0.03   coded_ber=0.02339  uncoded_ber=0.02766  [gain]
  p=0.05   coded_ber=0.04500  uncoded_ber=0.04875  [gain]
  p=0.08   coded_ber=0.07536  uncoded_ber=0.08203  [gain]

--- lift_size=8  H=(240, 320) k=96  n=320  rate=0.300 ---
  p=0.001  coded_ber=0.00031  uncoded_ber=0.00078  [gain]
  p=0.005  coded_ber=0.00292  uncoded_ber=0.00536  [gain]
  p=0.01   coded_ber=0.00490  uncoded_ber=0.00944  [gain]
  p=0.02   coded_ber=0.00984  uncoded_ber=0.01941  [gain]
  p=0.03   coded_ber=0.01328  uncoded_ber=0.02958  [gain]
  p=0.05   coded_ber=0.02151  uncoded_ber=0.04991  [gain]
  p=0.08   coded_ber=0.04641  uncoded_ber=0.07992  [gain]
```

Honest interpretation:

- **`lift_size=2` is too small.** Belief propagation performs *worse* than
  doing nothing at almost every tested error rate. This is a real,
  reproducible negative result, not a bug. (See [Tanner graph
  structure](#tanner-graph-structure-and-error-floor-screen) below for the
  actual cause -- it is *not* simply "lift_size=2 has short cycles and the
  others don't"; all three lift sizes have girth 4.)
- **`lift_size=4` (the default) and `lift_size=8`** give a consistent,
  genuine coding gain across the whole tested range (`p` from 0.001 to
  0.08). The gain grows with lift size, at the cost of a lower rate and more
  decoder iterations (avg ~1 iteration near `p=0.001`, up to ~23 near
  `p=0.08` for `lift_size=8`).
- This is the kind of "minimum viable lift size / threshold effect" behavior
  that real LDPC literature also discusses -- it is presented here as part of
  the educational value of the project, not hidden.

## Validation against Sionna

`python examples/sionna_compare.py` (requires `pip install -e ".[sionna]"`,
which pulls in [Sionna](https://developer.nvidia.com/sionna) and PyTorch)
cross-checks the parity-check matrix `H` against Sionna's generic
`LDPCBPDecoder`, which works with arbitrary parity-check matrices and is an
independent reference implementation. The same noisy AWGN LLRs are decoded
by both this project's layered min-sum decoder and Sionna's flooding
min-sum decoder:

```
H=(120, 160)  k=56  n=160  rate=0.350
Sanity check passed: Sionna BP decoder reproduces noise-free codewords.

 Eb/N0(dB)    own_ber  sionna_ber  uncoded_ber   own_FER  sionna_FER
       0.0    0.01368     0.01275      0.07880     0.504       0.486
       1.0    0.00557     0.00529      0.05656     0.270       0.254
       2.0    0.00268     0.00268      0.03729     0.136       0.138
       3.0    0.00086     0.00086      0.02255     0.044       0.046
       4.0    0.00039     0.00039      0.01230     0.022       0.022
       5.0    0.00007     0.00007      0.00635     0.004       0.004
```

Sionna's noise-free roundtrip succeeds (the `H`/`G` pair and the LLR sign
convention are mutually consistent), and its BER/FER track our own decoder
closely across the swept Eb/N0 range -- independent confirmation that `H` is
a genuine, decodable LDPC code and not an artifact of our own decoder
implementation.

## Tanner graph structure and error-floor screen

`python examples/girth_and_error_floor.py` computes the **girth** (shortest
cycle in the Tanner graph) and the number of **4-cycles** (`codes/graph_metrics.py`,
`LDPCCode.girth()` / `LDPCCode.num_4_cycles()`), then runs a coarse
high-Eb/N0 screen for an error floor:

```
--- lift_size=2  H=(60, 80)  k=36  n=80  rate=0.450 ---
  girth=4  4-cycles=24  (4-cycles/check=0.400, 4-cycles/var=0.300)
  Eb/N0=  2.0 dB  coded_ber=0.003972  FER=0.130  avg_iter=1.0
  Eb/N0=  4.0 dB  coded_ber=0.000389  FER=0.014  avg_iter=1.0  still falling
  Eb/N0=  6.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 2.8e-05)
  Eb/N0=  8.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 2.8e-05)
  Eb/N0= 10.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 2.8e-05)

--- lift_size=4  H=(120, 160)  k=56  n=160  rate=0.350 ---
  girth=4  4-cycles=24  (4-cycles/check=0.200, 4-cycles/var=0.150)
  Eb/N0=  2.0 dB  coded_ber=0.002732  FER=0.139  avg_iter=1.1
  Eb/N0=  4.0 dB  coded_ber=0.000339  FER=0.018  avg_iter=1.0  still falling
  Eb/N0=  6.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.8e-05)
  Eb/N0=  8.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.8e-05)
  Eb/N0= 10.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.8e-05)

--- lift_size=8  H=(240, 320)  k=96  n=320  rate=0.300 ---
  girth=4  4-cycles=24  (4-cycles/check=0.100, 4-cycles/var=0.075)
  Eb/N0=  2.0 dB  coded_ber=0.001781  FER=0.156  avg_iter=1.2
  Eb/N0=  4.0 dB  coded_ber=0.000146  FER=0.014  avg_iter=1.0  still falling
  Eb/N0=  6.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.0e-05)
  Eb/N0=  8.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.0e-05)
  Eb/N0= 10.0 dB  coded_ber=0.000000  FER=0.000  avg_iter=1.0  0 errors (below resolution 1.0e-05)
```

Honest interpretation:

- **Girth is 4 for every lift size**, including `lift_size=4` and `8`. The
  `(i+j) % modulus` protograph produces the same 24 four-cycles regardless
  of lift size -- this is a "naive" construction (no PEG / density
  evolution), so 4-cycles are expected, not a defect specific to
  `lift_size=2`.
- What *does* change with lift size is **density**: 24 four-cycles is 0.40
  per check node at `lift_size=2` but only 0.10 at `lift_size=8`. The same
  absolute number of "bad" structures makes up a shrinking fraction of an
  ever-larger graph, which is the more accurate explanation for why larger
  lifts tolerate them better.
- **No error floor within this trial budget** (1000 trials/point, up to
  `Eb/N0=10dB`): coded BER keeps falling and reaches 0 errors (i.e. below
  the ~`1e-5` resolution of 1000 trials) by 6dB for all three lift sizes.
  This is a screen, not a floor measurement -- resolving a real floor at
  `1e-9` would need ~10^7-10^9 simulated bits, infeasible for the
  pure-Python decoder here. No floor was *observed*, but none below `1e-5`
  can be ruled out either.

## Comparison with real FEC toolboxes

| | matrix-ldpc | [Sionna](https://developer.nvidia.com/sionna) (NVIDIA) | [AFF3CT](https://aff3ct.github.io/) |
|---|---|---|---|
| Purpose | Educational: explore a Matrix-Graph-derived QC-LDPC code | GPU-accelerated, differentiable link-level simulation for research/ML | High-throughput FEC simulation (LDPC, Turbo, Polar, RS, ...) |
| Codes | One custom QC-LDPC family (graph-derived) | Standard 5G NR LDPC/Polar + custom, full PHY chains | Polar, Turbo, LDPC, RSC, TPC, BCH, Reed-Solomon, RA, repetition |
| Performance | Pure NumPy, small block lengths, not optimized | PyTorch/GPU (Sionna 2.x) | C++11, SIMD, fixed-point, hundreds of Mb/s |
| License | MIT | Open source (NVIDIA) | MIT |
| Use it for... | Understanding QC-LDPC construction/decoding end to end, and seeing how a non-standard code behaves | Real 5G/6G PHY research, ML-based receivers | Production-grade or large-scale FEC benchmarking |

## Status

| Component | Status |
|---|---|
| Matrix Graph core (ported) | Done |
| Layered base matrix (Luar/Dalam) | Done |
| QC lifting (Vortex Fisik shifts) | Done |
| GF(2) systematic generator | Done |
| Encoder | Done |
| Layered min-sum decoder | Done |
| BSC / AWGN channel models | Done |
| BER/FER simulation | Done |
| Tanner graph metrics (girth, 4-cycles) | Done |
| Validation against Sionna's BP decoder | Done |
| Tests | 30/30 passing |

## License

MIT, see [LICENSE](LICENSE).
