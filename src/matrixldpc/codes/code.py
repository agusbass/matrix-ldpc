"""LDPCCode -- ties together protograph construction, QC expansion, GF(2)
generator-matrix derivation, and the layered min-sum decoder behind a single
encode/decode interface.
"""

import numpy as np

from matrixldpc.codes.base_matrix import build_layered_base_matrix
from matrixldpc.codes.qc import expand_qc
from matrixldpc.codes.gf2 import build_generator, verify_generator
from matrixldpc.codes.graph_metrics import compute_girth, count_4_cycles
from matrixldpc.decoder import decode_min_sum

# Lo Shu 3x3 magic square (gateway / puncturing pattern, mapping.html
# domain "Error Correcting Codes": Gateway -> puncturing/rate-matching).
LO_SHU = ((8, 1, 6), (3, 5, 7), (4, 9, 2))


class LDPCCode:
    """A QC-LDPC code whose parity-check matrix is derived from the
    matrik-core Matrix Graph (codes.base_matrix + codes.qc)."""

    def __init__(self, H: np.ndarray):
        self.H = (H % 2).astype(np.uint8)
        self.m, self.n = self.H.shape
        self.G, self.info_cols, self.rank = build_generator(self.H)
        self.k = len(self.info_cols)
        if not verify_generator(self.H, self.G):
            raise ValueError("Derived generator matrix does not satisfy H @ G.T == 0 (mod 2)")

    @classmethod
    def from_matrix_graph(cls, num_vars_base: int = 40, lift_size: int = 4) -> "LDPCCode":
        """Build the default code: stack the Luar (mod 10) and Dalam
        (mod 20) base matrices on `num_vars_base` columns ("3 Layer =
        multi-rate code"), then expand via QC lifting with `lift_size`
        ("Vortex Fisik ring = expansion sequence").
        """
        base = build_layered_base_matrix(num_vars_base)
        H = expand_qc(base, lift_size)
        return cls(H)

    @property
    def rate(self) -> float:
        """Code rate k/n (information bits per coded bit)."""
        return self.k / self.n

    def girth(self, max_depth: int = 10) -> int | None:
        """Shortest cycle length of the Tanner graph of `H` (see
        codes.graph_metrics.compute_girth)."""
        return compute_girth(self.H, max_depth)

    def num_4_cycles(self) -> int:
        """Number of length-4 cycles in the Tanner graph of `H` (see
        codes.graph_metrics.count_4_cycles)."""
        return count_4_cycles(self.H)

    def gateway_columns(self) -> list[int]:
        """Codeword column indices marked by the Lo Shu 3x3 magic square
        (values 1-9 -> 0-indexed columns mod n) -- the "gateway" puncturing
        pattern from mapping.html.
        """
        values = sorted({v for row in LO_SHU for v in row})
        return sorted({(v - 1) % self.n for v in values})

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode a length-k message (0/1 array) into a length-n codeword."""
        message = np.asarray(message, dtype=np.uint8)
        if message.shape != (self.k,):
            raise ValueError(f"message must have shape ({self.k},), got {message.shape}")
        return (message @ self.G) % 2

    def extract_message(self, codeword: np.ndarray) -> np.ndarray:
        """Recover the length-k message from a length-n codeword (systematic bits)."""
        return codeword[self.info_cols]

    def decode(self, llr: np.ndarray, max_iter: int = 20):
        """Decode channel LLRs into (codeword, message, iterations, converged)."""
        codeword, iterations, converged = decode_min_sum(self.H, llr, max_iter)
        message = self.extract_message(codeword)
        return codeword, message, iterations, converged
