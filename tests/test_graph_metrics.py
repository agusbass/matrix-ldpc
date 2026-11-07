import numpy as np
import pytest

from matrixldpc.codes.code import LDPCCode
from matrixldpc.codes.graph_metrics import compute_girth, count_4_cycles


def test_two_checks_sharing_two_vars_gives_girth_4_and_one_4_cycle():
    H = np.array([
        [1, 1, 0],
        [1, 1, 0],
    ], dtype=np.uint8)

    assert compute_girth(H) == 4
    assert count_4_cycles(H) == 1


def test_forest_has_no_cycles():
    H = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.uint8)

    assert compute_girth(H) is None
    assert count_4_cycles(H) == 0


def test_hexagon_bipartite_graph_has_girth_6_and_no_4_cycles():
    # check_i connects to var_i and var_(i+1 mod 3) -- a single 6-cycle.
    H = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ], dtype=np.uint8)

    assert compute_girth(H) == 6
    assert count_4_cycles(H) == 0


@pytest.mark.parametrize("lift_size", [2, 4, 8])
def test_default_codes_have_girth_4(lift_size):
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=lift_size)
    # Same (i+j)%modulus protograph for every lift -> same absolute number
    # of 4-cycles, but it makes up a shrinking fraction of the graph as
    # lift_size grows.
    assert code.girth() == 4
    assert code.num_4_cycles() == 24
