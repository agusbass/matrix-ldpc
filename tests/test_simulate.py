from matrixldpc.codes.code import LDPCCode
from matrixldpc.simulate import simulate_bsc, simulate_awgn


def test_simulate_bsc_provides_coding_gain():
    # lift_size=4 is the default product configuration (see
    # LDPCCode.from_matrix_graph). lift_size=2 is too small: low girth in
    # the lifted Tanner graph makes BP perform *worse* than uncoded at
    # these crossover probabilities (verified empirically).
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)

    results = simulate_bsc(code, p_values=[0.02], num_trials=200, max_iter=30, seed=123)
    stats = results[0.02]

    assert set(stats) == {"coded_ber", "uncoded_ber", "frame_error_rate", "avg_iterations"}
    assert 0.0 <= stats["coded_ber"] <= 1.0
    assert 0.0 <= stats["uncoded_ber"] <= 1.0
    assert 0.0 <= stats["frame_error_rate"] <= 1.0
    # The decoder should reduce the bit-error rate relative to the raw
    # (uncoded) channel error rate at this crossover probability.
    assert stats["coded_ber"] < stats["uncoded_ber"]


def test_simulate_awgn_runs_and_improves_with_snr():
    code = LDPCCode.from_matrix_graph(num_vars_base=40, lift_size=4)

    results = simulate_awgn(code, snr_db_values=[1.0, 4.0], num_trials=60, max_iter=30, seed=99)

    assert set(results) == {1.0, 4.0}
    for stats in results.values():
        assert 0.0 <= stats["coded_ber"] <= 1.0
        assert 0.0 <= stats["frame_error_rate"] <= 1.0

    # Higher Eb/N0 should not make the coded BER worse on average.
    assert results[4.0]["coded_ber"] <= results[1.0]["coded_ber"]
