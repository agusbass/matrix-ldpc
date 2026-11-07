from matrixldpc.core.matrix_graph import MatrixGraph


def test_build_layer_counts():
    g = MatrixGraph(11).build()
    counts = g.get_layer_counts()
    assert counts['luar'] == 11 ** 3 + 10 + 1  # grid + value + core nodes
    assert counts['dalam'] == 11 ** 3 + 20 + 1
    assert counts['total'] == counts['luar'] + counts['ms'] + counts['dalam']


def test_digital_root():
    g = MatrixGraph(11)
    assert g.digital_root(0) == 0
    assert g.digital_root(9) == 9
    assert g.digital_root(10) == 1
    assert g.digital_root(19) == 1
    assert g.digital_root(20) == 2


def test_cross_layer_links_exist():
    g = MatrixGraph(11).build()
    luar_core = g.get_node('luar_core_0')
    ms_core = g.get_node('ms_core_5')
    dalam_core = g.get_node('dalam_core_0')
    assert ms_core in luar_core.edges
    assert ms_core in dalam_core.edges
