import numpy as np
from pytest import approx

from cvxpygen.hwgen.c_parsing.array_visitor import FloatArrayVisitor, IntArrayVisitor
from cvxpygen.hwgen.c_parsing.grammars import pdaqp_c_grammar


def test_int_array() -> None:
    ast = pdaqp_c_grammar["int_array"].parse("""c_int pdaqp_hp_list[5] = {
(c_int)1,
(c_int)0,
(c_int)2,
(c_int)1,
(c_int)0,
};
""")

    v = IntArrayVisitor()
    v.visit(ast)

    assert "pdaqp_hp_list" in v.arrays

    arr = v.arrays["pdaqp_hp_list"]
    assert arr.name == "pdaqp_hp_list"
    assert arr.data.size == 5
    assert np.all(arr.data == np.array([1, 0, 2, 1, 0]))


def test_float_array() -> None:
    ast = pdaqp_c_grammar["float_array"].parse("""c_float_store pdaqp_halfplanes[2] = {
(c_float_store)0.5132947022251544,
(c_float_store)0.8582124146547812e-3,
};
""")

    v = FloatArrayVisitor()
    v.visit(ast)

    assert "pdaqp_halfplanes" in v.arrays

    arr = v.arrays["pdaqp_halfplanes"]
    assert arr.name == "pdaqp_halfplanes"
    assert arr.data.size == 2
    assert arr.data == approx([0.5132947022251544, 0.8582124146547812e-3])


def test_algorithm_impl() -> None:
    assert pdaqp_c_grammar["algorithm"].parse(
        "void pdaqp_pid_evaluate(c_float* parameter, "
        """c_float* solution){
    // Generic indented code block
}
"""
    )


def test_c_file_parsing() -> None:
    test_cases = [
        "pid",
        "mpc_re",
        "power",
    ]

    for case in test_cases:
        with open(f"test-vectors/{case}/pdaqp.c", "r") as c_file:
            pdaqp_c_grammar.parse(c_file.read())
