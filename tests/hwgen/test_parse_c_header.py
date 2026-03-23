from cvxpygen.hwgen.c_parsing.grammars import pdaqp_header_grammar
from cvxpygen.hwgen.c_parsing.header_visitor import HeaderVisitor

test_cases = [
    "pid",
    "mpc_re",
    "power",
]


def test_c_header_parsing() -> None:
    for case in test_cases:
        with open(f"test-vectors/{case}/pdaqp.h", "r") as header_file:
            ast = pdaqp_header_grammar.parse(header_file.read())
            assert ast, f"Parsing {case}/pdaqp.h failed"

            v = HeaderVisitor()
            v.visit(ast)

            assert v.problem_size.n_parameter > 0
            assert v.problem_size.n_solution > 0
