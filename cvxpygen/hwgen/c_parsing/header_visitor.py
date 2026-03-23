from parsimonious.nodes import NodeVisitor

from cvxpygen.hwgen.models import ProblemSize


class HeaderVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.problem_size = ProblemSize(0, 0)

    def visit_n_parameter(self, node, visited_children):
        self.problem_size.n_parameter = visited_children[2]

    def visit_n_solution(self, node, visited_children):
        self.problem_size.n_solution = visited_children[2]

    def visit_digits(self, node, _) -> int:
        return int(node.text)

    def generic_visit(self, node, visited_children):
        return visited_children or node
