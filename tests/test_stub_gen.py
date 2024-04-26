import itertools
from unittest import TestCase
import test_E2E_LP
import test_E2E_QP
import test_E2E_SOCP

class test_stub_gen(TestCase):
    def setUp(self) -> None:
        self.all_problems = itertools.chain(
            test_E2E_LP.name_to_prob.items(),
            test_E2E_QP.name_to_prob.items(),
            test_E2E_SOCP.name_to_prob.items(),
        )
        return super().setUp()
    
    def test_stub_valid(self):
        ...