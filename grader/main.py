from . import tests
from typing import Set

try:
    from .solution import BayesNet
except ImportError:
    pass

def run_tests(categories: Set[str] = None):
    tests.run_tests(BayesNet, categories)
