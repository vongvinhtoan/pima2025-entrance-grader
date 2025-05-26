from .test_decorators import *
from .test_helpers import *

@testcase(category='example', score=0.0)
def example(BayesNet: type):
    instances = {
        'A': {'+a', '-a'},
        'B': {'+b', '-b'},
        'C': {'+c', '-c'},
    }
    parents = {
        'A': [],
        'B': ['A'],
        'C': ['A'],
    }
    cpts = {
        'A': {
            ('+a', ()): 0.25,
            ('-a', ()): 0.75,
        },
        'B': {
            ('+b', ('+a',)): 0.5,
            ('-b', ('+a',)): 0.5,
            ('+b', ('-a',)): 0.25,
            ('-b', ('-a',)): 0.75,
        },
        'C': {
            ('+c', ('+a',)): 0.4,
            ('-c', ('+a',)): 0.6,
            ('+c', ('-a',)): 0.2,
            ('-c', ('-a',)): 0.8,
        },
    }

    bn = BayesNet(instances, cpts, parents)

    return {
        "instance": bn.instance_prob({'A': '+a', 'B': '-b', 'C': '+c'}),
        "event": bn.event_prob({'C': '+c'}),
        "condition": bn.conditional_prob({'B': '-b'}, {'A': '+a'}),
    }