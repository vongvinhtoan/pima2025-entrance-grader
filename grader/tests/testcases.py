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

@testcase(category="binary", score=0.0)
def independent(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 5))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        numtests = 5
        res[i] = {
            "instance": [bn.instance_prob(instance) for instance in random_instance(instances, numtests)],
            "event": [bn.event_prob(event) for event in random_event(instances, numtests)],
            # "condition": [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, numtests)],
            "condition": [bn.event_prob(event) for event, _ in random_conditional(instances, numtests)],
        }
    return res