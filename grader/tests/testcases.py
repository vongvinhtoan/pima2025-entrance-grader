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

@testcase(category='example', score=0.0)
def pima_island(BayesNet: type):
    instances = {
        'D': {'+d', '-d'},
        'M': {'+m', '-m'},
        'A': {'+a', '-a'},
        'B': {'+b', '-b'},
        'C': {'+c', '-c'},
        'R': {'+r', '-r'},
        'K': {'+k', '-k'},
        'G': {'+g', '-g'},
    }
    parents = {
        'D': [],
        'M': [],
        'A': ['D', 'M'],
        'B': ['M'],
        'C': ['M'],
        'R': ['A', 'B'],
        'K': ['C'],
        'G': ['A', 'R', 'K'],
    }
    cpts = {
        'D': {
            ('-d', ()): 0.8,
            ('+d', ()): 0.2,
        },
        'M': {
            ('-m', ()): 0.1,
            ('+m', ()): 0.9,
        },
        'A': {
            ('-a', ('-d', '-m')): 1.0,
            ('+a', ('-d', '-m')): 0.0,
            ('-a', ('+d', '-m')): 1.0,
            ('+a', ('+d', '-m')): 0.0,
            ('-a', ('-d', '+m')): 0.2,
            ('+a', ('-d', '+m')): 0.8,
            ('-a', ('+d', '+m')): 0.6,
            ('+a', ('+d', '+m')): 0.4,
        },
        'B': {
            ('-b', ('-m',)): 1.0,
            ('+b', ('-m',)): 0.0,
            ('-b', ('+m',)): 0.2,
            ('+b', ('+m',)): 0.8,
        },
        'C': {
            ('-c', ('-m',)): 1.0,
            ('+c', ('-m',)): 0.0,
            ('-c', ('+m',)): 0.2,
            ('+c', ('+m',)): 0.8,
        },
        'R': {
            ('-r', ('-a', '-b')): 1.0,
            ('+r', ('-a', '-b')): 0.0,
            ('-r', ('+a', '-b')): 0.7,
            ('+r', ('+a', '-b')): 0.3,
            ('-r', ('-a', '+b')): 0.7,
            ('+r', ('-a', '+b')): 0.3,
            ('-r', ('+a', '+b')): 0.3,
            ('+r', ('+a', '+b')): 0.7,
        },
        'K': {
            ('-k', ('-c',)): 1.0,
            ('+k', ('-c',)): 0.0,
            ('-k', ('+c',)): 0.1,
            ('+k', ('+c',)): 0.9,
        },
        'G': {
            ('-g', ('-a', '-r', '-k')): 1.0,
            ('+g', ('-a', '-r', '-k')): 0.0,
            ('-g', ('+a', '-r', '-k')): 1.0,
            ('+g', ('+a', '-r', '-k')): 0.0,
            ('-g', ('-a', '+r', '-k')): 1.0,
            ('+g', ('-a', '+r', '-k')): 0.0,
            ('-g', ('+a', '+r', '-k')): 1.0,
            ('+g', ('+a', '+r', '-k')): 0.0,
            ('-g', ('-a', '-r', '+k')): 1.0,
            ('+g', ('-a', '-r', '+k')): 0.0,
            ('-g', ('+a', '-r', '+k')): 1.0,
            ('+g', ('+a', '-r', '+k')): 0.0,
            ('-g', ('-a', '+r', '+k')): 0.1,
            ('+g', ('-a', '+r', '+k')): 0.9,
            ('-g', ('+a', '+r', '+k')): 0.01,
            ('+g', ('+a', '+r', '+k')): 0.99,
        },
    }

    bn = BayesNet(instances, cpts, parents)

    def conditional_independent(varA, varB, condition):
        instA = [{varA: v} for v in instances[varA]]
        instB = [{varB: v} for v in instances[varB]]
        instCond = [dict(zip(condition, vals)) for vals in it.product(*[instances[c] for c in condition])]
        for assA, assB, assCond in it.product(instA, instB, instCond):
            A = bn.conditional_prob(assA.copy(), assCond)
            B = bn.conditional_prob(assB.copy(), assCond)
            AB = bn.conditional_prob({**assA.copy(), **assB.copy()}, assCond)
            if A is None or B is None or AB is None: continue
            if abs(A * B - AB) >= 1e-9: return False
        return True

    return {
        "2.a.": bn.conditional_prob({'D': '+d'}, {'A': '+a', 'M': '+m'}),
        "2.b.": bn.conditional_prob({'R': '+r'}, {'D': '+d', 'M': '+m'}),
        "2.c.": bn.conditional_prob({'G': '+g'}, {'M': '+m'}),
        "2.d.": bn.conditional_prob({'C': '+c'}, {'G': '+g'}),
        "2.e.": bn.conditional_prob({'G': '+g'}, {'M': '-m'}),
        "2.f.": bn.conditional_prob({'A': '+a'}, {'G': '+g', 'M': '-m'}),
        "3.a": conditional_independent('D', 'M', []),
        "3.b": conditional_independent('M', 'K', []),
        "3.c": conditional_independent('M', 'K', ['C']),
        "3.d": conditional_independent('D', 'M', ['A']),
        "3.e": conditional_independent('D', 'M', ['B']),
        "3.f": conditional_independent('D', 'M', ['G']),
        "3.g": conditional_independent('D', 'M', ['K']),
        "3.h": conditional_independent('A', 'B', ['M']),
        "3.i": conditional_independent('R', 'K', ['C']),
        "3.j": conditional_independent('R', 'K', ['C', 'G']),
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

@testcase(category="binary", score=0.0)
def independent_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category="binary", score=0.0)
def independent_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res

@testcase(category="binary", score=0.0)
def independent_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def markov(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 5))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        numtests = 5
        res[i] = {
            "instance": [bn.instance_prob(instance) for instance in random_instance(instances, numtests)],
            "event": [bn.event_prob(event) for event in random_event(instances, numtests)],
            "condition": [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, numtests)],
        }
    return res

@testcase(category='binary', score=0.0)
def markov_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def markov_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def markov_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def dag(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 5))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, ord(var) - ord('A')))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        numtests = 5
        res[i] = {
            "instance": [bn.instance_prob(instance) for instance in random_instance(instances, numtests)],
            "event": [bn.event_prob(event) for event in random_event(instances, numtests)],
            "condition": [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, numtests)],
        }
    return res

@testcase(category='binary', score=0.0)
def dag_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def dag_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='binary', score=0.0)
def dag_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_binary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res

@testcase(category="k-ary", score=0.0)
def k_independent(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 5))
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

@testcase(category="k-ary", score=0.0)
def k_independent_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category="k-ary", score=0.0)
def k_independent_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res

@testcase(category="k-ary", score=0.0)
def k_independent_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [] for var in instances.keys() }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)
        
        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_markov(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 5))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        numtests = 5
        res[i] = {
            "instance": [bn.instance_prob(instance) for instance in random_instance(instances, numtests)],
            "event": [bn.event_prob(event) for event in random_event(instances, numtests)],
            "condition": [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, numtests)],
        }
    return res

@testcase(category='k-ary', score=0.0)
def k_markov_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_markov_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_markov_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { var: [chr(ord(var) - 1)] for var in instances.keys() }
        parents['A'] = []
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_dag(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 5))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, ord(var) - ord('A')))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        numtests = 5
        res[i] = {
            "instance": [bn.instance_prob(instance) for instance in random_instance(instances, numtests)],
            "event": [bn.event_prob(event) for event in random_event(instances, numtests)],
            "condition": [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, numtests)],
        }
    return res

@testcase(category='k-ary', score=0.0)
def k_dag_instance(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.instance_prob(instance) for instance in random_instance(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_dag_event(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.event_prob(event) for event in random_event(instances, 1)]
    return res

@testcase(category='k-ary', score=0.0)
def k_dag_condition(BayesNet):
    random.seed(42)
    n_runs = 10
    res = {}
    for i in range(n_runs):
        instances = make_k_ary_instances(random.randint(3, 10))
        parents = { 
            var: random.sample([chr(ord('A') + i) for i in range(ord(var) - ord('A'))], k = random.randint(0, random.randint(0, ord(var) - ord('A'))))
            for var in instances.keys()
        }
        cpts = random_cpts(instances, parents)
        bn = BayesNet(instances=instances, parents=parents, cpts=cpts)

        res[i] = [bn.conditional_prob(event, condition) for event, condition in random_conditional(instances, 1)]
    return res