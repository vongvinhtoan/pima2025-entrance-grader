import random
import itertools as it

def random_cpts(instances: dict[dict], parents: dict[list]):
    cpts = {}
    for var, insts in instances.items():
        cpt = {}
        for par_inst in it.product(*[instances[p] for p in parents[var]]):
            probs = [random.random() for _ in range(len(insts))]
            total = sum(probs)
            probs = [a / total for a in probs]
            cpt.update(dict(zip(it.product(insts, [par_inst]), probs)))
        cpts[var] = cpt
    return cpts

def make_binary_instances(n):
    return {
        chr(ord('A') + i): {c+chr(ord('a')+i) for c in ['+', '-']}
        for i in range(n)
    }

def random_instance(instances, numtests):
    inst = list(it.product(*instances.values()))
    for _ in range(numtests):
        choice = random.choice(inst)
        yield dict(zip(instances.keys(), choice))

def random_event(instances, numtests):
    for _ in range(numtests):
        observed = random.sample(list(instances.keys()), random.randint(1, len(instances)))
        choice = random.choice(list(it.product(*[instances[v] for v in observed])))
        yield dict(zip(observed, choice))

def random_conditional(instances, numtests):
    for i in range(numtests):
        observed = random.sample(list(instances.keys()), random.randint(1, len(instances)))
        split = random.randint(1, len(observed))
        condition = observed[split:]
        observed = observed[:split]
        choice_observed = random.choice(list(it.product(*[instances[v] for v in observed])))
        choice_condition = random.choice(list(it.product(*[instances[v] for v in condition])))
        yield dict(zip(observed, choice_observed)), dict(zip(condition, choice_condition))