from collections import defaultdict
from typing import Callable, Set, Tuple
import hashlib
import json
from pathlib import Path

THIS_FILE_PATH = Path(__file__).resolve()
answers_dir = THIS_FILE_PATH.parent.parent / "answers"
outputs_dir = THIS_FILE_PATH.parent.parent / "outputs"

# Registry: category -> list of (test_func, score)
all_tests: dict[str, list[Tuple[Callable, float]]] = defaultdict(list)

def round_floats(obj, digits=4):
    if isinstance(obj, float):
        return round(obj, digits)
    elif isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, digits) for i in obj]
    elif isinstance(obj, set):
        return [round_floats(i, digits) for i in sorted(obj)]
    else:
        return obj

def hash_result(data: dict):
    # return json.dumps(round_floats(data), sort_keys=True)
    data = json.dumps(round_floats(data), sort_keys=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def testcase(category: str = 'general', score: float = 0.0):
    def decorator(func: Callable):
        def wrapper(BayesNet: type):
            answers_dir.mkdir(parents=True, exist_ok=True)
            outputs_dir.mkdir(parents=True, exist_ok=True)
            try:
                solution_path = answers_dir / f"{func.__name__}.txt"
                try:
                    from ..solution import BayesNet as SolutionBayesNet
                    sol = hash_result(func(SolutionBayesNet))
                    solution_path.write_text(sol)
                except ImportError:
                    pass
                
                out = func(BayesNet)
                output_path = outputs_dir / f"{func.__name__}.txt"
                output_path.write_text(json.dumps(round_floats(out, 6), indent=2, sort_keys=True))
                out = hash_result(out)
                assert out == solution_path.read_text(), "Wrong output"
                return True, None
            except Exception as e:
                return False, e
        wrapper.__name__ = func.__name__
        all_tests[category].append((wrapper, score))
        return wrapper
    return decorator

def run_tests(BayesNet, categories: Set[str] = None):
    total_tests, passed_tests = 0, 0
    total_score, earned_score = 0.0, 0.0

    print("\n============================")
    print("🧪 Test Runner Starting...")
    print("============================")

    for category, tests in all_tests.items():
        if categories is None or category in categories:
            print(f"\n📂 Category: {category}")
            print("-" * 30)

            cat_total, cat_passed = 0, 0
            cat_score_total, cat_score_earned = 0.0, 0.0

            for i, (test_func, score) in enumerate(tests, 1):
                result, reason = test_func(BayesNet)
                passed = "✅" if result else "❌"

                print(f"  [{passed}] Test `{test_func.__name__}` | Score: {score:.2f}")
                if not result and reason is not None:
                    print(f"      Reason: {reason}")

                cat_total += 1
                cat_score_total += score
                if result:
                    cat_passed += 1
                    cat_score_earned += score

            total_tests += cat_total
            passed_tests += cat_passed
            total_score += cat_score_total
            earned_score += cat_score_earned

            print("-" * 30)
            print(f"  ✅ Passed: {cat_passed}/{cat_total}")
            print(f"  🏅 Score : {cat_score_earned:.2f} / {cat_score_total:.2f}")

    print("\n============================")
    print("📝 OVERALL SUMMARY")
    print("============================")
    print(f"🧪 Total Tests Run   : {total_tests}")
    print(f"✅ Tests Passed      : {passed_tests}")
    print(f"🏅 Total Score Earned: {earned_score:.2f} / {total_score:.2f}")
    print("============================\n")

