import importlib.util
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "lmms_eval" / "tasks" / "rxnbench_vqa"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"rxnbench_{name}", TASK_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = [_load("utils"), _load("utils_zh")]

CASES = [
    (r"\boxed{b}", "b"),
    (r"\boxed{(D)}", "d"),
    (r"\boxed{\text{C}}", "c"),
    (r"\boxed{B}. A is wrong", "b"),
    (r"The product is \boxed{Acetone}", ""),
    (r"\boxed{dimethylformamide}", ""),
    ("the answer is c", "c"),
    ("So the correct option is b.", "b"),
    ("The answer is a mixture of products.", ""),
    ("A nucleophilic attack gives the product, so the answer is C.", "c"),
    ("The C=O stretch confirms it. Answer: D", "d"),
    ("Option A is wrong because x. Option C is right. Answer: C", "c"),
    ("(B)", "b"),
    ("", ""),
]


@pytest.mark.parametrize("module", MODULES, ids=["en", "zh"])
@pytest.mark.parametrize("pred,expected", CASES)
def test_extract_choice(module, pred, expected):
    assert module._extract_choice(pred) == expected


@pytest.mark.parametrize("module", MODULES, ids=["en", "zh"])
def test_process_results_uses_extractor(module):
    doc = {"answer": 2, "question": "q", "choices": ["a", "b", "c", "d"]}
    assert module.rxnbench_process_results(doc, [r"\boxed{c}"])["exact_match"] == 1.0
    assert module.rxnbench_process_results(doc, [r"\boxed{Acetone}"])["exact_match"] == 0.0
