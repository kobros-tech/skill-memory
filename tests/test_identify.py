import numpy as np
import pytest

from skill_memory.registry import SkillRegistry
from skill_memory import identify


class FakeNet:
    """
    net.predict(X) -> np.ndarray, X is (N, 2) of *scaled* (x/10, y/10) pairs,
    matching skill-cloning's real TinyMLP.predict interface (see identify.py's
    INPUT_SCALE). fn is written in terms of raw operands, so we undo the
    scaling here to keep the fake's math intuitive.
    """
    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        X = np.asarray(X, dtype=float) * identify.INPUT_SCALE
        return np.array([self.fn(x, y) for x, y in X])


class FakeSkill:
    def __init__(self, net):
        self.net = net


def make_registry():
    reg = SkillRegistry(max_skills=8)
    reg.register("addition", FakeSkill(FakeNet(lambda x, y: x + y)))
    reg.register("multiplication", FakeSkill(FakeNet(lambda x, y: x * y)))
    reg.register("subtraction", FakeSkill(FakeNet(lambda x, y: x - y)))
    return reg


def test_unambiguous_batch_identifies_correct_skill():
    reg = make_registry()
    # 4*4=16, 3*5=15, 7*2=14: only multiplication explains all three
    examples = [(4, 4, 16), (3, 5, 15), (7, 2, 14)]
    result = identify.identify(examples, reg, tol=0.5)
    assert result.status == identify.KNOWN
    assert result.skill_name == "multiplication"


def test_unknown_when_no_skill_matches():
    reg = make_registry()
    examples = [(4, 4, 999), (3, 5, 999), (7, 2, 999)]
    result = identify.identify(examples, reg, tol=0.5)
    assert result.status == identify.UNKNOWN
    assert result.skill_name is None


def test_single_example_can_be_ambiguous():
    reg = make_registry()
    # 2+2=4 and 2*2=4 both explain this single example
    examples = [(2, 2, 4)]
    result = identify.identify(examples, reg, tol=0.5)
    assert result.status == identify.AMBIGUOUS


def test_batch_disambiguates_previously_ambiguous_case():
    reg = make_registry()
    # (2,2,4) alone is ambiguous, but a second example rules out multiplication
    examples = [(2, 2, 4), (3, 5, 8)]  # 3+5=8 but 3*5=15
    result = identify.identify(examples, reg, tol=0.5)
    assert result.status == identify.KNOWN
    assert result.skill_name == "addition"


def test_empty_examples_raises():
    reg = make_registry()
    with pytest.raises(ValueError):
        identify.identify([], reg)


def test_empty_registry_is_unknown():
    reg = SkillRegistry(max_skills=4)
    result = identify.identify([(1, 2, 3)], reg)
    assert result.status == identify.UNKNOWN


def test_execute_uses_only_x_y():
    reg = make_registry()
    preds = identify.execute(reg, "multiplication", [(2, 2), (7, 4)])
    assert np.allclose(preds, [4, 28])
