import numpy as np
import pytest

from skill_memory.config import Config
from skill_memory import identify
from controller.dynamic_skill_controller import DynamicSkillController


def make_controller(max_skills=4, max_steps=1500, seed=0):
    cfg = Config(max_skills=max_skills, max_steps=max_steps, random_seed=seed)
    return DynamicSkillController(config=cfg)


def test_first_skill_acquired_from_scratch():
    ctrl = make_controller()
    result = ctrl.acquire("addition", seed=0)
    assert result["success"]
    assert result["record"].origin == "scratch"
    assert result["record"].parent is None
    assert result["record"].solved
    assert ctrl.registry.contains("addition")


def test_second_skill_tries_clone_before_scratch():
    ctrl = make_controller()
    ctrl.acquire("addition", seed=0)
    result = ctrl.acquire("multiplication", seed=1)
    assert result["success"]
    # compatibility.decide should have ranked "addition" as a candidate parent,
    # so the first (and here, successful) attempt should be a clone attempt
    assert result["attempts"][0]["parent"] == "addition"


def test_parent_is_not_mutated_by_child_acquisition():
    ctrl = make_controller()
    ctrl.acquire("addition", seed=0)
    before = {k: v.copy() for k, v in ctrl.registry.get("addition").skill.net.params.items()}

    ctrl.acquire("multiplication", seed=1)

    after = ctrl.registry.get("addition").skill.net.params
    for k in before:
        assert np.allclose(before[k], after[k]), f"parent param '{k}' changed after child acquisition"


def test_duplicate_acquisition_rejected():
    ctrl = make_controller()
    ctrl.acquire("addition", seed=0)
    with pytest.raises(ValueError):
        ctrl.acquire("addition", seed=0)


def test_capacity_exhaustion_reported_not_raised():
    ctrl = make_controller(max_skills=1)
    r1 = ctrl.acquire("addition", seed=0)
    assert r1["success"]
    r2 = ctrl.acquire("multiplication", seed=1)
    assert not r2["success"]
    assert "error" in r2


def test_identify_then_execute_roundtrip():
    ctrl = make_controller()
    ctrl.acquire("addition", seed=0)
    ctrl.acquire("multiplication", seed=1)

    result = ctrl.identify([(4, 4, 16), (3, 5, 15), (7, 2, 14)])
    assert result.status == identify.KNOWN
    assert result.skill_name == "multiplication"

    preds = ctrl.execute(result.skill_name, [(2, 2), (7, 4)])
    assert np.allclose(preds, [4, 28], atol=1.0)


def test_reproducibility_same_seed_same_result():
    ctrl_a = make_controller(seed=0)
    ctrl_b = make_controller(seed=0)
    r_a = ctrl_a.acquire("addition", seed=0)
    r_b = ctrl_b.acquire("addition", seed=0)
    params_a = ctrl_a.registry.get("addition").skill.net.params
    params_b = ctrl_b.registry.get("addition").skill.net.params
    for k in params_a:
        assert np.allclose(params_a[k], params_b[k])
    assert r_a["record"].final_accuracy == r_b["record"].final_accuracy
