import numpy as np
from skill_memory.skill import TinyMLP
from skill_memory.tasks import sample_task
from skill_memory.compatibility import decide

def test_clone_is_deep_and_optimizer_is_fresh():
    net = TinyMLP(seed=1)
    clone = net.clone()
    clone.params["W1"][0, 0] += 1
    assert not np.array_equal(net.params["W1"], clone.params["W1"])
    assert all(state["t"] == 0 for state in clone._adam_state.values())

def test_signed_division_has_no_zero_divisors():
    X, y = sample_task("division", 500, seed=3, domain="signed")
    assert np.all(X[:, 1] != 0)
    assert np.isfinite(y).all()

def test_empty_decision_is_scratch():
    result = decide({}, "addition", base_seed=0)
    assert result["action"] == "scratch"
