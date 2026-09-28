# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import torch

from skill_memory.cl.training import _select_historical_samples


def test_full_replay_keeps_all_retained_history():
    inputs = torch.arange(10).reshape(5, 2)
    targets = torch.tensor([4, 4, 4, 4, 4])

    selected_inputs, selected_targets = _select_historical_samples(
        inputs,
        targets,
        limit=None,
        seed=0,
        class_id=4,
    )

    assert torch.equal(selected_inputs, inputs)
    assert torch.equal(selected_targets, targets)


def test_small_replay_caps_retained_history():
    inputs = torch.arange(20).reshape(10, 2)
    targets = torch.full((10,), 7)

    selected_inputs, selected_targets = _select_historical_samples(
        inputs,
        targets,
        limit=3,
        seed=0,
        class_id=7,
    )

    assert len(selected_inputs) == 3
    assert len(selected_targets) == 3
    assert torch.equal(selected_targets, torch.full((3,), 7))
    generator = torch.Generator().manual_seed(0 + 7)
    expected_indices = torch.randperm(10, generator=generator)[:3]
    assert torch.equal(selected_inputs, inputs[expected_indices])


def test_small_replay_is_different_from_full_replay_when_history_is_large():
    inputs = torch.arange(20).reshape(10, 2)
    targets = torch.arange(10)

    full_inputs, _ = _select_historical_samples(
        inputs,
        targets,
        limit=None,
        seed=0,
        class_id=1,
    )
    small_inputs, _ = _select_historical_samples(
        inputs,
        targets,
        limit=3,
        seed=0,
        class_id=1,
    )

    assert len(full_inputs) == 10
    assert len(small_inputs) == 3


def test_domain_training_consumes_full_vs_small_historical_replay():
    from types import SimpleNamespace

    from skill_memory.cl.training import train_skill_on_domain
    from skill_memory.evaluation.memory import EvaluationMemory

    class Experience:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [1.1], [1.2], [1.3]]),
            torch.tensor([1, 1, 1, 1]),
        )

    class Strategy:
        def __init__(self):
            self.model = torch.nn.Linear(1, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.clock = SimpleNamespace(train_iterations=0)

    retained = [
        EvaluationMemory(
            inputs=torch.arange(10, dtype=torch.float32).reshape(10, 1),
            targets=torch.zeros(10, dtype=torch.long),
            class_id=0,
        )
    ]

    full_counts = {}
    full_strategy = Strategy()
    train_skill_on_domain(
        full_strategy,
        Experience(),
        {0},
        {0, 1},
        epochs=1,
        batch_size=8,
        validation_fraction=0.0,
        retained_memory=retained,
        historical_samples_per_class=None,
        training_counts=full_counts,
    )

    small_counts = {}
    small_strategy = Strategy()
    train_skill_on_domain(
        small_strategy,
        Experience(),
        {0},
        {0, 1},
        epochs=1,
        batch_size=8,
        validation_fraction=0.0,
        retained_memory=retained,
        samples_per_class=4,
        historical_samples_per_class=3,
        training_counts=small_counts,
    )

    assert full_counts == {0: 10, 1: 4}
    assert small_counts == {0: 3, 1: 4}


def test_full_replay_is_not_capped_by_current_class_sample_budget():
    from types import SimpleNamespace

    from skill_memory.cl.training import train_skill_on_domain
    from skill_memory.evaluation.memory import EvaluationMemory

    class Experience:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [1.1], [1.2], [1.3], [1.4]]),
            torch.tensor([1, 1, 1, 1, 1]),
        )

    class Strategy:
        def __init__(self):
            self.model = torch.nn.Linear(1, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.clock = SimpleNamespace(train_iterations=0)

    retained = [
        EvaluationMemory(
            inputs=torch.arange(10, dtype=torch.float32).reshape(10, 1),
            targets=torch.zeros(10, dtype=torch.long),
            class_id=0,
        )
    ]
    counts = {}
    train_skill_on_domain(
        Strategy(),
        Experience(),
        {0},
        {0, 1},
        epochs=1,
        batch_size=8,
        validation_fraction=0.0,
        retained_memory=retained,
        samples_per_class=2,
        historical_samples_per_class=None,
        training_counts=counts,
    )

    assert counts == {0: 10, 1: 2}


def test_class_training_replay_budget_is_independent_of_current_class_budget(
    monkeypatch,
):
    from types import SimpleNamespace

    import skill_memory.cl.training as training
    from skill_memory.evaluation.memory import EvaluationMemory

    class Experience:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [1.1], [1.2], [1.3]]),
            torch.tensor([1, 1, 1, 1]),
        )

    class Strategy:
        def __init__(self):
            self.model = torch.nn.Linear(1, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.clock = SimpleNamespace(train_iterations=0)

    retained = [
        EvaluationMemory(
            inputs=torch.arange(10, dtype=torch.float32).reshape(10, 1),
            targets=torch.zeros(10, dtype=torch.long),
            class_id=0,
        )
    ]

    original_loader = training.DataLoader
    captured = []

    def spy_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        captured.append((len(loader.dataset), len(loader.sampler)))
        return loader

    monkeypatch.setattr(training, "DataLoader", spy_loader)

    training.train_on_class(
        Strategy(),
        Experience(),
        target_class=1,
        epochs=1,
        batch_size=8,
        mode="binary_one_vs_rest",
        validation_fraction=0.0,
        retained_memory=retained,
        samples_per_class=4,
        historical_samples_per_class=3,
    )

    # Four current samples + three historical negatives. The sampler must
    # describe that exact assembled dataset; it must not use the current-class
    # budget (4) for historical replay.
    assert captured[-1] == (7, 7)


def test_class_training_full_replay_uses_all_retained_history(monkeypatch):
    from types import SimpleNamespace

    import skill_memory.cl.training as training
    from skill_memory.evaluation.memory import EvaluationMemory

    class Experience:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [1.1], [1.2], [1.3]]),
            torch.tensor([1, 1, 1, 1]),
        )

    class Strategy:
        def __init__(self):
            self.model = torch.nn.Linear(1, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.clock = SimpleNamespace(train_iterations=0)

    retained = [
        EvaluationMemory(
            inputs=torch.arange(10, dtype=torch.float32).reshape(10, 1),
            targets=torch.zeros(10, dtype=torch.long),
            class_id=0,
        )
    ]

    original_loader = training.DataLoader
    captured = []

    def spy_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        captured.append((len(loader.dataset), len(loader.sampler)))
        return loader

    monkeypatch.setattr(training, "DataLoader", spy_loader)

    training.train_on_class(
        Strategy(),
        Experience(),
        target_class=1,
        epochs=1,
        batch_size=8,
        mode="binary_one_vs_rest",
        validation_fraction=0.0,
        retained_memory=retained,
        samples_per_class=2,
        historical_samples_per_class=None,
    )

    assert captured[-1] == (12, 12)


def test_binary_one_vs_rest_trains_against_current_negative_classes():
    from types import SimpleNamespace

    from skill_memory.cl.training import train_on_class

    class Experience:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [1.1], [-1.0], [-1.1], [0.5], [0.6]]),
            torch.tensor([0, 0, 1, 1, 2, 2]),
        )

    class Strategy:
        def __init__(self):
            self.model = torch.nn.Linear(1, 3, bias=False)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.clock = SimpleNamespace(train_iterations=0)

    strategy = Strategy()
    before = strategy.model.weight.detach().clone()

    train_on_class(
        strategy,
        Experience(),
        target_class=0,
        epochs=1,
        batch_size=16,
        mode="binary_one_vs_rest",
        validation_fraction=0.0,
    )

    after = strategy.model.weight.detach()
    assert not torch.equal(after[0], before[0])
    assert not torch.equal(after[1], before[1])
    assert not torch.equal(after[2], before[2])
