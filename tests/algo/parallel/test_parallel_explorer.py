# coding: utf-8

import pytest
import torch

from irl.algo.parallel.explorer import ParallelExplorer
from irl.utils import RWLock


@pytest.mark.timeout(10)
def test_ParallelExplorer(env_factory, device, model):
    model.to(device=device)

    def select_action(engine, obs):
        return model(obs).sample()

    lock = RWLock()
    agents = [
        ParallelExplorer(
            env=env_factory(),
            select_action=select_action,
            model_lock=lock,
            device=device,
            dtype=torch.float
        ) for _ in range(4)
    ]

    for a in agents:
        a.run(10, 10)

    for a in agents:
        a.thread.join()
        assert a.state.epoch == 10
