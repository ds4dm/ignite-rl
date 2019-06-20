# coding: utf-8

import torch

from irl.exploration.explorer import Explorer
import irl.exploration.metrics as metrics


class Env:
    def __init__(self):
        self.cnt = 0

    def step(self, action):
        self.cnt += 1
        return torch.rand(10), 1.0, self.cnt > 5, {"info_member": 3}

    def reset(self):
        self.cnt = 0
        return torch.rand(10)

    def close(self):
        pass


def test_TransitionMetric():
    agent = Explorer(lambda eng, obs: None)
    metrics.TransitionMetric("reward").attach(agent, "Return")
    agent.run(Env(), 2)
    assert agent.state.metrics["Return"] == 6


def test_Return():
    agent = Explorer(lambda eng, obs: None, metrics={"Return": metrics.Return()})
    agent.run(Env(), 2)
    assert agent.state.metrics["Return"] == 6


def test_InfosMetric():
    agent = Explorer(lambda eng, obs: None)
    metrics.InfoMetric("info_member").attach(agent, "Member")
    agent.run(Env(), 2)
    assert agent.state.metrics["Member"] == 6 * 3


def test_EpisodeLength():
    agent = Explorer(lambda eng, obs: None)
    metrics.EpisodeLength().attach(agent, "Len")
    agent.run(Env(), 2)
    assert agent.state.metrics["Len"] == 6
