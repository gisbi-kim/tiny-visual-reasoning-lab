"""Regression checks for seven-way decoding; no checkpoint/GPU needed."""
import pytest
import torch
from engine import DecisionEngine
from schema import ACTIONS

@pytest.mark.parametrize('winner',ACTIONS)
def test_all_seven_answers_are_reachable(winner):
    engine=DecisionEngine.__new__(DecisionEngine)
    engine.device=torch.device('cpu');engine.dtype=torch.float32
    engine.temperature=1.;engine.quantized=False;engine.cpu_fallback_enabled=False
    logits=torch.full((len(ACTIONS),),-3.);logits[ACTIONS.index(winner)]=3.
    engine.logits=lambda image,instruction:logits
    result=engine.predict(None,'unchanged instruction')
    assert result['action']==winner
    assert set(result['probabilities'])==set(ACTIONS)
    assert sum(result['probabilities'].values())==pytest.approx(1.)
