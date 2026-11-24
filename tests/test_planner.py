import pytest

from planner.agents import Agent, Prediction, build_messages, offline_response
from planner.data import balanced_indices
from planner.metrics import evaluate


def test_balanced_indices_on_label_sorted_data():
    labels = [0] * 500 + [1] * 500          # IMDB-style: sorted by label
    idx = balanced_indices(labels, 100, seed=1)
    picked = [labels[i] for i in idx]
    assert len(idx) == 100 and len(set(idx)) == 100
    assert picked.count(0) == picked.count(1) == 50
    assert picked[:10] != [0] * 10 or picked[-10:] != [0] * 10   # shuffled, not blocks


def test_balanced_indices_rejects_impossible_request():
    with pytest.raises(ValueError):
        balanced_indices([0] * 5 + [1] * 100, 20)


def test_evaluate_reports_both_classes():
    m = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


def test_prompt_contains_label_name_and_confidence():
    msgs = build_messages("analyst", "Great film!", Prediction("positive", 0.97))
    assert msgs[0]["role"] == "system"
    assert "positive" in msgs[1]["content"] and "97%" in msgs[1]["content"]


def test_agent_works_offline_without_key():
    agent = Agent("strategist", api_key="")
    assert not agent.online
    reply = agent.respond("I could not focus.", Prediction("negative", 0.9))
    assert reply == offline_response("strategist", Prediction("negative", 0.9))


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        Agent("poet", api_key="")
