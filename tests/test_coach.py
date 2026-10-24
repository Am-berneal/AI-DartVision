import json

import pandas as pd

from dartvision import coach
from dartvision.data import simulate_players

THROWS = simulate_players(seed=125)
PLAYER = "Player C (low & left)"


def test_rule_based_feedback_is_fully_verified(monkeypatch):
    monkeypatch.setenv("DARTVISION_LLM", "off")
    r = coach.coach_feedback(THROWS, PLAYER)
    assert r["mode"] == "Rule-based"
    assert r["verified"] and r["unverified"] == []


def test_verify_rejects_invented_numbers():
    f = coach.player_facts(THROWS, PLAYER)
    ok, bad = coach.verify_numbers("You lose 99.9% of darts and drift 123.4 mm", f)
    assert not ok and "99.9%" in bad and "123.4mm" in bad


def test_verify_ignores_counts_and_region_codes():
    f = coach.player_facts(THROWS, PLAYER)
    ok, _ = coach.verify_numbers("Throw 3 sets of 30 darts at T20 in session 4", f)
    assert ok


def test_llm_answer_with_bad_numbers_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.delenv("DARTVISION_LLM", raising=False)
    bad = {k: "You will gain 77.7% next week." for k in coach.FEEDBACK_SCHEMA["required"]}
    monkeypatch.setattr(coach, "_call_llm", lambda *a: (json.dumps(bad), "Gemini", "test-model"))
    monkeypatch.setattr(coach, "_log", lambda e: None)
    r = coach.coach_feedback(THROWS, PLAYER)
    assert r["mode"] == "Rule-based" and r["attempts"] == 2 and r["verified"]


def test_llm_answer_with_true_numbers_is_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.delenv("DARTVISION_LLM", raising=False)
    f = coach.player_facts(THROWS, PLAYER)
    good = {k: f"Your hit rate is {f['hit_rate_pct']}% and your offset is {f['bias_mm']} mm."
            for k in coach.FEEDBACK_SCHEMA["required"]}
    monkeypatch.setattr(coach, "_call_llm", lambda *a: (json.dumps(good), "Gemini", "test-model"))
    monkeypatch.setattr(coach, "_log", lambda e: None)
    r = coach.coach_feedback(THROWS, PLAYER, language="Hindi")
    assert r["provider"] == "Gemini" and r["attempts"] == 1 and r["language"] == "Hindi"