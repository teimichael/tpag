"""Test llm adapter for the TPAG replication package."""

from tpag.adapters.llm import strip_thinking
from tpag.agents.guards import parse_decision


def test_strip_thinking_pure_unit():
    assert strip_thinking("<think>reasoning</think>BLOCK") == "BLOCK"
    assert strip_thinking("APPROVE") == "APPROVE"


def test_parse_decision_json_and_fallback():
    assert parse_decision('{"decision": "APPROVE"}') is True
    assert parse_decision('{"decision": "BLOCK"}') is False
    assert parse_decision("BLOCK -- unsafe") is False
    assert parse_decision("nonsense") is False
