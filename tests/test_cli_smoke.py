from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage

from oscal_agent_lab import cli


@dataclass
class FakeGraph:
    reply: str = "stub answer"

    def invoke(self, state):
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=self.reply))
        return {"messages": messages}


def _set_inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    iterator = iter(values)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(iterator))


def test_help_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    _set_inputs(monkeypatch, ["help", "exit"])

    cli.main()

    out = capsys.readouterr().out
    assert "Available commands:" in out


def test_repl_prompt_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph(reply="AC-2 is account management."))
    _set_inputs(monkeypatch, ["What does AC-2 require?", "exit"])

    cli.main()

    out = capsys.readouterr().out
    assert "Agent: AC-2 is account management." in out


def test_validate_missing_file(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    _set_inputs(monkeypatch, ["validate /definitely/missing.json", "exit"])

    cli.main()

    out = capsys.readouterr().out
    assert "Error: File not found" in out
