"""Local CLI protocol fixtures; these tests make no inference requests."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import pytest

from vehicle_days import codex_model
from vehicle_days.codex_model import (
    CodexModel, CodexModelError, CodexProtocolError, CodexQuotaError,
    CodexResourceLimitError, DISABLED_FEATURES, INSTRUCTION_SUFFIX, MAX_TEXT_BYTES,
    CONTROLLER_MEMORY_RESERVE, MEMORY_LIMIT,
)


def fixture_cli(tmp_path, body, *, login="Logged in using ChatGPT"):
    path = tmp_path / "fixture-codex"
    path.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys, time
def emit(event):
    print(json.dumps(event), flush=True)
if "--version" in sys.argv:
    print("codex-cli TEST-FIXTURE")
    sys.exit(0)
if "login" in sys.argv:
    print({login!r})
    sys.exit(0)
prompt = sys.stdin.read()
configs = dict(sys.argv[i+1].split("=", 1) for i,arg in enumerate(sys.argv) if arg == "-c")
{body}
''')
    path.chmod(0o755)
    return path


def success(text_expression='"fixture output"', *, output_tokens=7):
    return f'''
emit({{"type":"thread.started","thread_id":"fixture-only","model":"gpt-6-luna"}})
emit({{"type":"turn.started"}})
emit({{"type":"item.completed","item":{{"id":"fixture-message","type":"agent_message","text":{text_expression}}}}})
emit({{"type":"turn.completed","usage":{{"input_tokens":41,"cached_input_tokens":12,"output_tokens":{output_tokens},"reasoning_output_tokens":0}}}})
'''


def test_isolated_cli_flags_instructions_usage_and_metadata(tmp_path, monkeypatch):
    body = '''
assert list(pathlib.Path.cwd().iterdir()) == []
instruction_path = pathlib.Path(json.loads(configs["model_instructions_file"]))
text = json.dumps({"instructions":instruction_path.read_text(),"prompt":prompt,"cwd":str(pathlib.Path.cwd()),"api_key_present":("OPENAI_API_KEY" in os.environ or "CODEX_API_KEY" in os.environ),"configs":configs})
''' + success("text", output_tokens=19)
    executable = fixture_cli(tmp_path, body)
    monkeypatch.setenv("OPENAI_API_KEY", "TEST-ONLY-MUST-NOT-PASS")
    monkeypatch.setenv("CODEX_API_KEY", "TEST-ONLY-MUST-NOT-PASS")
    provider = CodexModel(executable=executable)
    result = provider.generate("owner-private-system", "owner-private-prompt", seed=73,
                               max_tokens=10, deadline=time.monotonic()+5)
    response = json.loads(result["text"])
    assert response["instructions"] == "owner-private-system" + INSTRUCTION_SUFFIX
    assert "owner-private-prompt" in response["prompt"]
    assert response["api_key_present"] is False
    assert not Path(response["cwd"]).exists()
    for feature in DISABLED_FEATURES:
        assert response["configs"][f"features.{feature}"] == "false"
    assert response["configs"]["forced_login_method"] == '"chatgpt"'
    assert result["input_tokens"] == 41 and result["cached_input_tokens"] == 12
    assert result["output_tokens"] == 19 and result["budget_overrun_tokens"] == 9
    assert result["actual_model"] == "gpt-6-luna"
    assert "CLI lacks hard server cap" in result["output_budget_control"]
    assert "no remote sampling seed" in result["seed_control"]
    assert len(result["provider_attempts"][0]["events"]) == 4
    provider.close()


def test_api_key_authentication_is_refused(tmp_path):
    executable = fixture_cli(tmp_path, "", login="Logged in using an API key")
    with pytest.raises(CodexModelError, match="API-key authentication is refused"):
        CodexModel(executable=executable)


def test_tool_event_fails_immediately_and_preserves_evidence(tmp_path):
    executable = fixture_cli(tmp_path, '''
emit({"type":"item.started","item":{"type":"command_execution","command":"TEST-FIXTURE-NOT-EXECUTED"}})
time.sleep(5)
''')
    provider = CodexModel(executable=executable)
    start = time.monotonic()
    with pytest.raises(CodexProtocolError, match="no tools are permitted") as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=start+10)
    assert time.monotonic() - start < 2
    assert len(exc.value.evidence["attempts"]) == 1
    assert "command_execution" in exc.value.evidence["attempts"][0]["stdout"]
    assert not provider._processes


def test_quota_failure_stops_without_retry(tmp_path):
    executable = fixture_cli(tmp_path, '''
emit({"type":"error","message":"Usage limit reached. Try again after reset."})
time.sleep(5)
''')
    provider = CodexModel(executable=executable, retries=2)
    with pytest.raises(CodexQuotaError) as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=time.monotonic()+10)
    assert len(exc.value.evidence["attempts"]) == 1


def test_deadline_kills_generation_and_saves_partial_protocol(tmp_path):
    executable = fixture_cli(tmp_path, '''
emit({"type":"turn.started"})
time.sleep(5)
''')
    provider = CodexModel(executable=executable)
    start = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=start+.2)
    assert time.monotonic() - start < 2
    assert exc.value.evidence["attempts"][0]["events"] == [{"type":"turn.started"}]
    assert not provider._processes


def test_parallel_calls_have_separate_sessions_prompts_and_process_tracking(tmp_path):
    executable = fixture_cli(tmp_path, "time.sleep(.1)\n" + success('json.dumps({"prompt":prompt,"cwd":str(pathlib.Path.cwd())})'))
    provider = CodexModel(executable=executable)
    def call(index):
        return provider.generate(f"private system {index}", f"private prompt {index}", seed=index,
                                 max_tokens=20, deadline=time.monotonic()+5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call, range(4)))
    payloads = [json.loads(item["text"]) for item in results]
    assert len({item["cwd"] for item in payloads}) == 4
    for index, item in enumerate(payloads):
        assert f"private prompt {index}" in item["prompt"]
        assert all(f"private prompt {peer}" not in item["prompt"] for peer in range(4) if peer != index)
    assert not provider._processes


def test_transient_retry_preserves_same_model_and_records_both_attempts(tmp_path, monkeypatch):
    provider = CodexModel(executable=fixture_cli(tmp_path, success()), retries=1)
    calls = []
    def attempt(*args):
        calls.append(1)
        if len(calls) == 1:
            raise CodexModelError("network connection reset", {"usage": {"input_tokens":3,"output_tokens":2},"stdout":"fixture partial"})
        return {"usage": {"input_tokens":5,"output_tokens":4},"actual_model":None,"text":"complete fixture"}
    monkeypatch.setattr(provider, "_attempt", attempt)
    result = provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=time.monotonic()+5)
    assert len(result["provider_attempts"]) == 2
    assert result["input_tokens"] == 8 and result["output_tokens"] == 6
    assert result["model_requested"] == "gpt-6-luna"


def test_cancellation_stops_active_generation_with_its_partial_evidence(tmp_path):
    executable = fixture_cli(tmp_path, '''
emit({"type":"turn.started"})
time.sleep(5)
''')
    provider = CodexModel(executable=executable)
    start = time.monotonic()
    with pytest.raises(InterruptedError) as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=start+10,
                          cancel=lambda: time.monotonic()-start > .2)
    assert time.monotonic() - start < 2
    assert exc.value.evidence["attempts"][0]["events"] == [{"type":"turn.started"}]
    assert not provider._processes


def test_oversized_output_is_rejected_and_preserved_without_truncation(tmp_path):
    executable = fixture_cli(tmp_path, success(f'"x" * {MAX_TEXT_BYTES + 1}'))
    provider = CodexModel(executable=executable)
    with pytest.raises(CodexProtocolError, match="preserved without truncation") as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=time.monotonic()+5)
    events = exc.value.evidence["attempts"][0]["events"]
    messages = [event["item"] for event in events if event.get("type") == "item.completed"]
    assert len(messages[0]["text"]) == MAX_TEXT_BYTES + 1


def test_memory_limit_counts_aggregate_rss_and_controller_reservation(tmp_path, monkeypatch):
    provider = CodexModel(executable=fixture_cli(tmp_path, success()), retries=2)
    # Each CLI could fit under the ceiling individually; it is the combined
    # process tree plus the separately bounded Docker workers that matters.
    monkeypatch.setattr(codex_model, "_process_tree_rss_bytes", lambda: {
        "rss_bytes": MEMORY_LIMIT - CONTROLLER_MEMORY_RESERVE + 1, "process_count": 5})
    with pytest.raises(CodexResourceLimitError, match="48-GiB") as exc:
        provider.generate("system", "prompt", seed=1, max_tokens=20, deadline=time.monotonic()+5)
    attempts = exc.value.evidence["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["memory_sample"]["accounted_bytes"] == MEMORY_LIMIT + 1
    assert attempts[0]["memory_sample"]["process_count"] == 5
    assert not provider._processes


def test_memory_monitor_includes_children_and_grandchildren_but_not_other_apps(monkeypatch):
    from types import SimpleNamespace
    parent = os.getpid()
    rows = f"{parent} 1 100\n999991 {parent} 200\n999992 999991 300\n999993 1 999999\n"
    monkeypatch.setattr(codex_model.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=rows))
    assert codex_model._process_tree_rss_bytes() == {"rss_bytes": 600 * 1024, "process_count": 3}
