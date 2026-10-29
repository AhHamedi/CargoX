"""Text-only inference through the official CLI's existing ChatGPT sign-in.

Every generation uses a fresh ephemeral session and empty working directory.
Credentials remain entirely with Codex CLI. No API-key or model substitution is
available here. CLI usage, protocol events, requested allowances and overruns
are preserved rather than claiming a sampling seed or hard token cap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time


APP_CLI = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
MAX_STDOUT = 1_048_576
MAX_STDERR = 262_144
MAX_TEXT_BYTES = 65_536
MEMORY_LIMIT = 48 * 1024**3
CONTROLLER_MEMORY_RESERVE = 8 * 128 * 1024**2
MEMORY_SAMPLE_SECONDS = 1.0
CONFIG_REFERENCE = "https://learn.chatgpt.com/docs/config-file/config-reference"
INSTRUCTION_SUFFIX = ("\n\nYou are a text-only research participant. Use only the supplied private context. "
                      "Do not access files, tools, browsers, other sessions, or external information. "
                      "Return only the requested response.\n")

# These capabilities are disabled in addition to using a read-only sandbox.
# Each native protocol item is also checked; unexpected tools fail the call.
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "remote_plugin",
    "multi_agent", "multi_agent_v2", "memories", "chronicle", "browser_use",
    "browser_use_external", "computer_use", "in_app_browser", "code_mode",
    "code_mode_host", "view_image", "image_generation", "workspace_dependencies",
    "goals", "sleep_tool", "skill_search", "hooks", "shell_snapshot",
    "unbounded_connection_retries", "tool_suggest", "step_model_switching",
    "current_time_reminder", "stable_environment_tools",
)
CONFIG = {
    "forced_login_method": "chatgpt", "model_provider": "openai",
    "web_search": "disabled", "project_doc_max_bytes": 0,
    "features.skip_host_skill_discovery": True,
    "skills.include_instructions": False, "skills.bundled.enabled": False,
    "include_apps_instructions": False,
    "include_collaboration_mode_instructions": False,
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
    "agents.enabled": False,
    "history.persistence": "none", "mcp_servers": {},
    "hide_agent_reasoning": True,
}


class CodexModelError(RuntimeError):
    """Provider failure; evidence is safe to save with failed stage artifacts."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


class CodexQuotaError(CodexModelError):
    """A membership usage limit stops the study at its recorded boundary."""


class CodexProtocolError(CodexModelError):
    """Unexpected tool use or an incomplete/unrecognized generation protocol."""


class CodexResourceLimitError(CodexModelError):
    """The local preparation process tree exceeded its accounted RAM ceiling."""


def _process_tree_rss_bytes():
    """Current parent/descendant RSS from ps; RSS values are KiB on macOS/Linux.

    Counting shared pages in every process is conservative. Docker's VM is not
    a descendant, so its eight per-controller limits are reserved separately.
    No process command lines or environment values are collected.
    """
    try:
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                                text=True, timeout=2, check=True)
        rows = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3:
                pid, parent, rss = (int(value) for value in fields)
                rows[pid] = (parent, rss * 1024)
        if os.getpid() not in rows:
            raise ValueError("preparation process was absent from RSS sample")
        descendants = {os.getpid()}
        while True:
            discovered = {pid for pid, (parent, rss) in rows.items() if parent in descendants}
            if discovered <= descendants:
                break
            descendants.update(discovered)
        return {"rss_bytes": sum(rows[pid][1] for pid in descendants),
                "process_count": len(descendants)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise CodexResourceLimitError(f"Cannot enforce local memory ceiling: RSS monitor failed: {exc}") from None


def _environment():
    # Do not inherit agent/thread control variables or any API-key credentials.
    keep = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
            "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY",
            "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "CODEX_HOME"}
    return {key: value for key, value in os.environ.items() if key in keep}


def _override(key, value):
    # JSON scalar/array/object syntax used here is also valid inline TOML for
    # these exact scalar keys (the sole object value is the empty table {}).
    return ["-c", f"{key}={json.dumps(value, separators=(',', ':'))}"]


def _quota(text):
    lowered = text.lower()
    return any(fragment in lowered for fragment in (
        "usage limit", "usage_limit", "quota exceeded", "insufficient_quota",
        "rate limit", "rate_limit", "too many requests", "http 429", "status 429",
        "purchase credits", "upgrade your plan",
    ))


def _transient(text):
    lowered = text.lower()
    return any(fragment in lowered for fragment in (
        "connection reset", "connection closed", "connection refused", "network error",
        "network connection", "websocket", "temporarily unavailable", "service unavailable",
        "bad gateway", "gateway timeout", "http 502", "http 503", "http 504",
        "status 502", "status 503", "status 504", "dns error", "stream disconnected",
    ))


def _stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def _check_event(event):
    if not isinstance(event, dict):
        raise CodexProtocolError("Codex protocol records must be JSON objects")
    kind = event.get("type")
    if kind in ("item.started", "item.updated", "item.completed"):
        item_kind = event.get("item", {}).get("type")
        if item_kind not in ("agent_message", "reasoning", "error"):
            raise CodexProtocolError(f"Forbidden or unknown Codex item: {item_kind!r}; no tools are permitted")
    elif kind not in ("thread.started", "turn.started", "turn.completed", "turn.failed", "error", "warning"):
        raise CodexProtocolError(f"Unrecognized Codex protocol event: {kind!r}")


class CodexModel:
    parallelism = 8

    def __init__(self, model="gpt-6-luna", reasoning_effort="low", *, executable=None, retries=2):
        selected = str(executable or (APP_CLI if APP_CLI.is_file() else shutil.which("codex") or ""))
        if not selected or not Path(selected).is_file():
            raise CodexModelError("Official Codex CLI is unavailable; no API-key fallback is configured")
        self.executable = selected
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self._processes = set()
        self._process_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._memory_last_checked = float("-inf")
        self._memory_snapshot = None
        self._memory_peak = 0
        environment = _environment()
        version = subprocess.run([selected, "--version"], env=environment, capture_output=True,
                                 text=True, timeout=10, check=True).stdout.strip()
        login = subprocess.run([selected, "-c", 'forced_login_method="chatgpt"', "login", "status"],
                               env=environment, capture_output=True, text=True, timeout=15)
        if login.returncode or "logged in using chatgpt" not in (login.stdout + login.stderr).lower():
            raise CodexModelError("Codex CLI must be signed in using ChatGPT; API-key authentication is refused")
        self.settings = {
            "model_id": model, "provider": "official-codex-cli-chatgpt", "cli_version": version,
            "auth": "existing ChatGPT sign-in; credentials managed by official CLI",
            "reasoning_effort": reasoning_effort, "temperature": None,
            "sampling_seed": "recorded metadata only; CLI has no remote sampling seed control",
            "output_budget": "requested allowance; CLI lacks hard server cap",
            "local_response_byte_limit": MAX_TEXT_BYTES,
            "session_isolation": "fresh ephemeral session, empty cwd, separate supplied instructions",
            "tools": "disabled; protocol rejects every tool item",
            "native_isolation_overrides": {"agents.enabled": False},
            "transient_retries": retries, "model_substitution": False,
            "parallelism": self.parallelism,
            "usage_accounting": "sum reported attempt usage; failed requests may omit token usage",
            "instruction_suffix": INSTRUCTION_SUFFIX,
            "actual_model_reporting": "record actual identifier if CLI emits it; otherwise requested slug only",
            "local_memory_ceiling_bytes": MEMORY_LIMIT,
            "memory_guard": "one-second process-tree RSS samples plus 1-GiB Docker reservation; shared pages counted conservatively",
            "configuration_reference": CONFIG_REFERENCE,
        }

    def _check_memory(self):
        with self._memory_lock:
            now = time.monotonic()
            if now - self._memory_last_checked >= MEMORY_SAMPLE_SECONDS:
                snapshot = _process_tree_rss_bytes()
                snapshot.update(controller_reserve_bytes=CONTROLLER_MEMORY_RESERVE,
                                accounted_bytes=snapshot["rss_bytes"] + CONTROLLER_MEMORY_RESERVE,
                                limit_bytes=MEMORY_LIMIT)
                self._memory_snapshot = snapshot
                self._memory_last_checked = now
                self._memory_peak = max(self._memory_peak, snapshot["accounted_bytes"])
            snapshot = dict(self._memory_snapshot)
        if snapshot["accounted_bytes"] > MEMORY_LIMIT:
            self.close()
            raise CodexResourceLimitError(
                "48-GiB local preparation memory ceiling exceeded "
                f"(process-tree RSS {snapshot['rss_bytes']} bytes plus "
                f"{CONTROLLER_MEMORY_RESERVE} reserved controller bytes)",
                {"memory_sample": snapshot})
        return snapshot

    def _command(self, directory, instructions):
        command = [self.executable, "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules",
                   "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--model", self.model, "--json", "--color", "never", "--cd", str(directory)]
        for key, value in CONFIG.items():
            command.extend(_override(key, value))
        command.extend(_override("model_reasoning_effort", self.reasoning_effort))
        command.extend(_override("model_instructions_file", str(instructions)))
        for name in DISABLED_FEATURES:
            command.extend(_override(f"features.{name}", False))
        command.append("-")
        return command

    def _attempt(self, system, prompt, max_tokens, deadline, cancel):
        raw = {"stdout": "", "stderr": "", "events": [], "exit_code": None,
               "actual_model": None, "usage": None}
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        pending_line = bytearray()
        with tempfile.TemporaryDirectory(prefix="vehicle-days-codex-") as temporary:
            temporary = Path(temporary)
            directory = temporary / "empty"
            directory.mkdir()
            instructions = temporary / "instructions.txt"
            instructions.write_text(system + INSTRUCTION_SUFFIX)
            requested = prompt + f"\nRequested output allowance: {max_tokens} tokens. Be concise; do not include hidden reasoning."
            payload = requested.encode("utf-8")
            command = self._command(directory, instructions)
            process = subprocess.Popen(command, cwd=directory, env=_environment(), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                       start_new_session=True)
            with self._process_lock:
                self._processes.add(process)
            selector = selectors.DefaultSelector()
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            sent = 0
            try:
                while selector.get_map():
                    if cancel():
                        raise InterruptedError("Campaign cancellation requested")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("20-minute daily preparation limit exceeded during Codex generation")
                    raw["memory_sample"] = self._check_memory()
                    for key, mask in selector.select(min(0.1, max(0, deadline-time.monotonic()))):
                        stream, channel = key.fileobj, key.data
                        if channel == "stdin":
                            try:
                                sent += os.write(stream.fileno(), payload[sent:sent + 65_536])
                            except BrokenPipeError:
                                sent = len(payload)
                            if sent == len(payload):
                                selector.unregister(stream)
                                stream.close()
                            continue
                        part = os.read(stream.fileno(), 65_536)
                        if not part:
                            selector.unregister(stream)
                            continue
                        buffers[channel].extend(part)
                        limit = MAX_STDOUT if channel == "stdout" else MAX_STDERR
                        if len(buffers[channel]) > limit:
                            raise CodexProtocolError(f"Codex {channel} exceeded {limit} bytes")
                        if channel == "stdout":
                            pending_line.extend(part)
                            while b"\n" in pending_line:
                                line, _, remaining = pending_line.partition(b"\n")
                                pending_line = bytearray(remaining)
                                if not line.strip():
                                    continue
                                try:
                                    event = json.loads(line)
                                except (ValueError, UnicodeError) as exc:
                                    raise CodexProtocolError(f"Invalid Codex JSONL event: {exc}") from None
                                raw["events"].append(event)
                                _check_event(event)
                                if event.get("type") == "turn.completed":
                                    raw["usage"] = event.get("usage")
                                if event.get("type") in ("error", "turn.failed") and _quota(json.dumps(event)):
                                    raise CodexQuotaError("ChatGPT membership usage limit reached; resume after reset")
                if pending_line.strip():
                    raise CodexProtocolError("Codex ended with an incomplete JSONL record")
                raw["exit_code"] = process.wait(timeout=1)
            except BaseException as exc:
                _stop(process)
                if isinstance(exc, CodexResourceLimitError):
                    raw["resource_limit"] = exc.evidence
                raw["stdout"] = buffers["stdout"].decode("utf-8", errors="replace")
                raw["stderr"] = buffers["stderr"].decode("utf-8", errors="replace")
                raw["exit_code"] = process.returncode
                exc.evidence = raw
                raise
            finally:
                _stop(process)
                selector.close()
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream and not stream.closed:
                        stream.close()
                with self._process_lock:
                    self._processes.discard(process)
        raw["stdout"] = buffers["stdout"].decode("utf-8", errors="replace")
        raw["stderr"] = buffers["stderr"].decode("utf-8", errors="replace")
        completed = [event for event in raw["events"] if event.get("type") == "turn.completed"]
        failed = [event for event in raw["events"] if event.get("type") == "turn.failed"]
        if raw["exit_code"] or failed or not completed:
            error_text = raw["stderr"] + "\n" + json.dumps([event for event in raw["events"] if event.get("type") in ("error", "turn.failed")])
            if _quota(error_text):
                raise CodexQuotaError("ChatGPT membership usage limit reached; resume after reset", raw)
            raise CodexModelError("Codex generation failed: " + error_text[-1500:], raw)
        if len(completed) != 1:
            raise CodexProtocolError("Expected exactly one completed Codex turn", raw)
        raw["usage"] = completed[0].get("usage")
        for event in raw["events"]:
            actual = event.get("model") or event.get("model_id")
            if actual:
                raw["actual_model"] = actual
                if actual != self.model:
                    raise CodexProtocolError(f"Unexpected model {actual!r}; requested {self.model!r}", raw)
        messages = [event["item"] for event in raw["events"] if event.get("type") == "item.completed"
                    and event.get("item", {}).get("type") == "agent_message"]
        finals = [item for item in messages if item.get("phase") == "final"]
        if not messages:
            raise CodexProtocolError("Codex completed without an agent response", raw)
        text = (finals or messages)[-1].get("text")
        if not isinstance(text, str) or not text.strip():
            raise CodexProtocolError("Codex returned an empty response", raw)
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise CodexProtocolError(f"Response exceeds local {MAX_TEXT_BYTES}-byte limit; preserved without truncation", raw)
        if not isinstance(raw["usage"], dict) or not all(isinstance(raw["usage"].get(key), int) for key in ("input_tokens", "output_tokens")):
            raise CodexProtocolError("Codex did not report input/output token usage", raw)
        raw["text"] = text
        return raw

    def generate(self, system, prompt, *, seed, max_tokens, deadline, cancel=lambda: False):
        started = time.monotonic()
        attempts = []
        for attempt_number in range(self.retries + 1):
            if cancel():
                raise InterruptedError("Campaign cancellation requested")
            if time.monotonic() >= deadline:
                raise TimeoutError("Daily preparation deadline reached before Codex generation")
            try:
                self._check_memory()
                result = self._attempt(system, prompt, max_tokens, deadline, cancel)
                attempts.append({"attempt": attempt_number + 1, **result})
                usage = result["usage"]
                reported_usage = [item["usage"] for item in attempts if isinstance(item.get("usage"), dict)]
                input_tokens = sum(item.get("input_tokens", 0) for item in reported_usage)
                output_tokens = sum(item.get("output_tokens", 0) for item in reported_usage)
                cached = [item["cached_input_tokens"] for item in reported_usage if isinstance(item.get("cached_input_tokens"), int)]
                reasoning = [item.get("reasoning_output_tokens", item.get("output_tokens_details", {}).get("reasoning_tokens")) for item in reported_usage]
                reasoning = [value for value in reasoning if isinstance(value, int)]
                return {
                    "system": system, "prompt": prompt, "seed": seed,
                    "provider_system": system + INSTRUCTION_SUFFIX,
                    "provider_prompt": prompt + f"\nRequested output allowance: {max_tokens} tokens. Be concise; do not include hidden reasoning.",
                    "seed_control": self.settings["sampling_seed"],
                    "allocated_output_tokens": max_tokens,
                    "output_budget_control": self.settings["output_budget"],
                    "budget_overrun_tokens": max(output_tokens - max_tokens, 0),
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "response_output_tokens": usage["output_tokens"],
                    "cached_input_tokens": sum(cached) if cached else None,
                    "reasoning_output_tokens": sum(reasoning) if reasoning else None,
                    "unreported_attempt_usage": len(attempts) - len(reported_usage),
                    "model_requested": self.model, "actual_model": result["actual_model"],
                    "actual_model_source": "CLI protocol" if result["actual_model"] else "not emitted; requested slug recorded",
                    "text": result["text"], "finish_reason": "completed",
                    "wall_seconds": time.monotonic() - started,
                    "peak_local_accounted_bytes": self._memory_peak,
                    "provider_attempts": attempts,
                }
            except (CodexModelError, InterruptedError, TimeoutError) as exc:
                evidence = getattr(exc, "evidence", {})
                attempts.append({"attempt": attempt_number + 1, "error": str(exc), **evidence})
                exc.evidence = {"model_requested": self.model, "seed": seed,
                                "allocated_output_tokens": max_tokens, "attempts": attempts}
                if isinstance(exc, (CodexQuotaError, CodexProtocolError, InterruptedError, TimeoutError)):
                    raise
                diagnostics = str(exc) + "\n" + json.dumps(evidence)
                if attempt_number >= self.retries or not _transient(diagnostics):
                    raise
                # An explicit, bounded network retry preserves the requested
                # model and fresh private context. The common deadline applies.
                until = min(deadline, time.monotonic() + (attempt_number + 1))
                while time.monotonic() < until:
                    if cancel():
                        interrupted = InterruptedError("Campaign cancellation requested")
                        interrupted.evidence = exc.evidence
                        raise interrupted
                    time.sleep(min(0.1, until-time.monotonic()))
        raise AssertionError("unreachable retry state")

    def close(self):
        with self._process_lock:
            processes = list(self._processes)
        for process in processes:
            _stop(process)
