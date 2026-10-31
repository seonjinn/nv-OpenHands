"""Tests for the prewarm hold helpers in evaluation/benchmarks/swe_bench/run_infer.py.

run_infer.py pulls in heavy eval dependencies, so the helpers are extracted from the
source with ast and executed against stubs. No replica can drift from the source.
"""

import ast
import os
import time
from pathlib import Path
from typing import Callable

import pytest

RUN_INFER = (
    Path(__file__).resolve().parents[3]
    / 'evaluation'
    / 'benchmarks'
    / 'swe_bench'
    / 'run_infer.py'
)


def _load(names: set[str], **stubs: object) -> dict:
    tree = ast.parse(RUN_INFER.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, 'id', None) in names for t in node.targets)
        )
    ]
    assert len(nodes) == len(names), f'not all of {sorted(names)} found'
    namespace = {'os': os, 'time': time, **stubs}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(RUN_INFER), 'exec'),
        namespace,
    )
    return namespace


def _load_helper(
    should_continue: Callable[[], bool],
) -> Callable[[str], tuple[str, float]]:
    namespace = _load(
        {'wait_for_release_signal', 'RELEASE_SIGNAL_POLL_S'},
        should_continue=should_continue,
    )
    namespace['RELEASE_SIGNAL_POLL_S'] = 0.01
    return namespace['wait_for_release_signal']


def _counting_continue(
    on_call: Callable[[int], bool],
) -> tuple[Callable[[], bool], list[int]]:
    calls = [0]

    def should_continue() -> bool:
        calls[0] += 1
        return on_call(calls[0])

    return should_continue, calls


def test_poll_interval_defaults_to_half_second(monkeypatch):
    monkeypatch.delenv('OH_RELEASE_SIGNAL_POLL_S', raising=False)
    assert _load({'RELEASE_SIGNAL_POLL_S'})['RELEASE_SIGNAL_POLL_S'] == 0.5


def test_poll_interval_reads_env(monkeypatch):
    monkeypatch.setenv('OH_RELEASE_SIGNAL_POLL_S', '0.05')
    assert _load({'RELEASE_SIGNAL_POLL_S'})['RELEASE_SIGNAL_POLL_S'] == 0.05


@pytest.mark.parametrize(
    'content, outcome',
    [('release', 'release'), ('cancel', 'cancel'), ('release\n', 'release')],
)
def test_signal_present_returns_immediately(tmp_path, content, outcome):
    signal = tmp_path / 'release_signal'
    signal.write_text(content)
    wait = _load_helper(lambda: True)

    got_outcome, wait_s = wait(str(signal))

    assert got_outcome == outcome
    assert 0 <= wait_s < 1


@pytest.mark.parametrize('content', ['release', 'cancel'])
def test_missing_file_keeps_waiting_until_signal(tmp_path, content):
    signal = tmp_path / 'release_signal'

    def on_call(n: int) -> bool:
        if n == 4:
            signal.write_text(content)
        return True

    should_continue, calls = _counting_continue(on_call)
    wait = _load_helper(should_continue)

    got_outcome, wait_s = wait(str(signal))

    assert got_outcome == content
    assert calls[0] == 4
    assert wait_s >= 3 * 0.01


def test_empty_or_unknown_content_keeps_waiting(tmp_path):
    signal = tmp_path / 'release_signal'
    signal.write_text('')

    def on_call(n: int) -> bool:
        if n == 2:
            signal.write_text('pending')
        if n == 3:
            signal.write_text('release')
        return True

    should_continue, calls = _counting_continue(on_call)
    wait = _load_helper(should_continue)

    assert wait(str(signal))[0] == 'release'
    assert calls[0] == 3


def test_shutdown_stops_wait_without_release(tmp_path):
    should_continue, calls = _counting_continue(lambda n: n < 3)
    wait = _load_helper(should_continue)

    outcome, wait_s = wait(str(tmp_path / 'never_written'))

    assert outcome == 'shutdown'
    assert calls[0] == 3
    assert wait_s >= 2 * 0.01


def _deadline() -> Callable[..., float | None]:
    return _load({'release_deadline_epoch'}, Mapping=dict)['release_deadline_epoch']


def test_deadline_shifts_cold_budget_by_hold():
    env = {'OH_TIMEOUT_START_EPOCH': '1000.5', 'OH_AGENT_TIMEOUT_S': '3600'}
    assert _deadline()(env, 42.0) == 1000.5 + 3600 + 42.0


@pytest.mark.parametrize(
    'env',
    [
        {},
        {'OH_TIMEOUT_START_EPOCH': '1000'},
        {'OH_AGENT_TIMEOUT_S': '3600'},
        {'OH_TIMEOUT_START_EPOCH': 'abc', 'OH_AGENT_TIMEOUT_S': '3600'},
        {'OH_TIMEOUT_START_EPOCH': '1000', 'OH_AGENT_TIMEOUT_S': ''},
    ],
)
def test_deadline_absent_or_bad_env_is_none(env):
    assert _deadline()(env, 1.0) is None


def _watchdog(exits: list[int]) -> Callable[[float], None]:
    import threading

    namespace = _load(
        {'start_release_deadline_watchdog', 'PREWARM_TIMEOUT_EXIT_CODE'},
        threading=threading,
    )
    namespace['os'] = type('FakeOs', (), {'_exit': staticmethod(exits.append)})
    return namespace['start_release_deadline_watchdog']


def test_watchdog_past_deadline_exits_at_once():
    exits: list[int] = []
    _watchdog(exits)(time.time() - 5)
    assert exits == [124]


def test_watchdog_future_deadline_exits_later():
    exits: list[int] = []
    _watchdog(exits)(time.time() + 0.2)
    assert exits == []
    deadline = time.time() + 5
    while not exits and time.time() < deadline:
        time.sleep(0.01)
    assert exits == [124]
