"""Tests for wait_for_release_signal in evaluation/benchmarks/swe_bench/run_infer.py.

run_infer.py pulls in heavy eval dependencies, so the helper is extracted from the
source with ast and executed against stubs. No replica can drift from the source.
"""

import ast
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


def _load_helper(
    should_continue: Callable[[], bool],
) -> Callable[[str], tuple[bool, float]]:
    tree = ast.parse(RUN_INFER.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef) and node.name == 'wait_for_release_signal'
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                getattr(t, 'id', None) == 'RELEASE_SIGNAL_POLL_S' for t in node.targets
            )
        )
    ]
    assert len(nodes) == 2, 'wait_for_release_signal or RELEASE_SIGNAL_POLL_S not found'
    namespace = {'time': time, 'should_continue': should_continue}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(RUN_INFER), 'exec'),
        namespace,
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


def test_poll_interval_is_half_second():
    assert 'RELEASE_SIGNAL_POLL_S = 0.5' in RUN_INFER.read_text()


@pytest.mark.parametrize(
    'content, released', [('release', True), ('cancel', False), ('release\n', True)]
)
def test_signal_present_returns_immediately(tmp_path, content, released):
    signal = tmp_path / 'release_signal'
    signal.write_text(content)
    wait = _load_helper(lambda: True)

    got_released, wait_s = wait(str(signal))

    assert got_released is released
    assert 0 <= wait_s < 1


@pytest.mark.parametrize('content, released', [('release', True), ('cancel', False)])
def test_missing_file_keeps_waiting_until_signal(tmp_path, content, released):
    signal = tmp_path / 'release_signal'

    def on_call(n: int) -> bool:
        if n == 4:
            signal.write_text(content)
        return True

    should_continue, calls = _counting_continue(on_call)
    wait = _load_helper(should_continue)

    got_released, wait_s = wait(str(signal))

    assert got_released is released
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

    assert wait(str(signal))[0] is True
    assert calls[0] == 3


def test_shutdown_stops_wait_without_release(tmp_path):
    should_continue, calls = _counting_continue(lambda n: n < 3)
    wait = _load_helper(should_continue)

    released, wait_s = wait(str(tmp_path / 'never_written'))

    assert released is False
    assert calls[0] == 3
    assert wait_s >= 2 * 0.01
