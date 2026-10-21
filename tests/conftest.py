"""Setup shared by all tests."""

import faulthandler
import signal
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from fleetlab.runtime import configure_process
from fleetlab.sim import lerobot_compat

if TYPE_CHECKING:
    from pathlib import Path

    from skills_support import Exports

# LIBERO asks for input on standard input when it is imported and finds no
# config file. pytest imports this file before any test module, so writing the
# config here keeps every test clear of that prompt, including one that imports
# LIBERO at module level.
configure_process()

# Cover setup, the test and teardown, including runs made by shared fixtures.
# SIGALRM raises in the main thread so the engine can clean up its workers;
# the hard deadline also bounds native code or a stuck cleanup. These tests
# run on Linux, where ITIMER_REAL is available.
TEST_DEADLINE_S = 300


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Iterator[None]:
    def expired(signum: int, frame: object) -> None:
        faulthandler.dump_traceback(file=2)
        pytest.fail(f"{item.nodeid} exceeded its {TEST_DEADLINE_S}s test deadline", pytrace=False)

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, TEST_DEADLINE_S)
    faulthandler.dump_traceback_later(TEST_DEADLINE_S + 30, exit=True, file=2)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def lerobot_on_robosuite_15() -> None:
    """For tests that build LeRobot's own LIBERO environments, which need the layer.

    It stays installed for the rest of the test process; nothing else notices it.
    """
    lerobot_compat.install()


@pytest.fixture(scope="session")
def skill_exports(tmp_path_factory: pytest.TempPathFactory) -> "Exports":
    """The skills' toy fleet and the log's toy run, written and exported once per session."""
    from skills_support import build_exports

    return build_exports(tmp_path_factory.mktemp("skills"))


@pytest.fixture(scope="session")
def mini_export() -> "Path":
    """The miniature fleet's export from its fixtures; skipped where absent."""
    from fleetlab.integration.fixtures import FIXTURES_ROOT

    export = FIXTURES_ROOT / "mini-v1" / "export"
    if not (export / "log" / "episodes.parquet").is_file():
        pytest.skip(f"the miniature fleet's fixtures are not on this machine ({export})")
    return export


# The interpreter's exit after the last test once hung for hours (a run of 2026-10-08:
# every test passed, then the process slept in a futex wait beside a multiprocessing
# resource tracker). If shutdown takes longer than this, every thread's stack is written
# to standard error and the process exits with status 1, so `make check` fails visibly
# and names the culprit instead of hanging.
EXIT_WATCHDOG_S = 120


def pytest_unconfigure(config: pytest.Config) -> None:
    faulthandler.dump_traceback_later(EXIT_WATCHDOG_S, exit=True, file=sys.stderr)
