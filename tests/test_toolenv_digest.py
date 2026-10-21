"""The production digest binds every mounted byte, bytecode included.

Test fixtures read theirs once (`toolenv_fixed`).
"""

from __future__ import annotations

from pathlib import Path

from toolenv_fixed import fixed_digest

from fleetlab.agents.toolenv import ToolEnv


def _env(tmp_path: Path) -> ToolEnv:
    for name in ("python", "site", "source", "skills"):
        (tmp_path / name).mkdir()
    (tmp_path / "site" / "pkg").mkdir()
    (tmp_path / "site" / "pkg" / "__init__.py").write_text("X = 1\n")
    (tmp_path / "source" / "mod.py").write_text("Y = 2\n")
    return ToolEnv(tmp_path / "python", tmp_path / "site", tmp_path / "source", tmp_path / "skills")


def test_the_production_digest_still_covers_bytecode_caches(tmp_path: Path) -> None:
    env = _env(tmp_path)
    before = env.digest()
    cache = tmp_path / "site" / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\x00compiled")
    assert env.digest() != before


def test_a_fixture_environment_keeps_the_fingerprint_it_was_created_with(tmp_path: Path) -> None:
    env = _env(tmp_path)
    fixed = fixed_digest(env)
    assert fixed.digest() == env.digest()
    (tmp_path / "site" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "site" / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    assert fixed.digest() != env.digest()
    assert (fixed.python, fixed.site, fixed.src, fixed.skills) == (
        env.python,
        env.site,
        env.src,
        env.skills,
    )
