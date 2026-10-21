"""A test fixture's tool environment with one fingerprint, read once.

`project_tool_env` mounts the host's shared interpreter and this checkout's
site-packages. Other processes using the same checkout (a run started from it,
another test session) write bytecode caches into them. The reuse comparison reads
the digest once per arm and refuses arms that differ. On aa8731b the full suite
failed with "the arms differ in ['tool_env']". A cache write between the two
readings is the plausible cause: a MuJoCo cache was created in that site-packages
at 23:42:03 while a run from the same checkout started. The exact changed entry
was not identified, because no per-arm listing survived. Production environments (`build_tool_env`)
are dedicated copies and keep hashing every mounted byte, bytecode included; only
these fixtures read their fingerprint once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fleetlab.agents.toolenv import ToolEnv


@dataclass(frozen=True)
class FixedDigestToolEnv(ToolEnv):
    """The same mounts as the wrapped environment; `digest()` is the reading taken at creation."""

    fixed: str = field(default="")

    def digest(self) -> str:
        return self.fixed


def fixed_digest(env: ToolEnv) -> FixedDigestToolEnv:
    return FixedDigestToolEnv(env.python, env.site, env.src, env.skills, fixed=env.digest())
