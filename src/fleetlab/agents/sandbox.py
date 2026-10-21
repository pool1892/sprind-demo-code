"""The two bubblewrap sandboxes of an agent call, and the static check of their command lines.

From the agent-runtime spike, section "Blinding", with its repair
applied (mount files, not the directories that hold them):

    CLI sandbox (outer)                      tool sandbox (inner, started by the CLI)
    /usr, /etc/ssl, /etc/ca-certificates     /usr
    /etc/resolv.conf                         -
    /home/agent (tmpfs)                      -
      .claude/.credentials.json  or          -
      .codex/ (private, per slot) and
      .codex/auth.json: the login file alone,
      read-write, only on the subscription route
    /opt/cli          the CLI's install dir  -
    /run/agent        this call's run dir    -
    /opt/tools/{python,site,src} read-only   the same
    /work             the export, read-only  the same
    /skills           read-only              the same
    /learned          read-only, if given    the same
    /memory, /outbox  read-write             the same
    /results          read-only              the same
    /log              the call log, rw       the same
    network                                  none
    environment from the caller (Popen)      --clearenv: PATH and LANG only

The inner sandbox is entered by the CLI when it starts its tool server, so its
mounts name paths of the outer one. Nothing of the home directory is mounted
but the CLI's install directory and the login file, and no host path appears
inside: every mount point has a neutral name.

`layout_problems` checks a constructed command line against these rules
without running it; `make check` runs it on every layout the runtime builds.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fleetlab.agents.cli import Backend
from fleetlab.agents.toolenv import BOX_PYTHON, BOX_SITE, BOX_SRC, ToolEnv, launcher_argv

__all__ = [
    "BOX_CLI",
    "BOX_EXPORT",
    "BOX_HOME",
    "BOX_LEARNED",
    "BOX_LOG",
    "BOX_MEMORY",
    "BOX_OUTBOX",
    "BOX_RESULTS",
    "BOX_RUN",
    "BOX_SKILLS",
    "Layout",
    "Mount",
    "ToolMounts",
    "cli_layout",
    "dangerous_links",
    "layout_problems",
    "login_file",
    "parse_bwrap",
    "tool_layout",
    "tool_server_command",
]

BOX_HOME = "/home/agent"
BOX_RUN = "/run/agent"
BOX_CLI = "/opt/cli"
BOX_EXPORT = "/work"
BOX_SKILLS = "/skills"
BOX_LEARNED = "/learned"  # a frozen library of learned skills (docs/design/loop.md, section 9)
BOX_MEMORY = "/memory"
BOX_OUTBOX = "/outbox"
BOX_RESULTS = "/results"
BOX_LOG = "/log"
TOOL_LOG_NAME = "tool_calls.jsonl"

HOME = Path.home()
LOGIN_FILES: dict[Backend, Path] = {
    "claude": HOME / ".claude" / ".credentials.json",
    "codex": HOME / ".codex" / "auth.json",
}
LOGIN_STORES = (HOME / ".claude", HOME / ".codex", HOME / ".claude.json", HOME / ".agents")
BOX_LOGIN: dict[Backend, str] = {
    "claude": f"{BOX_HOME}/.claude/.credentials.json",
    "codex": f"{BOX_HOME}/.codex/auth.json",
}

SYSTEM_RO = ("/usr", "/etc/ssl", "/etc/ca-certificates")
ENV_BINARY = "/usr/bin/env"
TOOL_ENV = {"PATH": "/usr/bin", "LANG": "C.UTF-8"}

MountKind = Literal["ro-bind", "bind", "dir", "tmpfs", "symlink", "proc", "dev"]


def login_file(backend: Backend) -> Path:
    return LOGIN_FILES[backend]


@dataclass(frozen=True)
class Mount:
    kind: MountKind
    src: str | None  # host path (or link target for a symlink); None for dir, tmpfs, proc, dev
    dst: str


@dataclass
class Layout:
    """One bwrap invocation: mounts, network, environment, working directory."""

    mounts: list[Mount] = field(default_factory=list)
    network: bool = False
    clearenv: bool = False
    setenv: dict[str, str] = field(default_factory=dict)  # never a secret: it lands in argv
    chdir: str | None = None
    new_session: bool = False
    die_with_parent: bool = True
    # When set, bwrap itself starts through `env -i` with only these variables. bwrap
    # stays the sandbox's PID 1 with its own environment, which `--clearenv` does not
    # touch and which any process inside can read in /proc/1/environ (review finding S4).
    parent_env: dict[str, str] | None = None

    def argv(self) -> list[str]:
        out: list[str] = []
        if self.parent_env is not None:
            out = [ENV_BINARY, "-i", *(f"{k}={v}" for k, v in self.parent_env.items())]
        out += ["bwrap"]
        if self.die_with_parent:
            out.append("--die-with-parent")
        out.append("--unshare-all")
        if self.network:
            out.append("--share-net")
        if self.new_session:
            out.append("--new-session")
        for m in self.mounts:
            if m.kind in ("ro-bind", "bind", "symlink"):
                assert m.src is not None
                out += [f"--{m.kind}", m.src, m.dst]
            else:
                out += [f"--{m.kind}", m.dst]
        if self.chdir:
            out += ["--chdir", self.chdir]
        if self.clearenv:
            out.append("--clearenv")
        for key, value in self.setenv.items():
            out += ["--setenv", key, value]
        return [*out, "--"]


def _base(mounts: list[Mount], system: Sequence[str]) -> None:
    for path in system:
        if os.path.exists(path):
            mounts.append(Mount("ro-bind", path, path))
    for link, target in (("/bin", "usr/bin"), ("/lib", "usr/lib"), ("/lib64", "usr/lib")):
        mounts.append(Mount("symlink", target, link))
    mounts += [
        Mount("proc", None, "/proc"),
        Mount("dev", None, "/dev"),
        Mount("tmpfs", None, "/tmp"),
    ]


@dataclass(frozen=True)
class ToolMounts:
    """Host paths of what the tool server serves (the outer sandbox maps them to BOX_ paths)."""

    env: ToolEnv
    export: Path
    memory: Path
    outbox: Path
    results: Path
    log_dir: Path  # the directory of this call's tool log
    learned: Path | None = None  # the analysis' frozen library of learned skills, if any

    def pairs(self) -> list[tuple[MountKind, Path | None, str]]:
        """(kind, host path, box path) of every mount the tool server needs."""
        return [
            ("ro-bind", self.env.python, BOX_PYTHON),
            ("ro-bind", self.env.site, BOX_SITE),
            ("ro-bind", self.env.src, BOX_SRC),
            ("ro-bind", self.export, BOX_EXPORT),
            ("ro-bind", self.env.skills, BOX_SKILLS),
            ("bind", self.memory, BOX_MEMORY),
            ("bind", self.outbox, BOX_OUTBOX),
            ("ro-bind", self.results, BOX_RESULTS),
            ("bind", self.log_dir, BOX_LOG),
            ("ro-bind", self.learned, BOX_LEARNED),
        ]


def tool_layout(tools: ToolMounts, *, nested: bool = True) -> Layout:
    """The inner sandbox. `nested`: entered from the CLI sandbox, so sources are BOX_ paths;
    otherwise (tests, the static check) sources are the host paths."""
    # PR_SET_PDEATHSIG follows the creating thread, not the whole CLI process.
    # A CLI's temporary spawn thread can exit while its analysis is still live.
    # Nested tools instead die with the outer PID namespace; the gate tests both
    # survival of that thread and cleanup when the outer sandbox exits.
    layout = Layout(
        clearenv=True,
        setenv=dict(TOOL_ENV),
        parent_env=dict(TOOL_ENV),
        die_with_parent=not nested,
    )
    _base(layout.mounts, ["/usr"])
    for kind, host, box in tools.pairs():
        if host is None:
            continue
        layout.mounts.append(Mount(kind, box if nested else str(host), box))
    layout.chdir = BOX_EXPORT
    return layout


def tool_server_command(
    tools: ToolMounts,
    *,
    session: int,
    enabled: Sequence[str] | None,
    max_experiment_episodes: int | None,
    nested: bool = True,
    max_experiment_requests: int | None = None,
    max_episodes_per_arm: int | None = None,
    budget_first_session: int | None = None,
) -> list[str]:
    """The tool server, in its sandbox: what the CLI is told to start."""
    args = [
        "--export", BOX_EXPORT,
        "--memory", BOX_MEMORY,
        "--outbox", BOX_OUTBOX,
        "--results", BOX_RESULTS,
        "--log", f"{BOX_LOG}/{TOOL_LOG_NAME}",
        "--session", str(session),
    ]  # fmt: skip
    if tools.env.skills is not None:
        args += ["--skills", BOX_SKILLS]
    if tools.learned is not None:
        args += ["--learned", BOX_LEARNED]
    if enabled is not None:
        args += ["--tools", ",".join(enabled)]
    if max_experiment_episodes is not None:
        args += ["--max-experiment-episodes", str(max_experiment_episodes)]
    if max_experiment_requests is not None:
        args += ["--max-experiment-requests", str(max_experiment_requests)]
    if max_episodes_per_arm is not None:
        args += ["--max-episodes-per-arm", str(max_episodes_per_arm)]
    if budget_first_session is not None:
        args += ["--budget-first-session", str(budget_first_session)]
    python = launcher_argv(tools.env.python_version)
    return [*tool_layout(tools, nested=nested).argv(), *python, *args]


def cli_layout(
    backend: Backend,
    cli_binary: Path,
    run_dir: Path,
    tools: ToolMounts | None,
    *,
    subscription: bool,
    codex_home: Path | None = None,
) -> Layout:
    """The CLI's sandbox: system files, network, the login file alone, the run directory."""
    layout = Layout(network=True, new_session=True)
    m = layout.mounts
    _base(m, SYSTEM_RO)
    resolv = os.path.realpath("/etc/resolv.conf")
    if os.path.exists(resolv):
        m.append(Mount("ro-bind", resolv, "/etc/resolv.conf"))
    m += [Mount("tmpfs", None, "/home"), Mount("dir", None, BOX_HOME)]
    box_login = BOX_LOGIN[backend]
    if backend == "codex":
        if codex_home is None:
            raise ValueError("a codex call needs its private CODEX_HOME")
        # codex fills its home with caches and databases on first use, so a private
        # one is kept per slot; it never holds the login, which is mounted over it.
        m.append(Mount("bind", str(codex_home), str(Path(box_login).parent)))
    else:
        m.append(Mount("dir", None, str(Path(box_login).parent)))
    if subscription:
        # Read-write, so that a token refresh inside lands in the real file.
        m.append(Mount("bind", str(login_file(backend)), box_login))
    m.append(Mount("ro-bind", str(cli_binary.parent), BOX_CLI))
    m.append(Mount("bind", str(run_dir), BOX_RUN))
    if tools is not None:
        for kind, host, box in tools.pairs():
            if host is not None:
                m.append(Mount(kind, str(host), box))
    layout.chdir = f"{BOX_RUN}/cwd"
    return layout


# Reading a command line back, and checking it.


def parse_bwrap(argv: Sequence[str]) -> tuple[Layout, list[str]]:
    """The layout of a bwrap command line and the command after `--`."""
    layout = Layout(die_with_parent=False)
    i = 0
    if argv and argv[0].endswith("/env"):
        if len(argv) < 2 or argv[1] != "-i":
            raise ValueError("env before bwrap without -i passes the environment on")
        layout.parent_env = {}
        i = 2
        while i < len(argv) and "=" in argv[i] and not argv[i].startswith("-"):
            key, value = argv[i].split("=", 1)
            layout.parent_env[key] = value
            i += 1
    if i < len(argv) and argv[i].endswith("bwrap"):
        i += 1
    two = {"--ro-bind", "--bind", "--symlink", "--setenv"}
    one = {"--dir", "--tmpfs", "--proc", "--dev", "--chdir"}
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            return layout, list(argv[i + 1 :])
        if arg in two:
            a, b = argv[i + 1], argv[i + 2]
            if arg == "--setenv":
                layout.setenv[a] = b
            else:
                layout.mounts.append(Mount(arg[2:], a, b))  # type: ignore[arg-type]
            i += 3
            continue
        if arg in one:
            value = argv[i + 1]
            if arg == "--chdir":
                layout.chdir = value
            else:
                layout.mounts.append(Mount(arg[2:], None, value))  # type: ignore[arg-type]
            i += 2
            continue
        if arg == "--share-net":
            layout.network = True
        elif arg == "--clearenv":
            layout.clearenv = True
        elif arg == "--new-session":
            layout.new_session = True
        elif arg == "--die-with-parent":
            layout.die_with_parent = True
        elif arg != "--unshare-all":
            raise ValueError(f"unexpected bwrap argument {arg!r}")
        i += 1
    raise ValueError("no -- in the bwrap command line")


def _under(path: str, roots: Iterable[Path]) -> Path | None:
    """The first root that `path` lies in, equals or contains (all resolved)."""
    resolved = Path(os.path.realpath(path))
    for root in roots:
        root = Path(os.path.realpath(root))
        if resolved == root or root in resolved.parents or resolved in root.parents:
            return root
    return None


def dangerous_links(root: Path) -> list[Path]:
    """Symbolic links under `root` whose target lies in the home directory, a login store or
    the sandbox home (where a login file is mounted): a program that follows them while it
    cleans or rewrites `root` reaches a login. Links elsewhere, such as codex's
    own `tmp/arg0/*/apply_patch -> /opt/cli/codex`, are not doors."""
    out = []
    home = Path(os.path.realpath(HOME))
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        target = target if target.is_absolute() else path.parent / target
        normal = Path(os.path.normpath(target))
        if (
            normal == home
            or home in normal.parents
            or normal == Path(BOX_HOME)
            or Path(BOX_HOME) in normal.parents
            or _under(str(normal), LOGIN_STORES) is not None
        ):
            out.append(path)
    return out


SECRET_NAMES = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
RW_INNER = {BOX_MEMORY, BOX_OUTBOX, BOX_LOG}
RW_OUTER = RW_INNER | {BOX_RUN}


def layout_problems(
    argv: Sequence[str],
    role: Literal["cli", "tool"],
    *,
    backend: Backend | None = None,
    forbidden_roots: Sequence[Path] = (),
    allowed_home: Sequence[Path] = (),
    env: Mapping[str, str] | None = None,
    export_dst: str | None = BOX_EXPORT,
) -> list[str]:
    """Why a sandbox command line breaks the rules; empty when it keeps them.

    `forbidden_roots`: host directories no mount may lie in, contain or equal
    (the repository, the truth side of a run). `allowed_home`: the only paths
    under the home directory a mount may name (the CLI's install directory,
    and the login file, which is checked separately). `env`: the environment
    the CLI sandbox is started with. The export (mounted at `export_dst`) may
    lie inside a forbidden root or the home directory, for example in the
    repository's `data/`, but not contain one: what it holds is checked by
    `fleetlab.log.export.export_problems`.
    """
    layout, _ = parse_bwrap(argv)
    problems: list[str] = []
    home = Path(os.path.realpath(HOME))
    for m in layout.mounts:
        if m.kind not in ("ro-bind", "bind") or m.src is None:
            continue
        src = os.path.realpath(m.src)
        where = f"{m.kind} {m.src} -> {m.dst}"
        is_login = backend is not None and src == os.path.realpath(login_file(backend))
        if role == "tool" and any(src == os.path.realpath(f) for f in LOGIN_FILES.values()):
            problems.append(f"{where}: a login file in the tool sandbox")
        if role == "tool" and m.dst == BOX_LOGIN.get(backend or "claude"):
            problems.append(f"{where}: a login file in the tool sandbox")
        if src in ("/", "/home", "/etc", "/var", "/run", "/tmp", str(home)):
            problems.append(f"{where}: a whole system or home directory")
        if not is_login:
            store = _under(src, LOGIN_STORES)
            if store is not None:
                problems.append(f"{where}: reaches into the login store {store}")
        elif m.kind != "bind" or role != "cli":
            problems.append(f"{where}: the login file is bound read-write in the CLI sandbox only")
        elif not m.dst.startswith(BOX_HOME + "/"):
            problems.append(f"{where}: the login file belongs in the sandbox home")
        is_export = export_dst is not None and m.dst == export_dst
        hit = _under(src, forbidden_roots)
        if hit is not None:
            contains = Path(src) == hit or Path(src) in hit.parents
            if contains or not is_export:
                problems.append(f"{where}: lies in or contains {hit}")
        in_home = Path(src) == home or home in Path(src).parents
        allowed_here = any(
            Path(src) == Path(os.path.realpath(a)) or Path(os.path.realpath(a)) in Path(src).parents
            for a in allowed_home
        )
        if in_home and not is_login and not is_export and not allowed_here:
            problems.append(f"{where}: part of the home directory that is not allowed")
        if m.kind == "bind" and not is_login:
            allowed = RW_INNER if role == "tool" else RW_OUTER | {f"{BOX_HOME}/.codex"}
            if m.dst not in allowed:
                problems.append(f"{where}: writable, but only {sorted(allowed)} may be")
        if m.kind == "bind" and Path(src).is_dir():
            for path in dangerous_links(Path(src)):
                problems.append(
                    f"{where}: holds a link toward a login {path} -> {os.readlink(path)}"
                )
    if role == "tool":
        if layout.network:
            problems.append("the tool sandbox has network")
        if not layout.clearenv:
            problems.append("the tool sandbox inherits the CLI's environment")
        if layout.parent_env is None or set(layout.parent_env) - set(TOOL_ENV):
            problems.append("bwrap, PID 1 of the tool sandbox, keeps the CLI's environment")
        if set(layout.setenv) - {"PATH", "LANG"}:
            problems.append(
                f"the tool sandbox sets {sorted(set(layout.setenv) - {'PATH', 'LANG'})}"
            )
    elif not layout.die_with_parent:
        problems.append("the outer CLI sandbox must die with its runtime parent")
    for key in [*layout.setenv, *(env or {})]:
        if any(word in key.upper() for word in SECRET_NAMES) and key in layout.setenv:
            problems.append(f"{key} is on the command line, where every process can read it")
    if role == "cli" and env is not None:
        extra = set(env) - _ALLOWED_CLI_ENV
        if extra:
            problems.append(f"the CLI sandbox gets unexpected variables {sorted(extra)}")
    return problems


_ALLOWED_CLI_ENV = {
    "HOME",
    "PATH",
    "LANG",
    "CODEX_HOME",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
    "DISABLE_AUTOUPDATER",
    "MAX_THINKING_TOKENS",
    # The API-key route (never stored, passed through the process environment).
    "ANTHROPIC_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    # The wire check points the CLI at a local recorder.
    "ANTHROPIC_BASE_URL",
}
