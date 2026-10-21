"""The tool server's environment: interpreter, packages, source and skills, outside the repository.

The inner sandbox mounts four directories read-only, each under a neutral
path, and nothing else of the machine besides `/usr`:

    /opt/tools/python   a copy of the uv-managed CPython the project runs on (relocatable)
    /opt/tools/site     the tool server's third-party packages only, at the project's versions
    /opt/tools/src      the fleetlab modules the tool server imports, and no other
    /skills             the skill library

All four live under the agent root (`cli.agent_root()`, by default
`/var/tmp/robots-agents-<uid>`), so that the sandbox mounts nothing that lies
next to project files. The source bundle is the import closure of
`fleetlab.toolserver`, computed by importing it in a fresh interpreter, so a
module the tool server does not import (`fleetlab.faults`, `fleetlab.sim`,
`fleetlab.log.runfiles`, this package) is not in the sandbox at all. Bundles
and site directories are named by the hash of their content and built once.

The interpreter runs with `-I -S`: no environment variables, no user site, no
`site` module and therefore no `.pth` files, and `sys.path` set explicitly to
the bundle and the site directory (`launcher_argv`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import fleetlab
from fleetlab.agents.cli import agent_root
from fleetlab.contracts.skills import SKILL_FILE

__all__ = [
    "BOX_PYTHON",
    "BOX_SITE",
    "BOX_SRC",
    "FORBIDDEN_MODULES",
    "TOOL_PACKAGES",
    "ToolEnv",
    "build_bundle",
    "build_skills",
    "build_tool_env",
    "import_closure",
    "launcher_argv",
    "project_tool_env",
]

BOX_PYTHON = "/opt/tools/python"
BOX_SITE = "/opt/tools/site"
BOX_SRC = "/opt/tools/src"

# What the tool server imports from outside the standard library; the
# versions are taken from the project environment, so the sandbox runs the
# same code the tests ran.
TOOL_PACKAGES = ("duckdb", "pyarrow", "numpy", "imageio", "imageio-ffmpeg", "pillow", "pydantic")

# Modules that hold truth or reach it, or that have no business in the sandbox.
FORBIDDEN_MODULES = (
    "fleetlab.contracts.identity",
    "fleetlab.contracts.oracle",
    "fleetlab.contracts.world",
    "fleetlab.contracts.rollout",
    "fleetlab.faults",
    "fleetlab.sim",
    "fleetlab.agents",
    "fleetlab.log.runfiles",
    "fleetlab.reference_eval",
    "fleetlab.smoke",
    "fleetlab.runtime",
)

_SRC_ROOT = Path(fleetlab.__file__).resolve().parent.parent  # .../src


@dataclass(frozen=True)
class ToolEnv:
    """Host directories of the tool server's environment, mounted at the BOX_ paths."""

    python: Path  # the interpreter's install directory (bin/python3.12 inside)
    site: Path
    src: Path
    skills: Path | None

    @property
    def python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def digest(self) -> str:
        """Bind the verification to mounted bytes, even if a named bundle was modified."""
        digest = hashlib.sha256()
        for label, root in (
            ("python", self.python),
            ("site", self.site),
            ("source", self.src),
            ("skills", self.skills),
        ):
            digest.update(label.encode())
            if root is None:
                continue
            for path in sorted(root.rglob("*")):
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                if path.is_symlink():
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f"tool environment link leaves its root: {path}")
                    digest.update(os.readlink(path).encode())
                elif path.is_file():
                    with path.open("rb") as stream:
                        digest.update(hashlib.file_digest(stream, "sha256").digest())
        return digest.hexdigest()[:16]


def launcher_argv(python_version: str, module: str = "fleetlab.toolserver") -> list[str]:
    """The command that starts `module` inside the sandbox, from the BOX_ paths."""
    bootstrap = (
        f"import sys; sys.path[:0] = [{BOX_SRC!r}, {BOX_SITE!r}]; "
        f"import runpy; runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
    )
    return [f"{BOX_PYTHON}/bin/python{python_version}", "-I", "-S", "-c", bootstrap]


# The source bundle.


def import_closure(module: str = "fleetlab.toolserver.__main__") -> dict[str, Path]:
    """Every fleetlab module that importing `module` loads, with its source file.

    Computed in a fresh interpreter of the project environment, so that what
    this process has imported does not count.
    """
    code = (
        "import json, sys, importlib; importlib.import_module(sys.argv[1]); "
        "print(json.dumps({n: m.__file__ for n, m in sys.modules.items() "
        "if (n == 'fleetlab' or n.startswith('fleetlab.')) and getattr(m, '__file__', None)}))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, module],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": os.environ.get("HOME", "/")},
    )
    found: dict[str, str] = json.loads(done.stdout.strip().splitlines()[-1])
    return {name: Path(path) for name, path in sorted(found.items())}


def closure_problems(modules: Iterable[str]) -> list[str]:
    """Forbidden modules in an import closure."""
    return [
        f"the tool server imports {name}"
        for name in modules
        for bad in FORBIDDEN_MODULES
        if name == bad or name.startswith(bad + ".")
    ]


def _content_dir(parent: Path, prefix: str, files: dict[str, bytes]) -> Path:
    """A directory named by the hash of `files` (relative path -> bytes), written once."""
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode() + b"\0" + hashlib.sha256(files[relative]).digest())
    target = parent / f"{prefix}-{digest.hexdigest()[:16]}"
    if target.is_dir():
        return target
    staging = parent / f".{prefix}-{uuid.uuid4().hex[:8]}"
    for relative, data in files.items():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o444)
    try:
        staging.rename(target)
    except OSError:  # built meanwhile by another process
        shutil.rmtree(staging, ignore_errors=True)
    return target


def build_bundle(parent: Path) -> Path:
    """The tool server's source, `<parent>/src-<hash>/fleetlab/...`; refuses a forbidden import."""
    closure = import_closure()
    problems = closure_problems(closure)
    if problems:
        raise RuntimeError("; ".join(problems))
    files: dict[str, bytes] = {}
    for path in closure.values():
        relative = path.resolve().relative_to(_SRC_ROOT)
        files[relative.as_posix()] = path.read_bytes()
    return _content_dir(parent, "src", files)


def build_skills(
    parent: Path,
    skills: Path,
    names: Sequence[str] | None = None,
    *,
    ablate_fault_classes: bool = False,
) -> Path:
    """A copy of the skill library (each directory with a SKILL.md) as `<parent>/skills-<hash>/`.

    `names` chooses the skills to mount (for an analysis run, the analysis skills
    and not the live supervisor's: skills/README.md); None mounts all. Helper
    files beside a SKILL.md are copied too: the agent can read them, no tool runs them.
    """
    available = sorted(p for p in skills.iterdir() if (p / SKILL_FILE).is_file())
    if names is not None:
        unknown = sorted(set(names) - {p.name for p in available})
        if unknown:
            raise ValueError(f"no skills {unknown} in {skills}")
        available = [p for p in available if p.name in set(names)]
    files: dict[str, bytes] = {}
    for skill in available:
        for path in sorted(skill.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"{path} is a link; skills are plain files")
            if path.is_file() and "__pycache__" not in path.parts:
                content = path.read_bytes()
                if ablate_fault_classes:
                    from fleetlab.skill_library import strip_fault_class_knowledge

                    content = strip_fault_class_knowledge(content.decode()).encode()
                files[path.relative_to(skills).as_posix()] = content
    return _content_dir(parent, "skills", files)


# The interpreter and the packages.


def _copy_python(parent: Path) -> Path:
    """A copy of the running interpreter's install directory (python-build-standalone)."""
    source = Path(sys.base_prefix)
    target = parent / f"python-{sys.version.split()[0]}"
    if (target / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}").exists():
        return target
    staging = parent / f".python-{uuid.uuid4().hex[:8]}"
    shutil.copytree(source, staging, symlinks=True)
    try:
        staging.rename(target)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _pins() -> list[str]:
    return [f"{name}=={importlib.metadata.version(name)}" for name in TOOL_PACKAGES]


def _uv() -> str:
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        raise RuntimeError("uv is needed to build the tool server's packages")
    return found


def _site(parent: Path, python: Path) -> Path:
    pins = _pins()
    key = hashlib.sha256(("\n".join([sys.version, *pins])).encode()).hexdigest()[:16]
    target = parent / f"site-{key}"
    if target.is_dir():
        return target
    staging = parent / f".site-{uuid.uuid4().hex[:8]}"
    interpreter = python / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    base = [_uv(), "pip", "install", "--quiet", "--target", str(staging), "--python"]
    command = [*base, str(interpreter), *pins]
    done = subprocess.run([*command, "--offline"], capture_output=True, text=True, timeout=600)
    if done.returncode != 0:  # not everything in uv's cache: fetch
        shutil.rmtree(staging, ignore_errors=True)
        subprocess.run(command, capture_output=True, text=True, timeout=600, check=True)
    (staging / "PINS.txt").write_text("\n".join(pins) + "\n")
    try:
        staging.rename(target)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def build_tool_env(
    skills: Path | None = None,
    root: Path | None = None,
    skill_names: Sequence[str] | None = None,
    role: str | None = None,
    ablate_fault_classes: bool = False,
) -> ToolEnv:
    """The tool server's environment under the agent root, built where it does not exist yet.

    `role` ("analysis" or "supervisor") mounts that role's skills from the
    checkout's library (`fleetlab.skill_library`); otherwise `skills` and
    `skill_names` say which.
    """
    if role is not None:
        from fleetlab.skill_library import ROLES, library_dir

        if role not in ROLES:
            raise KeyError(f"no role {role!r}; the roles are {sorted(ROLES)}")
        skills = skills or library_dir()
        skill_names = ROLES[role]
    base = (root or agent_root()) / "toolenv"
    base.mkdir(parents=True, exist_ok=True)
    python = _copy_python(base)
    return ToolEnv(
        python=python,
        site=_site(base, python),
        src=build_bundle(base),
        skills=(
            build_skills(base, skills, skill_names, ablate_fault_classes=ablate_fault_classes)
            if skills is not None
            else None
        ),
    )


def project_tool_env(parent: Path, skills: Path | None = None) -> ToolEnv:
    """For tests: the project's interpreter and site-packages; the bundle and skills under `parent`.

    Nothing is copied or installed; the site-packages directory is mounted as
    it is. `-S` keeps its `.pth` files (the editable install of fleetlab)
    from being read, so fleetlab still comes from the bundle only.
    """
    parent.mkdir(parents=True, exist_ok=True)
    return ToolEnv(
        python=Path(sys.base_prefix),
        site=Path(sysconfig.get_paths()["purelib"]),
        src=build_bundle(parent),
        skills=build_skills(parent, skills) if skills is not None else None,
    )
