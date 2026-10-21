"""Publishing page bundles with the `hub` CLI, and the manifest that keeps their artifact ids.

`dashboard/ids.json` maps each page to its hub artifact: the progress page
under "progress", each run page under "runs" with the run directory it was
built from, each world page under "worlds" with its world root. A page that
has an id is republished in place (`hub publish --id`); a new one gets an id
from the hub, which is written to the manifest at once, so an interrupted
publish keeps what it got. The manifest belongs in the repository: commit it
after a publish.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["HubError", "Manifest", "PageEntry", "check_bundle", "hub_origin", "publish_bundle"]

PROJECT = "robots"
TAG = "dashboard"
ENTRY = "index.md"


class HubError(RuntimeError):
    """The hub CLI refused or failed."""


@dataclass
class PageEntry:
    id: str
    url: str = ""
    run_dir: str = ""  # run pages: the directory the page was last built from
    world: str = ""
    root: str = ""  # world pages: the world root the page was last built from

    def to_json(self) -> dict[str, str]:
        out = {"id": self.id, "url": self.url}
        if self.run_dir:
            out["run_dir"] = self.run_dir
        if self.world:
            out["world"] = self.world
        if self.root:
            out["root"] = self.root
        return out


@dataclass
class Manifest:
    """The artifact ids of the dashboard's pages."""

    path: Path | None
    progress: PageEntry | None = None
    runs: dict[str, PageEntry] = field(default_factory=dict)
    worlds: dict[str, PageEntry] = field(default_factory=dict)  # world pages by world id

    @staticmethod
    def load(path: Path | None) -> Manifest:
        if path is None or not path.is_file():
            return Manifest(path)
        data = json.loads(path.read_text())
        progress = data.get("progress")
        return Manifest(
            path=path,
            progress=PageEntry(**progress) if progress else None,
            runs={k: PageEntry(**v) for k, v in data.get("runs", {}).items()},
            worlds={k: PageEntry(**v) for k, v in data.get("worlds", {}).items()},
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.progress is not None:
            data["progress"] = self.progress.to_json()
        data["runs"] = {k: self.runs[k].to_json() for k in sorted(self.runs)}
        if self.worlds:
            data["worlds"] = {k: self.worlds[k].to_json() for k in sorted(self.worlds)}
        return data

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(self.path)

    def run_url(self, run_id: str) -> str:
        entry = self.runs.get(run_id)
        return entry.url if entry else ""

    def world_url(self, world_id: str) -> str:
        entry = self.worlds.get(world_id)
        return entry.url if entry else ""

    def progress_url(self) -> str:
        return self.progress.url if self.progress else ""


def _hub(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["hub", *arguments], capture_output=True, text=True, timeout=600, check=False
        )
    except OSError as error:
        raise HubError(f"the hub CLI could not be run: {error}") from error


def check_bundle(bundle: Path) -> list[str]:
    """`hub check` on a page bundle; raises on content errors, returns the warnings."""
    done = _hub(["check", "--project", PROJECT, "--entry", ENTRY, "--json", str(bundle)])
    if done.returncode == 2:
        raise HubError(f"hub check {bundle}: {done.stderr.strip() or done.stdout.strip()}")
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise HubError(f"hub check {bundle} gave no JSON: {done.stdout[:200]!r}") from error
    findings = [
        f"{f.get('severity', '?')} {f.get('kind', '')} line {f.get('line', '?')}: "
        f"{f.get('message', '')}"
        for f in report.get("findings", [])
    ]
    skipped = [line for line in done.stderr.splitlines() if line.startswith("hub: warning")]
    if done.returncode != 0:
        raise HubError(f"hub check {bundle} found errors: " + "; ".join(findings))
    return skipped + findings


def _field(data: dict[str, Any], *names: str) -> str:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value:
            return value
    nested = data.get("artifact")
    if isinstance(nested, dict):
        return _field(nested, *names)
    return ""


def publish_bundle(
    bundle: Path,
    title: str,
    artifact_id: str | None,
    tags: tuple[str, ...] = (),
    origin: str = "",
) -> PageEntry:
    """Publish a page bundle (its `index.md` the entry) and return its id and URL.

    `origin` (from `hub_origin`, looked up before publishing) makes the URL the
    hub answers with, a path, absolute.
    """
    arguments = ["publish", "--project", PROJECT, "--entry", ENTRY, "--title", title]
    for tag in (TAG, *tags):
        arguments += ["--tag", tag]
    if artifact_id:
        arguments += ["--id", artifact_id]
    done = _hub([*arguments, "--json", str(bundle)])
    if done.returncode != 0:
        raise HubError(f"hub publish {bundle}: {done.stderr.strip() or done.stdout.strip()}")
    try:
        data = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise HubError(f"hub publish {bundle} gave no JSON: {done.stdout[:200]!r}") from error
    new_id = _field(data, "id", "artifact_id")
    if not new_id:
        raise HubError(f"hub publish {bundle}: no artifact id in {sorted(data)}")
    if artifact_id and new_id != artifact_id:
        raise HubError(f"hub publish {bundle}: asked for {artifact_id}, got {new_id}")
    url = _field(data, "url", "page_url")
    return PageEntry(id=new_id, url=origin + url if url.startswith("/") else url)


def hub_origin() -> str:
    """The hub's address as `hub config` reports it (that command redacts the token); "" if none."""
    try:
        done = _hub(["config"])
    except HubError:
        return ""
    for line in done.stdout.splitlines():
        if line.startswith("url:"):
            parts = line.split()
            if len(parts) > 1 and parts[1].startswith("http"):
                return parts[1].rstrip("/")
    return ""
