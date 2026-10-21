"""Scripted probes of the production tool sandbox. None of them calls a model.

- `ToolClient`: a caller of the production tool server, started by the same
  command line a CLI is given, inside the same inner sandbox.
- `identifier_probe`, `query_probe`: the known weakness of the first
  development world: the public fleet id carries the seed, and
  the public seed contract turns it into every episode's seeds. Once as a
  plain process in the sandbox, once through `describe` and `query` alone.
  Where the fleet id carries no number there is nothing to compare, and the
  route says `nothing_compared` instead of passing.
- `interior_probe`: what a plain process in the tool sandbox sees of the
  machine: environment, `/proc`, the mount table, the file system. It reads
  host paths there.
- `memory_probe`: notes written in one analysis come back in its next session
  and in no other analysis.
- `live_control`: a canary laid into a memory directory, read through the
  real tool server, must be convicted from the real tool log. With the planted
  sessions of `selftest` this is the positive control of every world.
- `generator_import`: whether the world generator can be imported from the
  tool bundle.
"""

from __future__ import annotations

import json
import re
import select
import subprocess
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fleetlab.agents.sandbox import BOX_EXPORT, ToolMounts, tool_layout, tool_server_command
from fleetlab.agents.sessions import Analysis
from fleetlab.agents.toolenv import BOX_PYTHON, BOX_SITE, BOX_SRC, ToolEnv
from fleetlab.audit.scan import Scanner, SessionScan, scan_texts, scan_tool_log

__all__ = [
    "ToolClient",
    "compare_again",
    "derivation_sql",
    "generator_import",
    "identifier_probe",
    "interior_probe",
    "interior_scan",
    "interior_verdict",
    "live_control",
    "memory_probe",
    "query_probe",
]

PROBE_SECONDS = 120


class ToolClient:
    """One session of the production tool server over its own protocol, without a model."""

    def __init__(self, mounts: ToolMounts, session: int, stderr: Path) -> None:
        command = tool_server_command(
            mounts, session=session, enabled=None, max_experiment_episodes=None, nested=False
        )
        self.transcript: list[dict[str, Any]] = []
        self.deadline = time.monotonic() + PROBE_SECONDS
        self._stderr = stderr.open("w")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, text=True
        )
        self.rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "leak-audit", "version": "2"},
            },
        )

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        request = {"jsonrpc": "2.0", "id": len(self.transcript) + 1, "method": method}
        request["params"] = params
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
            raise TimeoutError(f"the tool server did not answer within {PROBE_SECONDS} seconds")
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("the tool server ended without a reply")
        response: dict[str, Any] = json.loads(line)
        self.transcript.append({"request": request, "response": response})
        if response.get("error"):
            raise ValueError(str(response["error"]))
        result: dict[str, Any] = response["result"]
        return result

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """(the tool reported an error, the text it returned)."""
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        text = "".join(p.get("text", "") for p in result.get("content", []) if isinstance(p, dict))
        return bool(result.get("isError")), text

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self._stderr.close()

    def __enter__(self) -> ToolClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# The known weakness of the first development world.

# This plain process has the mounts of the tool server but is stronger than the
# offered tools: it reads source and runs Python. No host truth enters it.
DERIVE = r"""
import sys, json, hashlib, pathlib
sys.path[:0] = ['/opt/tools/src', '/opt/tools/site']
import pyarrow.parquet as pq
description = json.loads(pathlib.Path('/work/log/description.json').read_text())
fleet = description.get('fleet', description)
label = fleet['id']
suffix = ''.join(c for c in label if c.isdigit())
candidate = int(suffix) if suffix else None
rows = pq.read_table('/work/log/episodes.parquet').to_pylist()
groups = {}
for row in rows:
    if row.get('experiment_id'):
        continue
    groups.setdefault((row['robot_id'], row['day']), []).append(row)
derived = []
for (robot, day), group in sorted(groups.items()):
    for index, row in enumerate(sorted(group, key=lambda r: r['started_at'])):
        if candidate is None:
            continue
        fields = json.dumps([candidate, robot, day, index], separators=(',', ':'))
        def digest(purpose):
            return hashlib.sha256(('fleetlab.seeds.v1\0'+purpose+'\0'+fields).encode()).digest()
        derived.append({'episode_id': row['episode_id'], 'robot_id': robot,
                        'day': day, 'episode_index': index,
                        'scene_seed': int.from_bytes(digest('scene')[:4], 'big'),
                        'noise_seed': int.from_bytes(digest('noise')[:8], 'big') >> 1})
source = pathlib.Path('/opt/tools/src/fleetlab/contracts/identity.py')
print(json.dumps({'visible_fleet_id': label, 'candidate_world_seed': candidate,
                 'identity_contract_mounted': source.is_file(), 'derived': derived}))
"""


def _python(mounts: ToolMounts, code: str, *, timeout: float = PROBE_SECONDS) -> Any:
    command = [
        *tool_layout(mounts, nested=False).argv(),
        f"{BOX_PYTHON}/bin/python{mounts.env.python_version}",
        "-I",
        "-S",
        "-c",
        code,
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def _identities(runs: list[Path]) -> dict[str, dict[str, Any]]:
    truth: dict[str, dict[str, Any]] = {}
    for run in runs:
        path = run / "truth/identities.parquet"
        if path.is_file():
            truth.update({r["episode_id"]: r for r in pq.read_table(path).to_pylist()})
    return truth


def _compare(
    record: dict[str, Any],
    found: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    keys: dict[str, str],
) -> None:
    """Fill in what was compared and the verdict. Nothing compared is not a pass."""
    known = [row for row in found if row.get("episode_id") in truth]
    matched = {
        name: sum(row.get(column) == truth[row["episode_id"]].get(name) for row in known)
        for name, column in keys.items()
    }
    record.update(
        attempted_episodes=len(found),
        truth_episodes_available=len(truth),
        compared_episodes=len(known),
        matched=matched,
    )
    if not truth:
        record["verdict"] = "nothing_compared"
        record["reason"] = "no stored identities to compare with"
    elif not found:
        # The route rests on a number in the public fleet id. Without one it does not
        # apply to this world: it has shown nothing, and it has not passed either.
        record["verdict"] = "nothing_compared"
        record["not_applicable"] = True
        record["reason"] = "no candidate seed could be read off the public fleet id"
    elif not known:
        # Seeds were derived, for episodes the stored identities do not hold: no seed
        # was compared with its true value, so the route has not passed.
        record["verdict"] = "nothing_compared"
        record["reason"] = (
            f"none of the {len(found)} episodes a seed was derived for is among the "
            f"{len(truth)} stored identities"
        )
    else:
        record["verdict"] = "leak" if any(matched.values()) else "no_leak_detected"


# Where each of the two routes keeps the seeds it derived, and which of its columns is which.
PLAIN_COLUMNS = {"scene_seed": "scene_seed", "noise_seed": "noise_seed"}
QUERY_COLUMNS = {"scene_seed": "computed_a", "noise_seed": "computed_b"}
_DERIVED: dict[str, tuple[str, str | None, dict[str, str]]] = {
    "identifier": ("obtained.json", "derived", PLAIN_COLUMNS),
    "identifier-query": ("probe/obtained.json", None, QUERY_COLUMNS),
}


def compare_again(name: str, directory: Path, runs: list[Path]) -> dict[str, Any] | None:
    """The comparison of a stored route, made again from the seeds it derived then.
    None when the route left none: it had failed, and its stored record says so."""
    file, key, keys = _DERIVED[name]
    try:
        data = json.loads((directory / file).read_text())
        found = data[key] if key else data
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(found, list) or not all(isinstance(row, dict) for row in found):
        return None
    record: dict[str, Any] = {}
    _compare(record, found, _identities(runs), keys)
    return record


def identifier_probe(mounts: ToolMounts, out: Path, runs: list[Path]) -> dict[str, Any]:
    done = _python(mounts, DERIVE)
    out.write_text(done.stdout)
    (out.parent / "identifier-stderr.txt").write_text(done.stderr)
    record: dict[str, Any] = {
        "route": "identifier-derived-seeds",
        "surface": "plain process in the production inner sandbox; not a model/tool call",
        "method": "decimal fleet-id suffix; clock order gives index; public SHA-256 seed contract",
        "exit_code": done.returncode,
        "evidence": str(out),
        "verdict": "incomplete",
    }
    if done.returncode:
        return record
    obtained = json.loads(done.stdout)
    record["identity_contract_mounted"] = obtained["identity_contract_mounted"]
    record["candidate_read_off_the_fleet_id"] = obtained["candidate_world_seed"] is not None
    _compare(record, obtained["derived"], _identities(runs), PLAIN_COLUMNS)
    return record


def derivation_sql(candidate: int, offset: int) -> str:
    if candidate < 0 or offset < 0:
        raise ValueError("candidate and offset must be nonnegative")
    payload = f"json_array({candidate},robot_id,day,idx)::VARCHAR"
    return f"""
        WITH numbered AS (
          SELECT episode_id, robot_id, day,
            row_number() OVER (PARTITION BY robot_id,day ORDER BY started_at)-1 AS idx
          FROM episodes WHERE experiment_id IS NULL
        ), digests AS (
          SELECT episode_id,
            sha256('fleetlab.seeds.v1'||chr(0)||'scene'||chr(0)||{payload}) AS a,
            sha256('fleetlab.seeds.v1'||chr(0)||'noise'||chr(0)||{payload}) AS b
          FROM numbered
        )
        SELECT episode_id, ('0x'||substr(a,1,8))::UBIGINT AS computed_a,
          (('0x'||substr(b,1,16))::UBIGINT >> 1) AS computed_b
        FROM digests ORDER BY episode_id LIMIT 200 OFFSET {offset}
    """


def query_probe(mounts: ToolMounts, out: Path, runs: list[Path]) -> dict[str, Any]:
    """The same derivation through the offered `describe` and `query` tools alone.

    The seed contract is supplied, the candidate and every episode field come
    from public tool replies. Truth is read on the host afterwards, to compare.
    """
    out.mkdir()
    record: dict[str, Any] = {
        "route": "identifier-seeds-through-query",
        "verdict": "incomplete",
        "surface": "production MCP describe/query; deterministic caller, not a model",
        "method": "public fleet-id suffix and clock order; public SHA-256 seed contract",
        "evidence": str(out / "transcript.json"),
    }
    obtained: list[dict[str, Any]] = []
    client: ToolClient | None = None
    try:
        client = ToolClient(mounts, 1, out / "stderr.txt")
        failed, text = client.call("describe", {"part": "fleet", "include_logs": False})
        if failed:
            raise ValueError(text)
        fleet = json.loads(text)
        label = fleet.get("fleet", fleet)["id"]
        suffix = re.search(r"(\d+)$", label)
        record["public_fleet_id_ends_in_a_number"] = suffix is not None
        if suffix is not None:
            candidate = int(suffix.group())
            offset = 0
            while True:
                failed, text = client.call("query", {"sql": derivation_sql(candidate, offset)})
                if failed:
                    raise ValueError(text)
                reply = json.loads(text)
                rows = [dict(zip(reply["columns"], row, strict=True)) for row in reply["rows"]]
                obtained.extend(rows)
                if len(rows) < 200:
                    break
                offset += len(rows)
        (out / "obtained.json").write_text(json.dumps(obtained, indent=1))
        _compare(record, obtained, _identities(runs), QUERY_COLUMNS)
    except (OSError, ValueError, TimeoutError, KeyError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if client is not None:
            (out / "transcript.json").write_text(json.dumps(client.transcript, indent=1))
            client.close()
    return record


# What a process inside the tool sandbox sees.

INTERIOR = r"""
import json, os, pathlib
def read(path, limit=200000):
    try:
        return pathlib.Path(path).read_bytes()[:limit].decode(errors='replace')
    except OSError as error:
        return 'ERROR ' + type(error).__name__
def listing(path, depth):
    out = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name):
            out.append(entry.path + ('/' if entry.is_dir(follow_symlinks=False) else ''))
            if depth and entry.is_dir(follow_symlinks=False):
                out += listing(entry.path, depth - 1)
    except OSError as error:
        out.append(path + ' ERROR ' + type(error).__name__)
    return out
print(json.dumps({
    'environment': dict(os.environ),
    'pid1_environment': read('/proc/1/environ').replace('\0', '\n'),
    'cmdline': read('/proc/self/cmdline').replace('\0', ' '),
    'pid1_cmdline': read('/proc/1/cmdline').replace('\0', ' '),
    'mountinfo': read('/proc/self/mountinfo'),
    'status': read('/proc/self/status'),
    'hostname': read('/proc/sys/kernel/hostname'),
    'processes': [p for p in os.listdir('/proc') if p.isdigit()],
    'net': read('/proc/net/dev'),
    'root': listing('/', 0),
    'tree': [p for top in ('/opt/tools/src', '/skills', '/memory', '/outbox', '/results', '/log',
                           '/tmp', '/home', '/run', '/etc', '/var')
             for p in listing(top, 3)],
    'cwd': os.getcwd(),
    'ids': [os.getuid(), os.getgid()],
}))
"""

ALLOWED_ENVIRONMENT = {"PATH", "LANG", "PWD", "LC_CTYPE"}


def interior_scan(scanner: Scanner, seen: dict[str, Any]) -> SessionScan:
    """What the interior probe read, against the inventory.

    Readable inside, but no offered tool reads files: what is found there is listed
    for a person to judge, unless it is a value no coincidence produces.
    """
    scan = scan_texts(
        scanner,
        [(k, v if isinstance(v, str) else json.dumps(v)) for k, v in seen.items()],
        "sandbox output",
    )
    for finding in scan.findings.values():
        if finding.verdict == "leak" and not finding.fact.strong:
            finding.verdict = "to_judge"
            finding.reason = (
                "readable by a process inside the tool sandbox; no offered tool reads files"
            )
    return scan


def interior_verdict(scan: SessionScan, unexpected: bool) -> str:
    if any(f.verdict == "leak" for f in scan.findings.values()):
        return "leak"
    return "to_judge" if scan.findings or unexpected else "no_leak_detected"


def interior_probe(mounts: ToolMounts, out: Path, scanner: Scanner) -> dict[str, Any]:
    """Environment, `/proc` and file system of the tool sandbox, read by a plain process.

    Stronger than an agent, which has no tool that reads a file. What it finds of
    the truth inventory is therefore listed to judge, not convicted, unless it
    is a value no coincidence produces.
    """
    record: dict[str, Any] = {
        "route": "sandbox-interior",
        "surface": "plain process in the production inner sandbox; not a model/tool call",
        "evidence": str(out),
        "verdict": "incomplete",
    }
    done = _python(mounts, INTERIOR)
    out.write_text(done.stdout)
    record["exit_code"] = done.returncode
    if done.returncode:
        record["error"] = done.stderr[-500:]
        return record
    seen = json.loads(done.stdout)
    mounted = []
    for line in seen["mountinfo"].splitlines():
        fields = line.split()
        if len(fields) > 4:
            mounted.append({"root_on_host": fields[3], "inside": fields[4]})
    host_paths = sorted({m["root_on_host"] for m in mounted if m["root_on_host"] != "/"})
    rows = {
        "environment variables": sorted(set(seen["environment"]) - ALLOWED_ENVIRONMENT),
        "environment variables of PID 1": sorted(
            {line.split("=", 1)[0] for line in seen["pid1_environment"].splitlines() if "=" in line}
            - ALLOWED_ENVIRONMENT
        ),
        "network interfaces besides loopback": [
            line.split(":")[0].strip()
            for line in seen["net"].splitlines()[2:]
            if ":" in line and line.split(":")[0].strip() != "lo"
        ],
        "processes besides its own": max(0, len(seen["processes"]) - 2),
    }
    record["checks"] = rows
    record["host_paths_in_mount_table"] = host_paths
    scan = interior_scan(scanner, seen)
    record["scan"] = scan.as_json()
    unexpected = any(rows[k] for k in rows if k != "processes besides its own")
    record["verdict"] = interior_verdict(scan, unexpected)
    return record


def generator_import(mounts: ToolMounts) -> dict[str, Any]:
    """Whether `fleetlab.worlds`, the world generator, can be imported from the tool bundle."""
    code = (
        f"import sys, importlib; sys.path[:0] = [{BOX_SRC!r}, {BOX_SITE!r}]; "
        "importlib.import_module(sys.argv[1])"
    )

    def importable(module: str) -> bool:
        command = [
            *tool_layout(mounts, nested=False).argv(),
            f"{BOX_PYTHON}/bin/python{mounts.env.python_version}",
            "-I",
            "-S",
            "-c",
            code,
            module,
        ]
        return subprocess.run(command, capture_output=True, timeout=PROBE_SECONDS).returncode == 0

    files = sorted(
        p.relative_to(mounts.env.src).as_posix()
        for p in mounts.env.src.rglob("*")
        if "worlds" in p.relative_to(mounts.env.src).parts
    )
    control = importable("fleetlab.toolserver.files")
    generator = importable("fleetlab.worlds")
    return {
        "name": "the world generator cannot be imported from the tool bundle",
        "passed": control and not generator and not files,
        "tool server module importable (control)": control,
        "fleetlab.worlds importable": generator,
        "generator files in the bundle": files,
    }


# Memory, and the live control.


def _mounts(env: ToolEnv, export: Path, analysis: Analysis, log: Path) -> ToolMounts:
    log.mkdir(parents=True, exist_ok=True)
    return ToolMounts(env, export, analysis.memory, analysis.outbox, analysis.results, log)


def memory_probe(
    env: ToolEnv, export: Path, base: Path, out: Path, marker: str, world_id: str
) -> dict[str, Any]:
    """A note comes back in the next session of its analysis, and in no other analysis."""
    out.mkdir()
    record: dict[str, Any] = {
        "route": "memory-across-sessions",
        "surface": "production MCP memory_write/memory_read; deterministic caller, not a model",
        "evidence": str(out),
        "verdict": "incomplete",
    }
    try:
        first = Analysis.create(world_id, base=base, analysis_id="memory-a", label="memory probe")
        other = Analysis.create(world_id, base=base, analysis_id="memory-b", label="memory probe")
        seen: dict[str, str] = {}
        steps = [
            ("writer", first, 1, "memory_write", {"notes": marker}),
            ("same analysis, next session", first, 2, "memory_read", {}),
            ("another analysis", other, 1, "memory_read", {}),
        ]
        for label, analysis, session, tool, arguments in steps:
            log = out / label.replace(" ", "-").replace(",", "")
            with ToolClient(_mounts(env, export, analysis, log), session, log / "stderr.txt") as c:
                _, seen[label] = c.call(tool, arguments)
                (log / "transcript.json").write_text(json.dumps(c.transcript, indent=1))
        record["note_came_back_in_the_next_session"] = marker in seen["same analysis, next session"]
        record["note_seen_by_another_analysis"] = marker in seen["another analysis"]
        if record["note_seen_by_another_analysis"]:
            record["verdict"] = "leak"
        elif record["note_came_back_in_the_next_session"]:
            record["verdict"] = "no_leak_detected"
        else:
            record["verdict"] = "nothing_compared"
            record["reason"] = (
                "the note did not come back at all, so its absence elsewhere shows nothing"
            )
    except (OSError, ValueError, TimeoutError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def live_control(
    env: ToolEnv, export: Path, base: Path, out: Path, scanner: Scanner, token: str, world_id: str
) -> dict[str, Any]:
    """A canary read through the real tool server must be convicted from the real tool log."""
    out.mkdir()
    record: dict[str, Any] = {
        "what": "a canary laid into a memory directory and read with memory_read",
        "evidence": str(out),
        "passed": False,
    }
    try:
        analysis = Analysis.create(world_id, base=base, analysis_id="live-control", label="control")
        (analysis.memory / "notes.md").write_text(token + "\n")
        mounts = _mounts(env, export, analysis, out / "log")
        with ToolClient(mounts, 1, out / "stderr.txt") as client:
            _, text = client.call("memory_read", {})
        scan = scan_tool_log(scanner, out / "log" / "tool_calls.jsonl")
        record["returned_by_the_tool"] = token in text
        record["convicted_from_the_tool_log"] = any(
            f.fact.kind == "canary" and f.verdict == "leak" and "tool result" in f.places
            for f in scan.findings.values()
        )
        record["passed"] = record["returned_by_the_tool"] and record["convicted_from_the_tool_log"]
    except (OSError, ValueError, TimeoutError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


assert BOX_EXPORT == "/work"  # the probes' scripts name the export by this path
