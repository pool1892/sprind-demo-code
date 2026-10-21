"""tools/film-narrate and tools/film-check-public, end to end against local servers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[4]
NARRATE = ROOT / "tools" / "film-narrate"
CHECK = ROOT / "tools" / "film-check-public"
KEY = "stub-key-not-real-3c9d"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ELEVENLABS_API_KEY"}
    env["FILM_PYTHON"] = f"{sys.executable}"
    env["FILM_NARRATE_RETRIES"] = "0"
    env.update(extra)
    return env


PLAIN_SCRIPT = """# A script without numbers

## open: Opening

This is a simulated robot installation. Its policy is fixed.

## a0-title: Story A

First story: make this one job.
"""


def _plain(tmp_path: Path) -> list[str]:
    path = tmp_path / "plain-script.md"
    path.write_text(PLAIN_SCRIPT)
    return ["--script", str(path)]


def _tts_requests(tmp_path: Path) -> list[dict[str, object]]:
    log = tmp_path / "log.jsonl"
    if not log.is_file():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [r for r in rows if "text-to-speech" in str(r.get("path"))]


def test_narrate_without_key_writes_requests_and_captions(tmp_path: Path) -> None:
    out = tmp_path / "voices"
    done = subprocess.run(
        [str(NARRATE), "voices", "--out", str(out)], env=_env(), capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "ELEVENLABS_API_KEY is not set" in done.stderr
    assert (out / "passage" / "voice-test.vtt").read_text().startswith("WEBVTT")
    assert json.loads((out / "passage" / "voice-test.request-1.json").read_text())["text"]
    assert "POST https://api.elevenlabs.io/v1/text-to-speech/" in (out / "dry-run.txt").read_text()
    out = tmp_path / "narration"
    done = subprocess.run(
        [str(NARRATE), "render", "--out", str(out)], env=_env(), capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "wrote caption files instead" in done.stderr
    assert (out / "captions.vtt").is_file() and not list(out.glob("*.mp3"))


@pytest.fixture
def stub(tmp_path: Path) -> Iterator[str]:
    port_file, log = tmp_path / "port", tmp_path / "log.jsonl"
    server = subprocess.Popen(
        [sys.executable, "-m", "fleetlab.film.tests.elevenlabs_stub", str(port_file), str(log)],
        env={**os.environ, "STUB_KEY": KEY},
    )
    try:
        for _ in range(100):
            if port_file.is_file() and port_file.read_text():
                break
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port_file.read_text()}"
    finally:
        server.terminate()
        server.wait()


def test_narrate_with_key_and_fallback(stub: str, tmp_path: Path) -> None:
    env = _env(ELEVENLABS_API_KEY=KEY, ELEVENLABS_API_BASE=stub)
    voices = tmp_path / "voices"
    done = subprocess.run(
        [str(NARRATE), "voices", "--out", str(voices)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in voices.glob("*.mp3")) == ["adam-m-adam.mp3", "brian-m-brian.mp3"]
    assert "model\televen_v4" in (voices / "choice.tsv").read_text()
    out = tmp_path / "narration"
    done = subprocess.run(
        [
            str(NARRATE),
            *("render", "--voice", "m-brian", "--model", "eleven_v4", "--out", str(out)),
            *_plain(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "refused as invalid" in done.stderr  # v4 refused previous_text; the plainer body passed
    texts = sorted(p.stem for p in out.glob("*.txt"))
    assert sorted(p.stem for p in out.glob("*.mp3")) == texts
    record = json.loads((out / "voice.json").read_text())
    assert record["voice_id"] == "m-brian" and sorted(record["beats"]) == texts
    assert all(len(b["text_sha256"]) == 64 for b in record["beats"].values())
    assert json.loads((out / "narration.json").read_text())["timing"].startswith("measured")
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix != ".mp3":
            assert KEY not in path.read_text(errors="replace"), path
    assert KEY not in done.stdout + done.stderr


MARK = '<meta name="generator" content="fleetlab.film-site/1">'
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16 + b"\x00\x00\x00\x08moov" + b"\x00" * 1000


def _page(src: str, mark: bool = True) -> bytes:
    head = MARK if mark else ""
    video = f'<video controls><source src="{src}" type="video/mp4"></video>'
    return f"<html><head>{head}</head><body>{video}</body></html>".encode()


class _Site(BaseHTTPRequestHandler):
    """Pages and films as public hosts and their failures serve them."""

    routes: ClassVar[dict[str, tuple[int, str, bytes, dict[str, str]]]] = {
        "/good/": (200, "text/html; charset=utf-8", _page("film/film.mp4"), {}),
        "/good/film/film.mp4": (200, "video/mp4", MP4, {}),
        "/to-login/": (302, "text/html", b"", {"Location": "/login?return_to=/to-login/"}),
        "/login": (200, "text/html", _page("film.mp4", mark=False), {}),
        "/soft-error/": (
            200,
            "text/html",
            b"<h1>Site not found</h1>" + _page("film.mp4", mark=False),
            {},
        ),
        "/hard-error/": (404, "text/html", _page("film.mp4"), {}),
        "/head-only/": (200, "text/html", _page("film.mp4"), {}),
        "/fake/": (200, "text/html", _page("film.mp4"), {}),
        "/fake/film.mp4": (200, "video/mp4", b"<html>an error page</html>", {}),
        "/big/": (200, "text/html", _page("film.mp4"), {}),
        "/big/film.mp4": (200, "video/mp4", MP4 + b"\x00" * 5000, {}),
    }

    def _answer(self, with_body: bool) -> None:
        if self.path == "/head-only/film.mp4":  # HEAD says yes, GET says no
            code, kind, body = (
                (200, "video/mp4", MP4) if not with_body else (404, "text/html", b"gone")
            )
            extra: dict[str, str] = {}
        else:
            route = self.path.split("?", 1)[0]
            code, kind, body, extra = self.routes.get(route, (404, "text/html", b"no", {}))
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._answer(True)

    def do_HEAD(self) -> None:
        self._answer(False)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def hosts() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _check(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CHECK), *args], capture_output=True, text=True, env={**os.environ, **env}
    )


def test_check_public_fails_closed(hosts: str) -> None:
    good = _check(f"{hosts}/good/")
    assert good.returncode == 0, good.stdout
    assert "verdict: PUBLIC" in good.stdout and "video/mp4" in good.stdout
    cases = {
        "login": (_check(f"{hosts}/to-login/"), "a login page"),
        "soft error": (_check(f"{hosts}/soft-error/"), "its mark is missing"),
        "hard error": (_check(f"{hosts}/hard-error/"), "page   404"),
        "head only": (_check(f"{hosts}/head-only/"), "film   404"),
        "fake film": (_check(f"{hosts}/fake/"), "not an MP4 file"),
        "too big": (_check(f"{hosts}/big/", FILM_CHECK_MAX_BYTES="2000"), "over the limit"),
        "--mp4 still checks the page": (
            _check(f"{hosts}/soft-error/", "--mp4", f"{hosts}/good/film/film.mp4"),
            "its mark is missing",
        ),
        "--mp4 head only": (
            _check(f"{hosts}/good/", "--mp4", f"{hosts}/head-only/film.mp4"),
            "film   404",
        ),
    }
    for name, (done, reason) in cases.items():
        assert done.returncode == 1, (name, done.stdout)
        assert "NOT PUBLIC" in done.stdout and reason in done.stdout, (name, done.stdout)
    assert _check("not-a-url").returncode == 2


def test_check_public_passes_a_built_site(tmp_path: Path) -> None:
    from fleetlab.film import build, manifest, website
    from fleetlab.film.tests import fixtures

    fleet = fixtures.fleet_run(tmp_path / "fleet")
    chain = fixtures.episode_run(tmp_path / "spike")
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain))
    build.build(m, tmp_path / "film", draft=True)
    spec = tmp_path / "site.json"
    spec.write_text(json.dumps({"format": website.FORMAT, "film": str(tmp_path / "film")}))
    website.build_site(spec, tmp_path / "site")
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path / "site"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        done = _check(f"http://127.0.0.1:{server.server_address[1]}/")
    finally:
        server.shutdown()
    assert done.returncode == 0, done.stdout


def _no_key_anywhere(root: Path, *outputs: str) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes(), path
    for output in outputs:
        assert KEY not in output


def test_a_server_that_reflects_the_key_gets_it_nowhere(stub: str, tmp_path: Path) -> None:
    (tmp_path / "log.jsonl.mode").write_text("reflect")
    env = _env(ELEVENLABS_API_KEY=KEY, ELEVENLABS_API_BASE=stub)
    voices = tmp_path / "voices"
    done = subprocess.run(
        [str(NARRATE), "voices", "--out", str(voices)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 1  # /v1/models answered 500, with the key in its body
    assert "HTTP 500" in done.stderr and "[key removed]" in done.stderr
    listing = (voices / "listing" / "voices.json").read_text()
    assert "seen with key [key removed]" in listing
    out = tmp_path / "narration"
    done_render = subprocess.run(
        [
            str(NARRATE),
            *("render", "--voice", "m-brian", "--model", "eleven_v4", "--out", str(out)),
            *_plain(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done_render.returncode == 0, done_render.stderr
    assert "for key [key removed]" in done_render.stderr  # the 422 refusals, reflected
    _no_key_anywhere(tmp_path, done.stdout, done.stderr, done_render.stdout, done_render.stderr)


def test_a_failed_recording_leaves_no_old_audio(stub: str, tmp_path: Path) -> None:
    env = _env(ELEVENLABS_API_KEY=KEY, ELEVENLABS_API_BASE=stub)
    out = tmp_path / "narration"
    command = [str(NARRATE), "render", "--voice", "m-adam", "--out", str(out), *_plain(tmp_path)]
    assert subprocess.run(command, env=env, capture_output=True, text=True).returncode == 0
    assert list(out.glob("*.mp3")) and (out / "voice.json").is_file()
    (tmp_path / "log.jsonl.mode").write_text("fail")
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    assert done.returncode == 1 and "no recording of it is kept" in done.stderr
    assert not list(out.glob("*.mp3")) and not (out / "voice.json").exists()


def test_unbound_numbers_are_never_sent_to_the_voice(stub: str, tmp_path: Path) -> None:
    env = _env(ELEVENLABS_API_KEY=KEY, ELEVENLABS_API_BASE=stub)
    out = tmp_path / "narration"
    done = subprocess.run(
        [str(NARRATE), "render", "--voice", "m-brian", "--out", str(out)],  # the repository's
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 4, done.stderr
    assert "bound to no record; nothing was sent" in done.stderr
    assert not _tts_requests(tmp_path) and not list(out.glob("*.mp3"))


def test_render_binds_the_numbers_first_and_uses_the_default_voice(
    stub: str, tmp_path: Path
) -> None:
    script = tmp_path / "script.md"
    script.write_text(
        "# S\n\n## a2-request: The request\n\nThe bar is {acceptance_k} of {acceptance_n}.\n"
    )
    (tmp_path / "job_run.json").write_text(json.dumps({"acceptance": {"N": 20, "K": 18}}))
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps({"format": "fleetlab.film-records/1", "records": {"job_three": "job_run.json"}})
    )
    out = tmp_path / "narration"
    command = [str(NARRATE), "render", "--script", str(script), "--records", str(records)]
    done = subprocess.run([*command, "--out", str(out)], env=_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr  # without the key: captions from the bound script
    assert "The bar is 18 of 20." in (out / "captions.vtt").read_text().replace("\n", " ")
    assert "/text-to-speech/nPczCjzI2devNBz1zQrb?" in (out / "dry-run.txt").read_text()
    env = _env(ELEVENLABS_API_KEY=KEY, ELEVENLABS_API_BASE=stub)
    done = subprocess.run(
        [*command, "--voice", "m-brian", "--out", str(out)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    sent = _tts_requests(tmp_path)
    assert sent and all("m-brian" in str(r["path"]) for r in sent)
    bindings = json.loads((out / "bindings.json").read_text())
    assert bindings["values"]["acceptance_k"]["value"] == 18 and not bindings["unbound"]
    record = json.loads((out / "voice.json").read_text())
    assert sorted(record["beats"]) == ["a2-request"]
