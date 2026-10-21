from __future__ import annotations

import socket
from pathlib import Path

from fleetlab.film import public


def test_public_refs_name_no_local_path() -> None:
    probes = Path.home() / ".cache" / "robots-probes" / "demo" / "spike" / "naive10"
    assert public.public_ref(probes) == "probes:demo/spike/naive10"
    assert public.public_ref("/var/tmp/robots-loop/driver/m2/log") == "loop:driver/m2/log"
    assert (
        public.public_ref("/home/someone/code/robots-wt-instrument/spikes/fig.png")
        == "repo:spikes/fig.png"
    )
    other = public.public_ref("/home/someone/elsewhere/clip.mp4")
    assert other.startswith("file:clip.mp4#") and "/home" not in other
    assert public.short_ref(probes) == "probes:spike/naive10"
    for ref in (public.public_ref(probes), other):
        assert not public.scan(ref)


def test_scan_finds_what_must_not_be_published() -> None:
    host = socket.gethostname().split(".")[0]
    text = "\n".join(
        [
            "a clip at /home/cs/.cache/robots-probes/x.mp4",
            "a run under /var/tmp/robots-loop/driver",
            "see ~/.cache/robots-probes/demo",
            "published at https://hub.example.ts.net/a/abc",
            "bead robots-abcd.6 tracks it",
            "mail someone@example.org",
            f"served from {host}.example.lan",
        ]
    )
    found = {h.what for h in public.scan(text, "page.html")}
    assert found >= {
        "a home directory",
        "a local path",
        "a home-relative path",
        "the tailnet",
        "the review hub",
        "a tracker id",
        "an e-mail address",
        "this machine's host name",
    }
    assert [h.what for h in public.scan("the tailnet tail1a2b3c")] == ["the tailnet"]
    clean = "The robot boy enjoys probes:demo/spike and loop:driver; a shift lead approves."
    assert public.scan(clean) == []


def test_a_cut_aliases_private_names_and_forbids_them() -> None:

    run = Path.home() / ".cache" / "robots-probes" / "x" / "w80-20261019T224617-a1f33d" / "s-02"
    public.configure({"w80": "w08"}, [r"(?<![A-Za-z0-9])w8[0-9](?![0-9])"])
    assert public.public_ref(run) == "probes:x/w08-20261019T224617-a1f33d/s-02"
    assert public.short_ref(run) == "probes:w08-20261019T224617-a1f33d/s-02"
    assert public.forbidden("fleet w80 and w81, not w08 or w800") == ["w80", "w81"]
    assert [h.what for h in public.scan("world w80")] == [
        "a name this cut forbids ((?<![A-Za-z0-9])w8[0-9](?![0-9]))"
    ]
    public.configure()
    assert "w80" in public.public_ref(run) and not public.forbidden("w80")


def test_the_film_refuses_to_show_a_forbidden_name(tmp_path: Path) -> None:
    import json

    import pytest

    from fleetlab.film import build, manifest
    from fleetlab.film.tests import fixtures

    fleet, chain = fixtures.fleet_run(tmp_path / "fleet"), fixtures.episode_run(tmp_path / "spike")
    beats = [{"id": "t", "title": "Fleet w80", "shots": [{"kind": "card", "title": "x"}]}]
    path = fixtures.manifest(tmp_path / "m.json", fleet, chain, beats=beats)
    data = json.loads(path.read_text())
    data["public"] = {"forbid": ["w8[0-9]"]}
    path.write_text(json.dumps(data))
    with pytest.raises(manifest.ManifestError, match="forbids"):
        build.plan(manifest.load(path))
    data["public"] = {"aliases": {"w80": "w08"}, "forbid": ["w8["]}
    path.write_text(json.dumps(data))
    with pytest.raises(manifest.ManifestError):
        manifest.load(path)
