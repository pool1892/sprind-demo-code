from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleetlab.film import build, manifest, website
from fleetlab.film.sources import sha256
from fleetlab.film.tests import fixtures


@pytest.fixture
def film(tmp_path: Path) -> Path:
    fleet = fixtures.fleet_run(tmp_path / "fleet")
    chain = fixtures.episode_run(tmp_path / "spike")
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain))
    build.build(m, tmp_path / "film", draft=True)
    return tmp_path / "film"


def test_site_holds_the_film_record_clips_and_artifacts(film: Path, tmp_path: Path) -> None:
    record = tmp_path / "record"
    (record / "figures").mkdir(parents=True)
    (record / "figures" / "a.png").write_bytes(b"png")
    (record / "index.md").write_text(
        "---\nid: x\n---\n# The record\n\nSee [round 1](round-1.md#top).\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        "\n<details>\n<summary>More</summary>\n\n**bold** inside\n\n</details>\n"
    )
    (record / "round-1.md").write_text("# Round 1\n\n![fig](figures/a.png)\n")
    library = tmp_path / "library" / "jobs"
    library.mkdir(parents=True)
    (library / "job-one.json").write_text(
        json.dumps({"name": "job-one", "purpose": "put things away"})
    )
    spec = tmp_path / "site.json"
    spec.write_text(
        json.dumps(
            {
                "format": website.FORMAT,
                "title": "Test site",
                "lede": "A test.",
                "base_url": "https://example.org/demo",
                "film": str(film),
                "record": str(record),
                "artifacts": [{"label": "Jobs", "path": str(library)}],
            }
        )
    )
    out = tmp_path / "site"
    result = website.build_site(spec, out)
    assert not result.errors
    index = (out / "index.html").read_text()
    assert '<source src="film/film.mp4" type="video/mp4">' in index
    assert 'poster="film/poster.jpg"' in index and (out / "film" / "poster.jpg").is_file()
    assert '<meta property="og:image" content="https://example.org/demo/film/poster.jpg">' in index
    assert "<h2>Transcript</h2>" in index
    assert "put things away" in index and 'href="artifacts/jobs/job-one.json" download' in index
    assert "J1/naive/s00" in index and "e1" in index
    clips = sorted(p.name for p in (out / "clips").iterdir())
    assert len(clips) == 3  # the chain, and e1 and e2 of the pair
    page = (out / "record" / "index.html").read_text()
    assert 'href="round-1.html#top"' in page and "<table>" in page
    assert "<strong>bold</strong>" in page  # Markdown inside <details> is rendered
    assert (out / "record" / "figures" / "a.png").is_file()
    assert (out / ".nojekyll").is_file() and (out / "static" / "style.css").is_file()
    listing = json.loads((out / "site-files.json").read_text())
    assert any(f["path"] == "film/film.mp4" for f in listing["files"])
    # a rebuild replaces its own output, but never a git checkout or a foreign folder
    website.build_site(spec, out)
    (out / ".git").mkdir()
    with pytest.raises(ValueError, match="git checkout"):
        website.build_site(spec, out)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        website.build_site(spec, foreign)


def test_minimal_converter() -> None:
    html = website.minimal_markdown(
        "# Title\n\nA *word* and `code` and [a link](b.md).\n\n- one\n- two\n\n```\nx < y\n```\n\n"
        "| h |\n| --- |\n| c |\n\n> quoted\n"
    )
    assert "<h1>Title</h1>" in html
    assert (
        "<em>word</em>" in html
        and "<code>code</code>" in html
        and '<a href="b.md">a link</a>' in html
    )
    assert "<ul>" in html and "<li>two</li>" in html
    assert "<pre><code>x &lt; y</code></pre>" in html
    assert "<th>h</th>" in html and "<td>c</td>" in html
    assert "<blockquote>" in html


def test_the_site_publishes_no_private_reference(film: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    spec = tmp_path / "site.json"
    spec.write_text(json.dumps({"format": website.FORMAT, "film": str(film)}))
    website.build_site(spec, out)
    assert not (out / "film" / "provenance.json").exists()
    assert not (out / "film" / "manifest.json").exists()
    public_film = json.loads((out / "film" / "film.json").read_text())
    assert public_film["format"] == website.PUBLIC_FORMAT
    clips = [c for b in public_film["beats"] for sh in b["shots"] for c in sh["clips"]]
    assert clips and all(c["file"] and c["file"].startswith("clips/") for c in clips)
    assert all(not c["run"].startswith("/") for c in clips)
    assert public_film["private_provenance_sha256"] == sha256(film / "provenance.json")
    for leak in (
        "Made on /home/cs/code/robots.",
        "Tracked in robots-abcd.6.",
        "Published on https://hub.example.ts.net/a/x.",
    ):
        record = tmp_path / "record"
        record.mkdir(exist_ok=True)
        (record / "index.md").write_text(f"# The record\n\n{leak}\n")
        spec.write_text(
            json.dumps({"format": website.FORMAT, "film": str(film), "record": str(record)})
        )
        with pytest.raises(website.SiteRefused):
            website.build_site(spec, tmp_path / "refused")
        assert not (tmp_path / "refused").exists()
    library = tmp_path / "library"
    library.mkdir()
    (library / "job.json").write_text(
        json.dumps({"purpose": "x", "made_in": "/var/tmp/robots-loop/x"})
    )
    spec.write_text(
        json.dumps(
            {
                "format": website.FORMAT,
                "film": str(film),
                "artifacts": [{"label": "Jobs", "path": str(library)}],
            }
        )
    )
    with pytest.raises(website.SiteRefused, match="a local path"):
        website.build_site(spec, tmp_path / "refused")
