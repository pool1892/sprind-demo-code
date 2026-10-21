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


def test_the_captions_note_follows_the_site_description(film: Path, tmp_path: Path) -> None:
    # burned-in captions, as the earlier cuts had, unless the description says otherwise
    spec = tmp_path / "site.json"
    spec.write_text(json.dumps({"format": website.FORMAT, "film": str(film)}))
    website.build_site(spec, tmp_path / "default")
    assert "captions in the picture" in (tmp_path / "default" / "index.html").read_text()
    spec.write_text(
        json.dumps(
            {"format": website.FORMAT, "film": str(film), "captions_note": "English captions (CC)"}
        )
    )
    website.build_site(spec, tmp_path / "track")
    index = (tmp_path / "track" / "index.html").read_text()
    assert "English captions (CC)" in index and "captions in the picture" not in index


def test_a_story_bundle_builds_the_story_page_and_moves_the_sources(
    film: Path, tmp_path: Path
) -> None:
    import re

    from fleetlab.film.tests.test_story import STORY, film_data

    (tmp_path / "story.md").write_text(STORY)
    (tmp_path / "film-data.json").write_text(json.dumps(film_data()))
    record = tmp_path / "record"
    record.mkdir()
    (record / "index.md").write_text("# The record\n\nCounts.\n")
    spec = tmp_path / "site.json"
    spec.write_text(
        json.dumps(
            {
                "format": website.FORMAT,
                "story": {"text": "story.md", "data": "film-data.json"},
                "film": str(film),
                "record": str(record),
                "public": {
                    "aliases": {"w80": "w08"},
                    "forbid": ["controller_overload", "(?<![A-Za-z0-9])w8[0-9](?![0-9])"],
                },
            }
        )
    )
    out = tmp_path / "site"
    result = website.build_site(spec, out)
    assert not result.errors
    index = (out / "index.html").read_text()
    assert "<h1>A test story</h1>" in index and '<main id="story"' in index
    assert '<a href="#first">First</a>' in index  # the contents line uses the short name
    assert 'content="The answer: 3 of 3, from 1."' in index  # the preview is the answer
    assert '<source src="film/film.mp4" type="video/mp4">' in index
    assert "{" not in re.sub(r"<script>.*?</script>", "", index, flags=re.S)
    sources = (out / "sources.html").read_text()
    assert "J1/naive/s00" in sources and '<h2 id="transcript-heading">' in sources
    assert 'href="../sources.html"' in (out / "record" / "index.html").read_text()
    values = json.loads((out / "story-values.json").read_text())
    assert values["values"]["r1_delivered"]["value"] == "3"
    assert values["charts"] and values["fleet_record"]["evaluation"]
    # the fleet record, under the site's public aliases: the private world id is renamed
    fleet = (out / "fleet-record.html").read_text()
    assert "The fleet record" in fleet and "probes:fleet/w08-x/experiments.jsonl" in fleet
    assert "w80" not in fleet + json.dumps(values)
    assert 'href="fleet-record.html"' in sources
    # every relative link of the two pages lands on a file of the site
    for page in ("index.html", "sources.html", "fleet-record.html"):
        for url in re.findall(r'(?:href|src)="(?![a-z]+:)([^"#]+)', (out / page).read_text()):
            assert (out / url).exists(), f"{page}: {url}"
    # a story that would publish a forbidden name is refused like any other page
    (tmp_path / "story.md").write_text(STORY + "\nThe controller_overload fault.\n")
    with pytest.raises(website.SiteRefused):
        website.build_site(spec, tmp_path / "refused")
