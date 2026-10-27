"""The public site: one static page with the film, the record, the clips and the artifacts.

`python -m fleetlab.film site BUNDLE --out DIR` reads a site description (`site.json`,
format `fleetlab.film-site/1`, or a folder holding one) and writes plain HTML and CSS, no
framework and no build step on the server:

    {"format": "fleetlab.film-site/1",
     "title": ..., "lede": "one paragraph under the title",
     "film": "<the output folder of python -m fleetlab.film build>",
     "record": "<a folder of Markdown pages with index.md, or one .md file>",
     "artifacts": [{"label": "Jobs", "path": "<a folder or a file>"}, ...],
     "base_url": "https://pool1892.github.io/<repo>/",   # optional: absolute link previews
     "public": {"aliases": {"w80": "w08"}, "forbid": ["w8[0-9]"]},  # optional, see public.py
     "story": {"text": "<story.md>", "data": "<film-data.json>"}}  # optional, see story.py

Paths are relative to the description's folder. The output:

- `index.html`: the film in a `<video>` element (H.264 and AAC, `film/film.mp4`, its
  poster, captions and chapter tracks), the beats with their start times, the clips the
  film shows (each the original recorded video, a verified re-render beside it where one
  is used), the artifacts as downloads, and the provenance files;
- `record/`: the record's Markdown pages as HTML (Python-Markdown from the environment,
  or a small converter of this module where it is missing), with their figures and files;
- `film/`, `clips/`, `artifacts/`, `static/style.css`, `.nojekyll` (GitHub Pages serves
  the folder as it is) and `site-files.json`, every file with its size and SHA-256.

With a `story`, `index.html` is the story page instead: the narrative with its numbers filled
from the records and its charts drawn from them (`fleetlab.film.story`, `fleetlab.film.charts`),
the film at its centre; the chapters, transcript, clips, artifacts and provenance move to
`sources.html`, and `story-values.json` lists every number the page used with its source.

A file over 100 MB fails the build (GitHub Pages refuses it); over 50 MB warns.
"""

from __future__ import annotations

import html
import importlib
import json
import re
import shutil
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.film.public import Hit, public_ref, scan_tree
from fleetlab.film.sources import sha256

__all__ = ["PUBLIC_FORMAT", "SiteRefused", "SiteResult", "build_site", "markdown_to_html"]

PUBLIC_FORMAT = "fleetlab.film-public/1"


class SiteRefused(ValueError):
    """The built site would publish something private; nothing is left to deploy."""

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        shown = "\n  ".join(str(h) for h in hits[:25])
        more = f"\n  ... and {len(hits) - 25} more" if len(hits) > 25 else ""
        super().__init__(f"the site would publish private references:\n  {shown}{more}")


FORMAT = "fleetlab.film-site/1"
SITE_ROOT = Path(__file__).resolve().parents[3] / "site"
LIMIT = 100 * 10**6
WARN = 50 * 10**6
CLIP_LIMIT = 25 * 10**6


@dataclass
class SiteResult:
    out: Path
    files: int
    bytes: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---- Markdown --------------------------------------------------------------------------------


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end() :]


def _python_markdown() -> Callable[[str], str] | None:
    try:
        module: Any = importlib.import_module("markdown")
    except ImportError:
        return None
    extensions = ["tables", "fenced_code", "md_in_html", "sane_lists", "attr_list", "toc"]

    def convert(text: str) -> str:
        text = re.sub(r"<details>", '<details markdown="1">', text)
        return str(module.markdown(text, extensions=extensions, output_format="html"))

    return convert


_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)"), r'<img alt="\1" src="\2">'),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"), r'<a href="\2">\1</a>'),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])"), r"<em>\1</em>"),
]


def _inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        # raw HTML in the source (anchors, details) is kept as written
        piece = part
        for pattern, replacement in _INLINE:
            piece = pattern.sub(replacement, piece)
        out.append(piece)
    return "".join(out)


def minimal_markdown(text: str) -> str:
    """Headings, paragraphs, lists, block quotes, fenced code, pipe tables, inline code,
    emphasis, links and images: enough for the record when Python-Markdown is missing."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^(`{3,})(.*)$", line)
        if fence:
            mark, body = fence.group(1), []
            i += 1
            while i < len(lines) and not lines[i].startswith(mark):
                body.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(body))}</code></pre>")
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[i + 1])
        ):
            head = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{_inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
            )
            out += ["<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows]
            out.append("</tbody></table>")
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>")
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                out.append(f"<li>{_inline(item)}</li>")
            out.append(f"</{tag}>")
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> "))
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(quote))}</p></blockquote>")
            continue
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("<"):
            out.append(line)
            i += 1
            continue
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#|>|\||`{3}|\s*([-*]|\d+\.)\s)", lines[i])
        ):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


def markdown_to_html(text: str) -> tuple[dict[str, str], str, str]:
    """Front matter, the page's HTML body, and which converter made it."""
    meta, body = _front_matter(text)
    convert = _python_markdown()
    if convert is not None:
        return meta, convert(body), "Python-Markdown"
    return meta, minimal_markdown(body), "fleetlab.film.website minimal converter"


def _md_links_to_html(body: str) -> str:
    return re.sub(
        r'href="([^":#?]+)\.md(#[^"]*)?"',
        lambda m: f'href="{m.group(1)}.html{m.group(2) or ""}"',
        body,
    )


# ---- the page ------------------------------------------------------------------------------


def _template(name: str) -> string.Template:
    return string.Template((SITE_ROOT / "templates" / name).read_text(encoding="utf-8"))


def _human(size: int) -> str:
    if size >= 10**6:
        return f"{size / 10**6:.1f} MB"
    if size >= 10**3:
        return f"{size / 10**3:.0f} kB"
    return f"{size} B"


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _load(bundle: Path) -> tuple[dict[str, Any], Path]:
    if bundle.is_dir():
        if (bundle / "site.json").is_file():
            bundle = bundle / "site.json"
        else:
            spec: dict[str, Any] = {"format": FORMAT, "film": "film"}
            if (bundle / "record").exists():
                spec["record"] = "record"
            if (bundle / "library").is_dir():
                spec["artifacts"] = [
                    {"label": p.name.capitalize(), "path": f"library/{p.name}"}
                    for p in sorted((bundle / "library").iterdir())
                    if p.is_dir()
                ]
            return spec, bundle
    spec = json.loads(bundle.read_text(encoding="utf-8"))
    if spec.get("format") != FORMAT:
        raise ValueError(f"{bundle}: not a {FORMAT} description")
    return spec, bundle.parent


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def build_site(bundle: Path, out: Path) -> SiteResult:
    spec, base = _load(bundle)
    # the story page, when the description names one: rendered first, so a missing record or
    # an unknown placeholder fails before the previous build is replaced
    told = None
    if spec.get("story"):
        from fleetlab.film import story

        entry = spec["story"]
        if not isinstance(entry, dict) or not entry.get("text") or not entry.get("data"):
            raise ValueError("story: an object with text (Markdown) and data (film-data.json)")
        told = story.render(_resolve(base, str(entry["text"])), _resolve(base, str(entry["data"])))
    if (out / ".git").exists():
        raise ValueError(f"{out} is a git checkout; build the site elsewhere and copy it in")
    if out.exists() and any(out.iterdir()):
        if not (out / "site-files.json").is_file():
            raise ValueError(f"{out} is not empty and is not a site this command built")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    result = SiteResult(out, 0, 0)
    film_dir = _resolve(base, str(spec.get("film", "film")))
    if not (film_dir / "film.mp4").is_file():
        raise ValueError(f"{film_dir} holds no film.mp4 (python -m fleetlab.film build)")
    provenance = json.loads((film_dir / "provenance.json").read_text())
    # the cut's own public rules: the film's manifest's and the site description's, together
    from fleetlab.film import public

    film_manifest = film_dir / "manifest.json"
    film_rules = public.public_rules(
        json.loads(film_manifest.read_text()).get("public") if film_manifest.is_file() else None
    )
    site_rules = public.public_rules(spec.get("public"))
    public.configure({**film_rules[0], **site_rules[0]}, film_rules[1] + site_rules[1])
    title = str(
        spec.get("title")
        or (told.title if told else "")
        or provenance.get("manifest", {}).get("title")
        or "The film"
    )
    sources_href = "sources.html" if told else "index.html#technical"

    # the film and the files beside it
    # the full provenance and the manifest name local paths: they stay with the records,
    # and the site gets film.json, written below from public identifiers
    for name in (
        "film.mp4",
        "poster.jpg",
        "captions.vtt",
        "chapters.vtt",
        "chapters.json",
        "timeline.png",
        "timeline.md",
    ):
        if (film_dir / name).is_file():
            _copy(film_dir / name, out / "film" / name)
    chapters = (
        json.loads((film_dir / "chapters.json").read_text())
        if (film_dir / "chapters.json").is_file()
        else []
    )
    web: dict[str, Any] = next(
        (o for o in provenance.get("outputs", []) if o.get("file") == "film.mp4"), {}
    )

    # the record
    record_link = ""
    converter = None
    record = spec.get("record")
    if record:
        record_path = _resolve(base, str(record))
        pages = [record_path] if record_path.is_file() else sorted(record_path.rglob("*.md"))
        root = record_path.parent if record_path.is_file() else record_path
        page_template = _template("page.html")
        for page in pages:
            meta, body, converter = markdown_to_html(page.read_text(encoding="utf-8"))
            body = _md_links_to_html(body)
            relative = page.relative_to(root).with_suffix(".html")
            if record_path.is_file():
                relative = Path("index.html")
            depth = len(relative.parts)
            target = out / "record" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            page_title = meta.get("title") or _first_heading(body) or title
            target.write_text(
                page_template.safe_substitute(
                    title=html.escape(page_title),
                    site_title=html.escape(title),
                    up="../" * depth,
                    body=body,
                    converter=html.escape(converter),
                    sources=sources_href,
                ),
                encoding="utf-8",
            )
        if record_path.is_dir():
            for file in sorted(record_path.rglob("*")):
                if file.is_file() and file.suffix != ".md" and not file.name.startswith("."):
                    _copy(file, out / "record" / file.relative_to(record_path))
        record_link = "record/index.html"

    # the clips the film shows
    clip_rows: list[str] = []
    seen: set[str] = set()
    copied: dict[str, str] = {}  # the clip's video SHA-256 -> its file in the site
    for beat in provenance.get("beats", []):
        for shot in beat.get("shots", []):
            for clip in shot.get("clips", []):
                videos = [
                    (
                        "original",
                        clip.get("original_video") or clip["video"],
                        clip.get("original_sha256") or clip["video_sha256"],
                    )
                ]
                if clip.get("original_video"):
                    videos.append(("re-render", clip["video"], clip["video_sha256"]))
                links = []
                for role, path, digest in videos:
                    src = Path(path)
                    name = (
                        f"{beat['id']}-{_slug(clip['episode'])}-{digest[:8]}"
                        + (".rerender" if role == "re-render" else "")
                        + ".mp4"
                    )
                    if not src.is_file():
                        result.warnings.append(f"clip {path} is gone; not copied")
                        continue
                    if src.stat().st_size > CLIP_LIMIT:
                        result.warnings.append(f"clip {path} is over 25 MB; not copied")
                        continue
                    if sha256(src) != digest:
                        result.warnings.append(
                            f"clip {path} changed since the film was built; not copied"
                        )
                        continue
                    if name not in seen:
                        _copy(src, out / "clips" / name)
                        seen.add(name)
                    copied[digest] = f"clips/{name}"
                    links.append(f'<a href="clips/{name}">{role}</a>')
                verified = ""
                if clip.get("rerender"):
                    state = clip["rerender"]
                    verified = (
                        " re-render used, verified"
                        if state.get("used")
                        else f" re-render refused: {html.escape(str(state.get('reason')))}"
                    )
                steps = clip.get("control_steps_shown") or ["", ""]
                run_id = public_ref(clip["run"]) if clip.get("run") else "placeholder: no run"
                original_sha = str(clip.get("original_sha256") or clip["video_sha256"])
                clip_rows.append(
                    "<tr>"
                    f"<td>{html.escape(beat['title'])}</td>"
                    f"<td>{html.escape(clip.get('label') or '')}</td>"
                    f"<td><code>{html.escape(run_id)}</code></td>"
                    f"<td><code>{html.escape(clip.get('episode') or '')}</code></td>"
                    f"<td>{steps[0]} to {steps[1]}</td>"
                    f"<td>{html.escape(str(clip.get('outcome') or ''))}{verified}</td>"
                    f'<td><code title="{original_sha}">{original_sha[:12]}</code></td>'
                    f"<td>{' · '.join(links) or 'not copied'}</td>"
                    "</tr>"
                )

    # the artifacts
    artifact_rows: list[str] = []
    for entry in spec.get("artifacts") or []:
        label = str(entry.get("label") or "Files")
        src = _resolve(base, str(entry["path"]))
        if not src.exists():
            result.warnings.append(f"artifacts {src} do not exist; none listed for {label}")
            continue
        files = (
            [src]
            if src.is_file()
            else sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith("."))
        )
        for file in files:
            shown = file.name if src.is_file() else str(file.relative_to(src))
            target = out / "artifacts" / _slug(label) / shown
            if file.stat().st_size > CLIP_LIMIT:
                result.warnings.append(f"artifact {file} is over 25 MB; not copied")
                continue
            _copy(file, target)
            purpose = _purpose(file)
            href = target.relative_to(out).as_posix()
            artifact_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f'<td><a href="{href}" download>{html.escape(shown)}</a></td>'
                f"<td>{html.escape(purpose)}</td>"
                f"<td>{_human(file.stat().st_size)}</td>"
                f"<td><code>{sha256(file)[:12]}</code></td>"
                "</tr>"
            )

    # the page
    beat_rows = []
    for ch in chapters:
        status = ch.get("status", "")
        beat_rows.append(
            "<tr>"
            f'<td><a href="film/film.mp4#t={ch["start_s"]}" data-t="{ch["start_s"]}">'
            f"{_clock(ch['start_s'])}</a></td>"
            f"<td>{html.escape(ch.get('story') or '')}</td>"
            f"<td>{html.escape(ch['title'])}</td>"
            f'<td class="status status-{_slug(status)}">{html.escape(status)}</td>'
            "</tr>"
        )
    code = provenance.get("code", {})
    # the preview text: the description's own, else the story's teaser, else its summary
    teaser = told.meta.get("teaser", "") if told else ""
    lede = str(spec.get("lede") or teaser or (told.lede_text if told else ""))
    transcript = []
    for beat in provenance.get("beats", []):
        spoken = (beat.get("narration") or {}).get("text")
        if spoken:
            transcript.append(
                f'<p><a href="film/film.mp4#t={beat["start_s"]}" data-t="{beat["start_s"]}">'
                f"{_clock(beat['start_s'])}</a> <strong>{html.escape(beat['title'])}.</strong> "
                f"{html.escape(spoken)}</p>"
            )
    base_url = str(spec.get("base_url") or "")
    if base_url and not base_url.endswith("/"):
        base_url += "/"
    preview = [
        f'<meta property="og:title" content="{html.escape(title)}">',
        f'<meta property="og:description" content="{html.escape(lede)}">',
        '<meta property="og:type" content="video.other">',
        '<meta name="twitter:card" content="summary_large_image">',
    ]
    if (out / "film" / "poster.jpg").is_file():
        preview.append(
            f'<meta property="og:image" content="{html.escape(base_url)}film/poster.jpg">'
        )
    preview.append(f'<meta property="og:video" content="{html.escape(base_url)}film/film.mp4">')
    preview.append('<meta property="og:video:type" content="video/mp4">')
    if base_url:
        preview.append(f'<meta property="og:url" content="{html.escape(base_url)}">')
    parts = dict(
        title=html.escape(title),
        lede=html.escape(lede),
        film_seconds=_clock(float(provenance.get("seconds", 0))),
        # how the film carries its captions: burned into the picture (the default, as the
        # earlier cuts did) or a selectable track, as the site description says
        captions_note=html.escape(str(spec.get("captions_note") or "captions in the picture")),
        film_size=_human(int(web.get("bytes", 0))) if web else "",
        film_sha=html.escape(str(web.get("sha256", ""))),
        poster='poster="film/poster.jpg"' if (out / "film" / "poster.jpg").is_file() else "",
        beats="\n".join(beat_rows) or '<tr><td colspan="4">No chapters.</td></tr>',
        record=(
            f'<p><a class="button" href="{record_link}">Read the record</a> '
            "The page behind the film: what was run, what was counted, and where every "
            "number comes from.</p>"
            if record_link
            else "<p>The record page is not part of this build yet.</p>"
        ),
        clips="\n".join(clip_rows) or '<tr><td colspan="8">No clips.</td></tr>',
        artifacts="\n".join(artifact_rows)
        or '<tr><td colspan="5">No artifacts in this build.</td></tr>',
        commit=html.escape(str(code.get("commit") or "unknown")[:12]),
        transcript="\n".join(transcript) or "<p>No narration in this cut.</p>",
        preview="\n".join(preview),
        built=html.escape(str(provenance.get("built", ""))),
    )
    if told is None:
        (out / "index.html").write_text(
            _template("index.html").safe_substitute(parts), encoding="utf-8"
        )
    else:
        _write_story(out, told, parts, chapters)
    (out / "film" / "film.json").write_text(
        json.dumps(
            _public_film(provenance, copied, sha256(film_dir / "provenance.json")),
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    # the stylesheet and the page's images (the opening's background frame among them)
    for asset in sorted(p for p in (SITE_ROOT / "static").iterdir() if p.is_file()):
        _copy(asset, out / "static" / asset.name)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    # sizes and the file list
    listing = []
    for file in sorted(p for p in out.rglob("*") if p.is_file()):
        size = file.stat().st_size
        rel = file.relative_to(out).as_posix()
        if size > LIMIT:
            result.errors.append(f"{rel} is {size / 1e6:.1f} MB, over GitHub Pages' 100 MB a file")
        elif size > WARN:
            result.warnings.append(f"{rel} is {size / 1e6:.1f} MB (GitHub warns above 50 MB)")
        listing.append({"path": rel, "bytes": size, "sha256": sha256(file)})
        result.bytes += size
    result.files = len(listing) + 1
    (out / "site-files.json").write_text(
        json.dumps(
            {
                "format": "fleetlab.film-site-files/1",
                "film": web,
                "record_converter": converter,
                "files": listing,
                "warnings": result.warnings,
                "errors": result.errors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for error in result.errors:
        result.warnings.append(error)
    hits = scan_tree(out)
    if hits:
        shutil.rmtree(out)
        raise SiteRefused(hits)
    return result


def _public_film(
    provenance: dict[str, Any], copied: dict[str, str], private_sha: str
) -> dict[str, Any]:
    """The film's provenance for the public: every clip, count and file by its stable
    identifier and SHA-256, the clips by their place in the site; no local path."""

    def clip(c: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": c.get("label"),
            "layout": c.get("layout"),
            "run": public_ref(c["run"]) if c.get("run") else None,
            "episode": c.get("episode"),
            "control_steps_shown": c.get("control_steps_shown"),
            "outcome": c.get("outcome"),
            "video_sha256": c.get("original_sha256") or c.get("video_sha256"),
            "file": copied.get(str(c.get("original_sha256") or c.get("video_sha256"))),
            "records_read": [
                {"id": public_ref(path), "sha256": digest}
                for path, digest in (c.get("files_read") or {}).items()
            ],
        }
        if c.get("placeholder"):
            out["placeholder"] = True
        if c.get("rerender"):
            r = c["rerender"]
            out["rerender"] = {
                "used": r.get("used"),
                "reason": r.get("reason"),
                "video_sha256": c.get("video_sha256") if r.get("used") else None,
                "file": copied.get(str(c.get("video_sha256"))) if r.get("used") else None,
            }
            if r.get("comparison"):  # the replay's comparison: numbers and hashes, no path
                out["rerender"]["comparison"] = dict(r["comparison"])
        return out

    def count(c: dict[str, Any]) -> dict[str, Any]:
        out = {k: c.get(k) for k in ("text", "detail", "kind", "k", "n", "where", "success", "key")}
        out["tables"] = [
            {"id": public_ref(t["path"]), "sha256": t["sha256"]} for t in c.get("tables") or []
        ]
        if c.get("kind") == "literal":
            out["source"] = c.get("source")
        return out

    beats = []
    for b in provenance.get("beats", []):
        shots = []
        for sh in b.get("shots", []):
            shots.append(
                {
                    "kind": sh.get("kind"),
                    "start_s": sh.get("start_s"),
                    "end_s": sh.get("end_s"),
                    "clips": [clip(c) for c in sh.get("clips", [])],
                    "counts": [count(c) for c in sh.get("counts", [])],
                    "files": [
                        {"id": public_ref(path), "sha256": digest}
                        for path, digest in (sh.get("files") or {}).items()
                    ],
                }
            )
        spoken = b.get("narration") or None
        beats.append(
            {
                "id": b.get("id"),
                "story": b.get("story"),
                "title": b.get("title"),
                "start_s": b.get("start_s"),
                "end_s": b.get("end_s"),
                "status": b.get("status"),
                "placeholder": b.get("placeholder"),
                "needs": b.get("needs"),
                "narration": None
                if spoken is None
                else {
                    k: spoken.get(k) for k in ("words", "seconds", "timing", "text", "voice_sha256")
                },
                "shots": shots,
            }
        )
    code = provenance.get("code") or {}
    return {
        "format": PUBLIC_FORMAT,
        "note": "Public provenance: runs, tables and files by stable identifier and SHA-256. "
        "The full provenance, with where each file lies, is kept with the run records.",
        "built": provenance.get("built"),
        "code_commit": code.get("commit"),
        "film": [
            {
                k: o.get(k)
                for k in (
                    "role",
                    "file",
                    "sha256",
                    "bytes",
                    "width",
                    "height",
                    "fps",
                    "seconds",
                    "video",
                    "audio",
                )
            }
            for o in provenance.get("outputs", [])
        ],
        "private_provenance_sha256": private_sha,
        "beats": beats,
    }


def _write_story(
    out: Path, told: Any, parts: dict[str, str], chapters: list[dict[str, Any]]
) -> None:
    """The story page as index.html, the clips, files and provenance on sources.html, and
    every number the page used, with where it came from, in story-values.json."""
    acts: list[str] = []
    seen: set[str] = set()
    for ch in chapters:  # the film's acts: the first chapter of each
        act = str(ch.get("story") or ch["title"])
        if act in seen:
            continue
        seen.add(act)
        acts.append(
            f'          <li><a href="film/film.mp4#t={ch["start_s"]}" data-t="{ch["start_s"]}">'
            f'<span class="t">{_clock(ch["start_s"])}</span> {html.escape(act)}</a></li>'
        )
    toc = [
        f'      <li><a href="#{html.escape(ident)}">{html.escape(text)}</a></li>'
        for ident, text in told.toc
    ]
    page = _template("story.html").safe_substitute(
        parts,
        title=html.escape(told.title or parts["title"]),
        kicker=html.escape(told.kicker),
        subtitle=html.escape(told.subtitle),
        teaser=html.escape(told.meta.get("teaser", "")),
        lede=told.lede,
        description=html.escape(told.meta.get("teaser") or told.lede_text),
        acts="\n".join(acts) or "<li>No chapters.</li>",
        toc="\n".join(toc),
        body=told.body,
    )
    from fleetlab.film import story

    (out / "index.html").write_text(page, encoding="utf-8")
    step_ms = told.values["b_step_ms"].value if "b_step_ms" in told.values else ""
    (out / "sources.html").write_text(
        _template("sources.html").safe_substitute(parts, step_ms=step_ms), encoding="utf-8"
    )
    # the fleet record: experiment two's counts with their intervals and records, written now
    # that the site's public rules (aliases) are in force
    record = story.fleet_record(told.data, told.meta.get("healthy", "").split())
    (out / "fleet-record.html").write_text(
        _template("fleet.html").safe_substitute(
            title=html.escape(record.title),
            site_title=html.escape(told.title or parts["title"]),
            body=record.body,
            step_ms=step_ms,
        ),
        encoding="utf-8",
    )
    charts = [
        {**c, "sources": [{**i, "id": story.public_id(i["id"])} for i in c["sources"]]}
        for c in told.charts
    ]
    (out / "story-values.json").write_text(
        json.dumps(
            {
                "format": "fleetlab.film-story-values/2",
                "note": "Every number on the story page: in 'values', each number of the text "
                "with the value it was filled with and where it came from ('numbers' are the "
                "film's bound numbers, bound from the run records; the others are derived "
                "from the film data's recorded tables); in 'charts', every chart's numbers "
                "with the records they were read from; in 'fleet_record', the tables of the "
                "fleet record page.",
                "text_sha256": told.text_sha256,
                "data_sha256": told.data_sha256,
                "values": {k: {"value": v.value, "source": v.source} for k, v in told.used.items()},
                "charts": charts,
                "fleet_record": record.tables,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _first_heading(body: str) -> str | None:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else None


def _purpose(file: Path) -> str:
    """A saved job, check or recipe says what it is for; other files say nothing."""
    if file.suffix != ".json":
        return ""
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(data, dict):
        return str(data.get("purpose") or data.get("description") or "")[:200]
    return ""
