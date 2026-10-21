"""Build the agent's workspace from a fleet run, and check that it names no fault.

    python export.py RUN_DIR WORKSPACE_DIR

The export reads RUN_DIR/log and nothing else: it never opens RUN_DIR/truth. It
writes

    WORKSPACE/episodes.parquet     one row per episode
    WORKSPACE/steps.parquet        one row per control step, all episodes
    WORKSPACE/video/<id>.scene.mp4, <id>.wrist.mp4

and then searches the result (see `leak_check`) for anything that names a fault,
a seed or the truth. The words it searches for come from the truth side
(faults.py, the manifest of the run, a canary token), which is why the check is
a separate step that does look at the truth.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

GENERIC_WORDS = ["fault", "truth", "replay", "variant", "severity", "noise_seed", "scene_seed", "delay",
                 "camera_pan", "degrees", "oracle", "inject", "canary"]


def export(run_dir: Path, ws: Path) -> dict:
    log = run_dir / "log"
    if ws.exists():
        # The workspace is rebuilt from scratch. It holds copies only, and no links.
        assert ws.name.startswith("workspace") and not any(p.is_symlink() for p in ws.rglob("*")), ws
        shutil.rmtree(ws)
    (ws / "video").mkdir(parents=True)
    episodes = pq.read_table(log / "episodes.parquet")
    pq.write_table(episodes, ws / "episodes.parquet", compression="zstd")
    ids = episodes.column("episode_id").to_pylist()
    steps = pa.concat_tables([pq.read_table(log / "steps" / f"{i}.parquet") for i in ids])
    pq.write_table(steps, ws / "steps.parquet", compression="zstd")
    for i in ids:
        for cam in ("scene", "wrist"):
            shutil.copyfile(log / "video" / f"{i}.{cam}.mp4", ws / "video" / f"{i}.{cam}.mp4")
    sizes = {
        "episodes": len(ids), "step_rows": steps.num_rows,
        "episodes_parquet_bytes": (ws / "episodes.parquet").stat().st_size,
        "steps_parquet_bytes": (ws / "steps.parquet").stat().st_size,
        "video_scene_bytes": sum((ws / "video" / f"{i}.scene.mp4").stat().st_size for i in ids),
        "video_wrist_bytes": sum((ws / "video" / f"{i}.wrist.mp4").stat().st_size for i in ids),
    }
    sizes["bytes_per_episode"] = round(
        (sizes["episodes_parquet_bytes"] + sizes["steps_parquet_bytes"] + sizes["video_scene_bytes"]
         + sizes["video_wrist_bytes"]) / len(ids)
    )
    return sizes


def leak_check(ws: Path, words: list[str], extra_texts: dict[str, str] | None = None) -> dict:
    """Search the workspace for the given words. Returns every hit; an empty list means clean.

    Three passes: the raw bytes of every file (catches file contents, container
    tags and Parquet footers); file and directory names; and the decoded Parquet
    side (column names, key-value metadata, every distinct text value), because
    Parquet compresses its pages and a raw search would miss a word in a cell.
    Words shorter than five letters are searched in text only: in 30 MB of video
    a three-letter sequence turns up by chance.
    """
    import av

    hits, searched = [], {"files": 0, "bytes": 0, "parquet_text_values": 0, "names": 0, "video_tags": 0}
    lower = [w.lower() for w in words]

    # A word counts when no letter stands directly before it: "fault_table" and "faults" are
    # hits, "default" is not.
    text_pat = {w: re.compile("(?<![a-z])" + re.escape(w)) for w in lower}
    byte_pat = {w: re.compile(b"(?<![a-z])" + re.escape(w.encode())) for w in lower}

    def scan_text(where: str, text: str) -> None:
        t = text.lower()
        for w in lower:
            m = text_pat[w].search(t)
            if m:
                hits.append({"where": where, "word": w, "context": text[max(0, m.start() - 30): m.start() + 50]})

    for p in sorted(ws.rglob("*")):
        searched["names"] += 1
        scan_text(f"name of {p.relative_to(ws)}", p.name)
        if p.is_symlink():
            hits.append({"where": str(p.relative_to(ws)), "word": "(symbolic link)", "context": ""})
            continue
        if not p.is_file():
            continue
        data = p.read_bytes().lower()
        searched["files"] += 1
        searched["bytes"] += len(data)
        for w in lower:
            m = byte_pat[w].search(data) if len(w) >= 5 else None
            if m:
                hits.append({"where": f"bytes of {p.relative_to(ws)}", "word": w,
                             "context": repr(data[max(0, m.start() - 20): m.start() + 40])})
        if p.suffix == ".parquet":
            t = pq.read_table(p)
            scan_text(f"columns of {p.name}", " ".join(t.column_names))
            scan_text(f"metadata of {p.name}", json.dumps(
                {k.decode(errors="replace"): v.decode(errors="replace") for k, v in (t.schema.metadata or {}).items()}))
            scan_text(f"file metadata of {p.name}", str(pq.ParquetFile(p).metadata.created_by))
            for name in t.column_names:
                col = t.column(name)
                if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                    for v in col.unique().to_pylist():
                        searched["parquet_text_values"] += 1
                        scan_text(f"value in {p.name}.{name}", str(v))
        if p.suffix == ".mp4":
            with av.open(str(p)) as c:
                tags = dict(c.metadata)
                for s in c.streams:
                    tags.update({f"stream.{k}": v for k, v in s.metadata.items()})
            searched["video_tags"] += len(tags)
            scan_text(f"tags of {p.relative_to(ws)}", json.dumps(tags))
    for where, text in (extra_texts or {}).items():
        scan_text(where, text)
    return {"words": words, "searched": searched, "hits": hits}


def truth_words(run_dir: Path) -> list[str]:
    """Words that would give the truth away, collected from the truth side of the run."""
    words = set(GENERIC_WORDS)
    for line in (run_dir / "truth" / "episodes.jsonl").read_text().splitlines():
        for f in json.loads(line).get("faults") or []:
            words.add(f["kind"])  # the parameter names ("steps") are ordinary words and are left out
    canary = run_dir / "truth" / "canary.txt"
    if canary.exists():
        words.add(canary.read_text().strip())
    return sorted(words)


if __name__ == "__main__":
    run_dir, ws = Path(sys.argv[1]), Path(sys.argv[2])
    ids = pq.read_table(run_dir / "log" / "episodes.parquet").column("episode_id").to_pylist()
    if not all(re.fullmatch(r"e\d{4}", i) for i in ids) and "--test" not in sys.argv:
        # Dose and replay runs name their episodes after fault levels. Only a run with neutral
        # episode numbers may become a workspace.
        sys.exit("episode ids of this run are not neutral numbers; refusing to export it (--test overrides)")
    sizes = export(run_dir, ws)
    print(json.dumps(sizes, indent=1))
    result = leak_check(ws, truth_words(run_dir))
    print(json.dumps({k: v for k, v in result.items() if k != "hits"}, indent=1))
    print(f"hits: {len(result['hits'])}")
    for h in result["hits"]:
        print("  ", h)
    (ws.parent / "leak_check_workspace.json").write_text(json.dumps({"sizes": sizes, **result}, indent=1))
    sys.exit(1 if result["hits"] else 0)
