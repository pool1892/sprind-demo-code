"""DuckDB over an exported episode log: one view per table, steps and frames by episode.

    reader = LogReader(export_dir)                       # views over the Parquet files
    reader.query("SELECT outcome, count(*) FROM episodes GROUP BY outcome")
    reader.frames("e000003", [0, 40, 80], Camera.scene)  # images at the logged frames

`root` is a directory with a `log/` tree in the layout of
`contracts.log` (an export, or a run directory, of which only `log/` is
read). Every table of the visible schema gets a view, also when it has no
rows yet. With `materialize=True` the tables are copied into memory; with
`lock=True` DuckDB's file access is then switched off (materialized) or
limited to `log/` (views), and the setting locked. Only materialized and
locked is closed: with views and a lock, queries can still read every file
under `log/` and write there with `COPY ... TO`. The tool server runs with
`materialize=True, lock=True`.

This module reads only the visible tree and imports only `contracts`.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import duckdb
import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from fleetlab.contracts.env import Camera
from fleetlab.contracts.ids import IdKind, check_neutral
from fleetlab.contracts.log import DESCRIPTION_FILE, visible_schema
from fleetlab.contracts.model import _LOGS
from fleetlab.log.dictionary import DICTIONARY_FILE, data_dictionary

__all__ = ["Frame", "LogReader"]


class Frame(NamedTuple):
    """A video frame: the step asked for, the step it shows, its index in the video, the image."""

    step: int
    frame_step: int
    frame_index: int
    image: npt.NDArray[np.uint8]


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _files(root: Path, template: str) -> tuple[str, bool]:
    """The file or glob of a table and whether anything matches it."""
    if "{" in template:
        pattern = template.split("{", 1)[0] + "*" + template.rsplit("}", 1)[1]
        return str(root / pattern), any(root.glob(pattern))
    return str(root / template), (root / template).is_file()


class LogReader:
    """A DuckDB connection with one view (or table) per visible table of a log tree."""

    def __init__(self, root: Path | str, *, materialize: bool = False, lock: bool = False) -> None:
        self.root = Path(root).resolve()
        self.log_dir = self.root / "log"
        if not self.log_dir.is_dir():
            raise FileNotFoundError(f"{self.root} has no log/ directory")
        self.con = duckdb.connect(":memory:")
        self.tables: tuple[str, ...] = tuple(visible_schema())
        for name, spec in visible_schema().items():
            self._create(name, spec.path, [(c.name, c.type) for c in spec.columns], materialize)
        self.locked = False
        if lock:
            self.lock(materialized=materialize)

    def _create(
        self, name: str, template: str, columns: list[tuple[str, Any]], materialize: bool
    ) -> None:
        source, exists = _files(self.root, template)
        kind = "TABLE" if materialize else "VIEW"
        if exists:
            self.con.execute(
                f"CREATE {kind} {name} AS SELECT * FROM read_parquet({_sql_string(source)})"
            )
            return
        empty = pa.schema([pa.field(n, t) for n, t in columns]).empty_table()
        self.con.register("_empty", empty)
        self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _empty")
        self.con.unregister("_empty")

    def lock(self, *, materialized: bool) -> None:
        """Switch file access off (or limit it to `log/`) and lock the configuration.

        Materialized, DuckDB can open no file at all. With views it must still
        read the Parquet files, so access is limited to the `log/` directory:
        queries can then read any file there and DuckDB can also write there
        (`COPY ... TO`). The tool server therefore runs materialized and locked.
        """
        if not materialized:
            allowed = _sql_string(str(self.log_dir) + "/")
            self.con.execute(f"SET allowed_directories = [{allowed}]")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self.locked = True

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> LogReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pa.Table:
        """Run SQL and return the result as an Arrow table."""
        result: pa.Table = self.con.execute(sql, list(params or [])).to_arrow_table()
        return result

    def episode_ids(self) -> list[str]:
        ids: list[str] = self.query("SELECT episode_id FROM episodes ORDER BY episode_id")[
            "episode_id"
        ].to_pylist()
        return ids

    def episode(self, episode_id: str) -> dict[str, Any]:
        """The `episodes` row of one episode; KeyError if there is none."""
        rows = self.query("SELECT * FROM episodes WHERE episode_id = ?", [episode_id]).to_pylist()
        if not rows:
            raise KeyError(f"no episode {episode_id!r}")
        row: dict[str, Any] = rows[0]
        return row

    def steps(self, episode_id: str) -> pa.Table:
        """The `steps` rows of one episode, in order."""
        return self.query("SELECT * FROM steps WHERE episode_id = ? ORDER BY step", [episode_id])

    def frames(self, episode_id: str, steps: Sequence[int], camera: Camera | str) -> list[Frame]:
        """The video frames nearest to the given steps (the earlier one on a tie)."""
        camera = Camera(camera)
        check_neutral(IdKind.episode, episode_id)
        video = self.episode(episode_id)[f"video_{camera.value}"]
        if video is None:
            raise ValueError(f"episode {episode_id} has no video")
        path = (self.log_dir / video).resolve()
        if not path.is_relative_to(self.log_dir) or not path.is_file():
            raise ValueError(f"episode {episode_id}: no video file {video!r}")
        logged = self.query(
            "SELECT step, frame_index FROM steps WHERE episode_id = ? "
            "AND frame_index IS NOT NULL ORDER BY step",
            [episode_id],
        ).to_pylist()
        if not logged:
            raise ValueError(f"episode {episode_id} has no frames")
        frame_steps = [r["step"] for r in logged]
        chosen = []
        for step in steps:
            i = bisect.bisect_left(frame_steps, step)
            candidates = [j for j in (i - 1, i) if 0 <= j < len(frame_steps)]
            j = min(candidates, key=lambda c: (abs(frame_steps[c] - step), frame_steps[c]))
            chosen.append((int(step), logged[j]["step"], logged[j]["frame_index"]))
        images = _decode(path, {index for _, _, index in chosen})
        return [Frame(s, fs, index, images[index]) for s, fs, index in chosen]

    def description(self, include_logs: bool = True) -> dict[str, Any]:
        """The system description; without the change and maintenance logs if asked."""
        data: dict[str, Any] = json.loads((self.root / DESCRIPTION_FILE).read_text())
        if not include_logs:
            data = {**data, "fleet": {k: v for k, v in data["fleet"].items() if k not in _LOGS}}
        return data

    def data_dictionary(self) -> str:
        """The export's data dictionary, or the generated one for a run directory."""
        path = self.root / DICTIONARY_FILE
        return path.read_text() if path.is_file() else data_dictionary()


def _decode(path: Path, wanted: set[int]) -> dict[int, npt.NDArray[np.uint8]]:
    out: dict[int, npt.NDArray[np.uint8]] = {}
    last = max(wanted)
    reader: Any = imageio.get_reader(path, format="FFMPEG")  # type: ignore[arg-type]
    try:
        for index, frame in enumerate(reader):
            if index in wanted:
                out[index] = np.asarray(frame, dtype=np.uint8)
            if index >= last:
                break
    finally:
        reader.close()
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"{path.name} has no frame {missing[0]}")
    return out
