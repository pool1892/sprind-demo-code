"""Suite-scoped verdicts and source directories in the canonical library record."""

import json
from pathlib import Path

import pytest

from fleetlab import dose_response as dr
from fleetlab import dose_response_stage3 as s3
from fleetlab.faults import records


@pytest.mark.parametrize("kind", ["illumination_change", "backdrop_change", "light_color_shift"])
def test_decoy_verdict_is_scoped_to_the_task_family(kind: str) -> None:
    spatial = records.measured(kind, dr.SUITE)
    objects = records.measured(kind, s3.OBJECT)
    assert spatial["suite"] == dr.SUITE and spatial["verdict"] == "decoy"
    assert objects["suite"] == s3.OBJECT and objects["verdict"] == "kept"
    assert {p["role"] for p in objects["points"]} == {"effect"}
    assert "Small cause" in objects["note"]
    assert all(
        p["interval_interpretation"] == "descriptive under precision-driven sampling"
        for p in objects["points"]
    )
    with pytest.raises(KeyError):
        records.measured(kind, "unmeasured-suite")


def test_every_canonical_point_and_joint_names_its_source_and_twins() -> None:
    curves = records._curves()
    points = [p for entry in curves["types"].values() for p in entry["points"]]
    points += curves["supplement"]["points"]
    assert len(points) == 105
    for point in points + records.joints():
        assert [Path(p).name for p in point["run_dirs"]] == point["run_id"].split(" + ")
        assert point["healthy_run_dirs"]
        assert all(Path(p).is_absolute() for p in point["run_dirs"] + point["healthy_run_dirs"])
        if "members" in point:
            assert len(point["single_run_dirs"]) == len(point["members"])
            assert all(point["single_run_dirs"])
    for kind, entry in curves["types"].items():
        assert "verdict" not in entry  # no unqualified type verdict survives
        for suite, verdict in entry["verdicts"].items():
            curve = records.measured(kind, suite)
            assert verdict == dr.type_verdict([p["role"] for p in curve["points"]])
    with pytest.raises(KeyError):
        records.measured("lens_contamination", s3.OBJECT)


def test_publication_refuses_a_point_without_its_source(tmp_path: Path) -> None:
    path = tmp_path / "curves.json"
    path.write_text(json.dumps(records._curves()))
    before = path.read_bytes()
    point = {**records.measured("camera_pose_offset")["points"][0], "chain_ok": True}
    point.pop("run_dirs")
    result = {"types": {}, "appended": {"camera_pose_offset": [point]}, "joints": []}
    with pytest.raises(ValueError, match="source run directories"):
        s3.write_curves(result, path)
    assert path.read_bytes() == before
