"""Sequential targeted removal checks, without any world inputs.

Usage: uv run python tests/classical_removals.py NEW_OUTPUT_DIRECTORY
Each mutation is applied to a private source copy before its test process starts.
The original checkout and stored records are never edited by this harness.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

# The tests assert behavioral contracts on independently planted or noisy data.
REMOVALS = [
    (
        "later-break",
        "analysis.py",
        "        if z > -1.0:",
        "        if False:",
        "test_classical_freeze3.py::test_later_distinct_break_does_not_support_an_earlier_edit",
    ),
    (
        "panel-envelope",
        "analysis.py",
        "        low -= prediction_radius\n        high += prediction_radius",
        "        pass",
        "test_classical_retained_rules.py::test_noise_free_parameter_draws_still_have_future_panel_noise",
    ),
    (
        "naming-floor",
        "analysis.py",
        '        and r["gain_points"] >= 0.5',
        '        and r["gain_points"] >= -100',
        "test_classical_retained_rules.py::test_one_naming_floor_and_multiplicity_screen",
    ),
    (
        "warning-support",
        "analysis.py",
        " or effect_confidence < 0.9",
        "",
        "test_classical_warning_lead.py::test_a_forecast_crossing_needs_support_for_the_health_effect",
    ),
    (
        "inheritance",
        "features.py",
        "                if token(self.initial[scope][key]) == token(parent.get(key)):",
        "                if False:",
        "test_classical_retained_rules.py::test_inherited_initial_value_and_day_five_persist_in_forecast",
    ),
    (
        "constant-exposure",
        "features.py",
        "self.unsupported = np.ptp(self.x, axis=0) < 1e-12",
        "self.unsupported = np.ptp(self.x, axis=0) < -1",
        "test_classical_retained_rules.py::test_inherited_initial_value_and_day_five_persist_in_forecast",
    ),
    (
        "day-five",
        "features.py",
        "float(min(day, 5) == d)",
        "float(day == d)",
        "test_classical_retained_rules.py::test_inherited_initial_value_and_day_five_persist_in_forecast",
    ),
    (
        "health-deadband",
        "health.py",
        "max(z - 1.0, 0.0)",
        "max(z, 0.0)",
        "test_classical_retained_rules.py::test_health_noise_deadband_suppresses_healthy_measurement_variation",
    ),
    (
        "health-search",
        "health.py",
        '2 * np.log(max(1, len(rows) - 2)) if kind in {"step", "hinge"} else 0.0',
        "0.0",
        "test_classical_retained_rules.py::test_searching_for_a_short_health_break_pays_a_noise_penalty",
    ),
    (
        "second-cut",
        "selection.py",
        "for size in (3, 3):",
        "for size in (3,):",
        "test_classical_retained_rules.py::test_second_cut_is_exact_conditional_on_first_three",
    ),
    (
        "label-probability",
        "signals.py",
        "            y[i] = probability",
        "            y[i] = 1.0",
        "test_classical_retained_rules.py::test_suspect_short_failures_keep_peer_error_probability",
    ),
    (
        "budget",
        "experiments.py",
        "    if not 1 <= requests <= 8 or not 1 <= episodes_per_arm <= 60:",
        "    if False:",
        "test_classical_retained_rules.py::test_public_budget_rejected_before_io",
    ),
    (
        "name-every-test",
        "experiments.py",
        "    accepted = named_actions(records)",
        "    accepted = set(range(len(records)))",
        "test_classical_freeze3.py::test_current_evidence_names_causes_in_both_conditions",
    ),
    (
        "alarm-direction",
        "experiments.py",
        ' and r["direction"] == "decline"',
        "",
        "test_classical_freeze3.py::test_improvement_alarm_does_not_replace_deficit_site_in_plan",
    ),
    (
        "hidden-traits",
        "data.py",
        "    if any(r.traits for r in fleet.robots()):",
        "    if False:",
        "test_classical_retained_rules.py::test_hidden_robot_traits_are_rejected_even_in_an_otherwise_valid_export",
    ),
    (
        "physical-bound",
        "analysis.py",
        "        low, high = max(low, -population), min(high, population)",
        "        pass",
        "test_classical_retained_rules.py::test_noisy_near_complete_repair_cannot_exceed_affected_population",
    ),
    (
        "multiplicity",
        "analysis.py",
        'eligible[i] for i in _bh([1 - records[i]["diagnosis_confidence"] for i in eligible])',
        "i for i in eligible",
        "test_classical_retained_rules.py::test_one_naming_floor_and_multiplicity_screen",
    ),
    (
        "full-levels",
        "features.py",
        "for v in sorted(counts))",
        "for v in sorted(counts)[1:])",
        "test_classical_retained_rules.py::test_beneficial_release_is_not_forced_to_be_harmful_by_reference_coding",
    ),
    (
        "remote-prehistory",
        "onset.py",
        'onset - 2 <= r["day"] < onset + 2',
        '0 <= r["day"] < onset + 2',
        "test_classical_freeze3.py::test_release_estimate_is_invariant_to_remote_declines",
    ),
    (
        "joint-release",
        "selection.py",
        "if np.any(self.offsets):",
        "if False:",
        "test_classical_freeze3.py::test_release_effect_is_carried_into_joint_selection_and_experiment_transport",
    ),
    (
        "paired-covariance",
        "experiments.py",
        "draws += scale * (posterior[:, 1] - posterior[:, 2])",
        "draws += scale * (rng.beta(30.5, 30.5, len(draws)) - rng.beta(30.5, 30.5, len(draws)))",
        "test_classical_freeze3.py::test_pair_covariance_controls_uncertainty_and_keeps_nonzero_null_noise",
    ),
    (
        "outward-rounding",
        "protocol.py",
        "value + 8 * float(np.spacing(value))",
        "value",
        "test_classical_freeze3.py::test_population_interval_bound_is_outward_for_equivalent_fractions",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    results = []
    for name, filename, old, new, test in REMOVALS:
        root = args.output / name
        source = root / "src/fleetlab"
        shutil.copytree(
            repository / "src/fleetlab",
            source,
            ignore=shutil.ignore_patterns("__pycache__", "*.json", "*.md"),
        )
        shutil.copyfile(repository / "pyproject.toml", root / "pyproject.toml")
        path = source / "baselines/classical" / filename
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f"mutation pattern is not unique: {name}")
        path.write_text(text.replace(old, new))
        env = {
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(repository / "tests" / test),
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(root / "pytest"),
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (root / "test.txt").write_text(result.stdout + result.stderr)
        row = dict(
            name=name,
            test=test,
            file=filename,
            old=old,
            replacement=new,
            exit_code=result.returncode,
            caught=result.returncode == 1 and "failed" in result.stdout,
        )
        results.append(row)
        print(name, "caught" if row["caught"] else "NOT CAUGHT", flush=True)
        (args.output / "results.json").write_text(
            json.dumps(
                dict(
                    removals=results,
                    max_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                ),
                indent=2,
            )
            + "\n"
        )
    if not all(r["caught"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
