"""Visible exposure histories, health smoothing and repair counterfactuals."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.baselines.classical.health import health_series
from fleetlab.baselines.classical.regression import Array
from fleetlab.contracts.actions import Action, cost_of
from fleetlab.contracts.ids import scope_covers
from fleetlab.contracts.vocabulary import Component, Verb


def token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Design:
    def __init__(self, data: VisibleData):
        self.data = data
        self.health = health_series(data)
        self.tasks = sorted({r["task_text"] for r in data.episodes})
        self.robots = sorted(r.id for r in data.fleet.robots())
        self.sites = sorted(s.id for s in data.fleet.sites)
        self.initial: dict[str, dict[str, Any]] = {
            "fleet": {str(k): v for k, v in data.fleet.config.values.items()},
            **{s.id: {str(k): v for k, v in s.config.values.items()} for s in data.fleet.sites},
            **{r.id: {str(k): v for k, v in r.config.values.items()} for r in data.fleet.robots()},
        }
        # Undo in reverse time; if a first local value equals its parent's initial
        # value, infer inheritance before that first logged override.
        for change in reversed(data.changes):
            self.initial[change["scope"]][change["key"]] = change["old"]
        for scope in [*self.sites, *self.robots]:
            parent = dict(self.initial["fleet"])
            if scope in self.robots:
                parent.update(self.initial[data.fleet.robot(scope).site_id])
            for key in list(self.initial[scope]):
                if token(self.initial[scope][key]) == token(parent.get(key)):
                    del self.initial[scope][key]
        configs = [self.configuration(r["robot_id"], r["day"]) for r in data.episodes]
        keys = sorted({k for c in configs for k in c})
        self.config_features: list[tuple[str, str]] = []
        self.observed_config_values: dict[str, set[str]] = {}
        for key in keys:
            counts = Counter(token(c.get(key)) for c in configs)
            self.observed_config_values[key] = set(counts)
            # Full coding is essential under the survival link's <=0 bound:
            # either level can carry the hazard, independent of which is modal.
            self.config_features.extend((key, v) for v in sorted(counts))
        self.condition_features: list[tuple[str, str]] = []
        self.condition_reference: dict[str, Any] = {}
        for key in sorted({k for s in data.fleet.sites for k in s.conditions}):
            values = [token(s.conditions.get(key)) for s in data.fleet.sites]
            reference = sorted(Counter(values).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            self.condition_reference[key] = json.loads(reference)
            self.condition_features.extend((key, v) for v in sorted(set(values)))
        self.nuisance = 1 + len(self.tasks) + len(self.sites) + len(self.robots) + 5
        self.names = (
            ["intercept"]
            + [f"task:{s}" for s in self.tasks]
            + [f"site:{s}" for s in self.sites]
            + [f"robot:{s}" for s in self.robots]
            + [f"day:{d}" for d in range(1, 6)]
            + [f"config:{k}={v}" for k, v in self.config_features]
            + [f"health:{h.component}.{h.metric}:{h.direction:g}" for h in self.health]
            + [f"condition:{k}={v}" for k, v in self.condition_features]
        )
        self.penalty = np.asarray(
            [0.01]
            + [1.0] * len(self.tasks)
            + [8.0] * len(self.sites)
            + [16.0] * len(self.robots)
            + [8.0] * 5
            + [0.5] * len(self.config_features)
            + [1.0] * len(self.health)
            + [0.5] * len(self.condition_features)
        )
        self.x = self.matrix(data.episodes)
        # A constant exposure, including a fleet change at day zero, has no
        # observed comparator. Give it no fitted effect (and flag it in evidence).
        self.unsupported = np.ptp(self.x, axis=0) < 1e-12
        self.unsupported[: self.nuisance] = False
        self.x[:, self.unsupported] = 0

    def configuration(
        self, robot: str, day: float, action: Action | None = None, actions: tuple[Action, ...] = ()
    ) -> dict[str, Any]:
        site = self.data.fleet.robot(robot).site_id
        states = {scope: dict(self.initial[scope]) for scope in ("fleet", site, robot)}
        for change in self.data.changes:
            if change["day"] <= day and change["scope"] in states:
                states[change["scope"]][change["key"]] = change["new"]
        for intervention in (*actions, action) if action else actions:
            if intervention.verb != Verb.revert or intervention.target.scope not in states:
                continue
            history = [
                c
                for c in self.data.changes
                if c["scope"] == intervention.target.scope
                and c["key"] == intervention.target.config_key
                and c["day"] <= day
            ]
            if history:
                last = history[-1]
                states[intervention.target.scope][last["key"]] = last["old"]
        return {**states["fleet"], **states[site], **states[robot]}

    def matrix(
        self,
        episodes: list[Row],
        *,
        future: bool = False,
        flat: bool = False,
        action: Action | None = None,
        actions: tuple[Action, ...] = (),
        forecast_day: float | None = None,
    ) -> Array:
        rows = []
        for row in episodes:
            robot, site = row["robot_id"], row["site_id"]
            # Standardize to each robot's observed task distribution, with
            # configuration persisting and health projected to scoring midpoint.
            day = (
                forecast_day if forecast_day is not None else (8.5 if future else float(row["day"]))
            )
            interventions = (*actions, action) if action else actions
            config = self.configuration(robot, day, actions=interventions)
            conditions = {str(k): v for k, v in self.data.fleet.site(site).conditions.items()}
            covered = [a for a in interventions if scope_covers(a.target.scope, robot, site)]
            for intervention in covered:
                if intervention.verb == Verb.change_condition:
                    conditions[str(intervention.target.condition_key)] = intervention.value
            features = [1.0] + [float(row["task_text"] == t) for t in self.tasks]
            features += [float(site == s) for s in self.sites]
            features += [float(robot == r) for r in self.robots]
            # Do not extrapolate an unobserved shared day shock.
            features += [float(min(day, 5) == d) for d in range(1, 6)]
            features += [float(token(config.get(k)) == v) for k, v in self.config_features]
            for h in self.health:
                camera = h.component in {Component.scene_camera, Component.wrist_camera}
                repair_verb = (
                    (Verb.recalibrate if h.metric == "calibration_residual" else Verb.clean)
                    if camera
                    else Verb.service
                )
                repaired = any(
                    a.target.component == h.component and a.verb == repair_verb for a in covered
                )
                features.append(0.0 if repaired else h.exposure(robot, day, flat=flat))
            features += [float(token(conditions.get(k)) == v) for k, v in self.condition_features]
            rows.append(features)
        result = np.asarray(rows, dtype=float)
        if hasattr(self, "unsupported"):
            result[:, self.unsupported] = 0
        return result

    def candidates(self) -> list[Action]:
        """Geometry recalibration and image cleaning are separate hypotheses.

        Calibration residuals support geometry repair; other image health readings
        support cleaning. These are declared engineering hypotheses, not proof
        that the reading can distinguish every physical camera failure.
        """
        actions = []
        for action in self.data.actions:
            component = action.target.component
            if component:
                desired = (
                    {Verb.recalibrate, Verb.clean}
                    if component in {Component.scene_camera, Component.wrist_camera}
                    else {Verb.service}
                )
                if action.verb not in desired:
                    continue
            elif action.verb == Verb.change_condition:
                key = str(action.target.condition_key)
                if key not in self.condition_reference:
                    continue
                action = action.model_copy(update={"value": self.condition_reference[key]})
            elif action.verb != Verb.revert or not any(
                c["scope"] == action.target.scope and c["key"] == action.target.config_key
                for c in self.data.changes
            ):
                continue
            if action.is_complete:
                actions.append(action)
        actions.sort(key=lambda a: (cost_of(a, self.data.costs) or 0, a.model_dump_json()))
        return list({a.model_dump_json(): a for a in actions}.values())
