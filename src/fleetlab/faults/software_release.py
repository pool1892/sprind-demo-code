"""A software release that brings a fault with it: a software fault.

The change log shows a `Change` of `software.version` and nothing else; what
the release carries (a longer delay in the command buffer, a changed policy
setting, a degraded image pipeline, ...) stays hidden in the release. `revert`
or `set` of `software.version` back to the old release removes it. The carried
faults act through their own types' hooks (`CarrierBase`), so a release that
carries a fault produces the very episodes that fault produces on its own.
The experimental design calls the type `software_version`, a name the vocabulary
already spells (`ConfigKey.software_version`), so the kind is
`software_release`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Field, JsonValue, model_validator

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import ADDED_CHANGE_KINDS
from fleetlab.contracts.faults import FaultSpec, Schedule
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass, TargetKind
from fleetlab.faults import get, register
from fleetlab.faults._carrier import CarrierBase
from fleetlab.faults._common import (
    SOFTWARE,
    DosePoint,
    FaultParams,
    config_key_of,
    config_truth,
    target_of,
)

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity

# A version as a fleet writes it: numbers and dots, nothing that names a fault.
VERSION_PATTERN = r"^[0-9]+(\.[0-9]+){1,3}$"


class Carried(FaultParams):
    """One fault a release brings: its kind, level, params, the thing it acts on
    (a component, configuration key or condition, by its vocabulary value) and,
    for a key or a condition, the value it replaces.
    """

    kind: str
    level: float = Field(gt=0.0)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    target: str
    nominal: ConfigValue = None


class ReleaseParams(FaultParams):
    """The release the robots now run (`version`) and the faults it brings (`carries`)."""

    version: Annotated[str, Field(pattern=VERSION_PATTERN)]
    carries: list[Carried] = Field(min_length=1)

    @model_validator(mode="after")
    def _carried_faults(self) -> ReleaseParams:
        """Build each carried fault once, so that a bad one fails when the world is made."""
        for carried in self.carries:
            if carried.kind == "software_release" or carried.kind in ADDED_CHANGE_KINDS:
                raise ValueError(f"a release cannot carry a fault of kind {carried.kind}")
            carried_spec(carried, "fleet", Schedule(onset_day=0), "f01")
        return self


@register("software_release")
class SoftwareRelease(CarrierBase):
    """A release of the robot software that brings the faults it carries (software class).

    Level 1 while the release is in force; `params.version` is the new
    version, `nominal` the one it replaced. The dose-response of a release is
    the dose-response of what it carries.
    """

    kind = "software_release"
    fault_class = FaultClass.software
    Params = ReleaseParams
    dose_levels = (0.0, 1.0)
    truth = config_truth(SOFTWARE)
    calibration: tuple[DosePoint, ...] = (
        DosePoint(
            "release with a 17-step delay",
            1.0,
            {
                "version": "2.5.0",
                "carries": [
                    {
                        "kind": "action_latency",
                        "level": 17.0,
                        "target": ConfigKey.control_action_delay_steps.value,
                        "nominal": 0,
                    }
                ],
            },
            ConfigKey.software_version.value,
            "2.4.1",
        ),
        # Two changes in one release, each flat on its own at 100 episodes (a 14-step
        # delay: -6 points; 10 of 30 actions per chunk: -4): what they do together.
        DosePoint(
            "release with a 14-step delay and 10 actions per chunk",
            1.0,
            {
                "version": "2.5.0",
                "carries": [
                    {
                        "kind": "action_latency",
                        "level": 14.0,
                        "target": ConfigKey.control_action_delay_steps.value,
                        "nominal": 0,
                    },
                    {
                        "kind": "policy_knob",
                        "level": 1.0,
                        "params": {"value": 10},
                        "target": ConfigKey.policy_actions_executed_per_chunk.value,
                        "nominal": 30,
                    },
                ],
            },
            ConfigKey.software_version.value,
            "2.4.1",
        ),
    )

    def inner_specs(self, spec: FaultSpec, identity: EpisodeIdentity) -> list[FaultSpec]:
        config_key_of(spec, {ConfigKey.software_version})
        params = ReleaseParams.model_validate(spec.params)
        return [
            carried_spec(c, spec.target.scope, spec.schedule, spec.fault_id) for c in params.carries
        ]

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return ReleaseParams.model_validate(spec.params).version


def carried_spec(carried: Carried, scope: str, schedule: Schedule, fault_id: str) -> FaultSpec:
    """The fault a release carries, at the release's scope, under the release's id."""
    try:
        carried_type = get(carried.kind)
    except KeyError as error:
        raise ValueError(str(error)) from None
    kind = carried_type.truth.target_kind
    spec = FaultSpec(
        fault_id=fault_id,
        kind=carried.kind,
        params=carried.params,
        level=carried.level,
        target=target_of(scope, carried.target, kind),
        schedule=schedule,
        nominal=None if kind is TargetKind.component else carried.nominal,
    )
    carried_type.Params.model_validate(spec.params)
    return spec
