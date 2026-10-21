"""Replay equivalence guaranteed by the engine, never by a measured zero gain.

Only *added* config_change faults on ConfigChange.NO_MECHANISM keys are omitted.
Original calibration/release faults remain: those kinds can carry real mechanisms.
The engine owns the sole key list; the oracle's hook regression exercises it.
"""

from collections.abc import Iterable, Mapping

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import CONFIG_CHANGE_KIND, FixSet, Resolution
from fleetlab.contracts.faults import FaultSpec, TruthTable
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import OracleTable
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey
from fleetlab.faults.config_change import NO_MECHANISM


def mechanism_additions(faults: Iterable[FaultSpec]) -> tuple[FaultSpec, ...]:
    return tuple(
        f
        for f in faults
        if not (f.kind == CONFIG_CHANGE_KIND and f.target.config_key in NO_MECHANISM)
    )


def mechanism_resolution(result: Resolution) -> Resolution:
    return Resolution(removed=result.removed, added=mechanism_additions(result.added))


class MechanismTable(OracleTable):
    """An ordinary table with the engine's equivalence rule at every public lookup."""

    def resolution(
        self,
        fix_set: FixSet,
        identity: EpisodeIdentity,
        truth: TruthTable,
        values: Mapping[ConfigKey | ConditionKey, ConfigValue] | None,
    ) -> Resolution:
        return mechanism_resolution(super().resolution(fix_set, identity, truth, values))

    def success(self, episode_id: str, faults: Iterable[FaultSpec]) -> bool | None:
        original = self._original_ids().get(episode_id, frozenset())
        requested = tuple(faults)
        canonical = tuple(
            f for f in requested if f.fault_id in original or mechanism_additions((f,))
        )
        known = super().success(episode_id, canonical)
        if canonical == requested:
            return known
        recorded = super().success(episode_id, requested)
        if known is not None and recorded is not None and known != recorded:
            raise ValueError(f"{episode_id}: a no-mechanism addition changed the outcome")
        return known if known is not None else recorded

    def _original_ids(self) -> dict[str, frozenset[str]]:
        if not hasattr(self, "_original"):
            self._original = {i.episode_id: i.fault_ids for i in self.identities}
        return self._original

    def restricted(self, identities: Iterable[EpisodeIdentity]) -> OracleTable:
        sub = MechanismTable(identities, excluded=self.excluded)
        sub._rows = self._rows
        return sub


def mechanism_table(table: OracleTable) -> MechanismTable:
    if isinstance(table, MechanismTable):
        return table
    result = MechanismTable(table.identities, excluded=table.excluded)
    result._rows = table._rows
    return result
