"""The fault registry: every fault type by its kind (docs/design/contracts.md, section 7).

A fault type is a module in this package that registers a class under its
`kind` when it is imported; the rollout needs no change for a new one.
`load_all()` imports every module of the package, so that the registry holds
every kind. Importing this package calls it once, so every process that
names the registry (`fleetlab.faults:REGISTRY`, as the engine's spawned
workers do) sees every kind; calling it again is harmless. This package is never mounted
into an agent's sandbox and the contracts never import it, so the fault kinds,
the unseen ones included, cannot be read off a module the agents see.

    from fleetlab.contracts import FaultTypeBase
    from fleetlab.faults import register

    @register("my_fault")
    class MyFault(FaultTypeBase):
        kind = "my_fault"
        ...

`register` checks the type against the contracts (`check_fault_type`): the
truth mapping's rules, and `writes_change` exactly for the configuration and
software classes. `truth_table()` gives the scorer and the oracle the truth
mapping of every registered kind (`contracts.actions.resolve` takes it as its
`truth`).

The two generic kinds for the changes an action adds
(`contracts.actions.ADDED_CHANGE_KINDS`, section 8) are registered here like
any other, one type each:

- `config_change`: `params = {"value": v}`, level 1.0, `nominal` the value it
  replaced, the target naming the configuration key at the action's scope. It
  dispatches on the key to the hook that key uses: `policy.sampling_steps`,
  `policy.image_views`, `policy.precision` and
  `policy.actions_executed_per_chunk` through the policy input hook,
  `control.action_delay_steps` through the action hook, and so on. A key with
  no mechanism is a no-op; `software.version` means something only through the
  software fault type's definition of what each version carries. Class
  configuration, so it writes a `Change`; fixed by `set` and `revert`.
- `condition_change`: the same for a condition key at a site (class site, no
  `Change`), dispatching to the hooks that condition uses (the cell hook, and
  for `placement` the observation and action hooks as well); fixed by
  `change_condition`.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, TypeVar

from fleetlab.contracts.faults import FaultSpec, FaultType, TruthMapping, check_fault_type

_T = TypeVar("_T", bound=type[Any])


class FaultRegistry:
    """Fault types by kind. The module-level `REGISTRY` is the one the engine uses."""

    def __init__(self) -> None:
        self._types: dict[str, FaultType] = {}

    def add(self, fault_type: object) -> None:
        """Register an instance under its own `kind`; refuse a kind taken by another class."""
        checked = check_fault_type(fault_type)
        existing = self._types.get(checked.kind)
        if existing is not None and _qualified(existing) != _qualified(checked):
            raise ValueError(
                f"fault kind {checked.kind!r} is already registered by {_qualified(existing)}"
            )
        self._types[checked.kind] = checked

    def register(self, kind: str) -> Callable[[_T], _T]:
        """Class decorator: registers an instance of the class under `kind`."""

        def decorate(cls: _T) -> _T:
            instance = cls()
            if getattr(instance, "kind", None) != kind:
                raise ValueError(f"{cls.__name__}.kind is not {kind!r}")
            self.add(instance)
            return cls

        return decorate

    def get(self, kind: str) -> FaultType:
        try:
            return self._types[kind]
        except KeyError:
            known = ", ".join(self.kinds()) or "none"
            raise KeyError(f"no fault type registered as {kind!r}; registered: {known}") from None

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))

    def truth_table(self) -> dict[str, TruthMapping]:
        """The truth mapping of every registered kind."""
        return {kind: fault_type.truth for kind, fault_type in self._types.items()}

    def check(self, spec: FaultSpec) -> None:
        """Raise if `spec` names an unknown kind or does not fit its type's truth mapping."""
        self.get(spec.kind).truth.check(spec)
        self.get(spec.kind).Params.model_validate(spec.params)


def _qualified(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


REGISTRY = FaultRegistry()


def register(kind: str) -> Callable[[_T], _T]:
    """Register a fault type class in the default registry under `kind`."""
    return REGISTRY.register(kind)


def get(kind: str) -> FaultType:
    """The fault type registered under `kind` in the default registry."""
    return REGISTRY.get(kind)


def kinds() -> tuple[str, ...]:
    """The kinds registered in the default registry, sorted."""
    return REGISTRY.kinds()


def truth_table() -> dict[str, TruthMapping]:
    """The truth mapping of every kind in the default registry."""
    return REGISTRY.truth_table()


def load_all() -> tuple[str, ...]:
    """Import every module of this package, so that each registers its types; return the kinds."""
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
    return kinds()


# Every process that imports the registry, a worker the engine spawns included, sees every
# kind. The fault modules import `register` and `get` from here, which exist by now.
load_all()
