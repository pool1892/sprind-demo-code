"""The loop's development fleets w70 and w71, as stored, through LoopWorld: the compensation
setting is in their public catalogues, an arm or a ledger entry can set it at a remote site,
and the public description and the engine's effective policy agree on its value, before and
after.

Both were composed before `WorldSpec.later_settings` existed; their stored fleet models
carry the setting's fleet default, and the loop reads the stored model (`LoopWorld.fleet_on`
starts from `WorldTruth.fleet`), so no bridge is needed. Runs only where the development
worlds are stored (this machine); nothing is simulated or written."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.actions import Action, catalogue
from fleetlab.contracts.faults import Target
from fleetlab.contracts.model import describe_for_agents, effective_config
from fleetlab.contracts.vocabulary import ConfigKey, Verb

WORLDS = Path.home() / ".cache/robots-probes/worlds"
KEY = ConfigKey.control_latency_compensation_steps
CONDITION = "remote_link_latency"


@pytest.fixture(scope="module", params=["w70", "w71"])
def world(request: pytest.FixtureRequest) -> Any:
    from fleetlab.loop.world import LoopWorld

    root = WORLDS / request.param
    if not (root / "files").is_dir():
        pytest.skip(f"{request.param} is not stored here")
    return LoopWorld.load(root)


def _public(world: Any, ledger: Any = None, day: int | None = None) -> Any:
    from fleetlab.loop.state import Ledger

    ledger = ledger or Ledger.empty(world.world_id)
    return world.public_fleet(day or world.spec.windows.observed.last, ledger)


def _remote_sites(public: Any) -> list[str]:
    """The sites whose public note says their robots are driven over a remote link."""
    return [s.id for s in public.sites if any("remote" in n or "wide-area" in n for n in s.notes)]


def test_the_setting_is_in_the_public_catalogue_and_description(world: Any) -> None:
    assert "later_settings" not in (world.files / "spec.json").read_text()  # composed before
    public = _public(world)
    assert public.config.values[KEY] == 0
    remote = _remote_sites(public)
    assert remote
    offered = {(a.verb, a.target.scope) for a in catalogue(public) if a.target.config_key is KEY}
    for site in remote:
        assert {(Verb.set, site), (Verb.revert, site)} <= offered
    assert (Verb.set, "fleet") in offered
    vocabulary = describe_for_agents(public)["vocabulary"]["config_keys"]
    assert KEY.value in {k["name"] for k in vocabulary}


def test_an_arm_sets_it_at_a_remote_site_and_the_engine_runs_that_value(world: Any) -> None:
    from fleetlab.loop.state import Ledger

    day = world.spec.windows.experiment.first
    ledger = Ledger.empty(world.world_id)
    public = _public(world)
    for site in _remote_sites(public):
        action = Action(verb=Verb.set, target=Target(scope=site, config_key=KEY), value=17)
        for robot in [r.id for r in public.site(site).robots]:
            plain = world.state_on(robot, day, ledger)
            armed = world.state_on(robot, day, ledger, [action])
            # the description's value and the engine's effective policy agree, before and after
            assert effective_config(public, robot)[KEY] == 0
            assert plain.policy_of(list(plain.faults)).latency_compensation_steps == 0
            assert armed.policy_of(list(armed.faults)).latency_compensation_steps == 17
            assert armed.current[KEY] == 17
            # the lasting condition stays in force under the setting
            assert CONDITION in {f.kind for f in plain.faults}
            assert CONDITION in {f.kind for f in armed.faults}
        # a robot of another site is untouched by this site's setting
        other = next(r.id for r in public.robots() if r.site_id != site)
        state = world.state_on(other, day, ledger, [action])
        assert state.policy_of(list(state.faults)).latency_compensation_steps == 0


def test_a_ledger_entry_sets_it_too(world: Any) -> None:
    from fleetlab.loop.state import Adoption, Ledger

    site = _remote_sites(_public(world))[0]
    action = Action(verb=Verb.set, target=Target(scope=site, config_key=KEY), value=17)
    day = world.spec.windows.experiment.last + 1
    ledger = Ledger.empty(world.world_id).adopt(
        1, day, adopt=[Adoption(action=action, rationale="probe")]
    )
    robot = _public(world).site(site).robots[0].id
    state = world.state_on(robot, day, ledger)
    assert state.policy_of(list(state.faults)).latency_compensation_steps == 17
    assert effective_config(_public(world, ledger, day), robot)[KEY] == 17


@pytest.mark.parametrize("name", ["LEAD", "LATER"])
def test_a_newly_composed_loop_fleet_has_the_setting(name: str) -> None:
    """The fleet lane's composer opts every loop fleet in (`WorldSpec.later_settings`)."""
    from fleetlab.worlds import loop_fleets as lf

    composed = lf.compose(lf.RECIPES[name])
    assert composed.spec.later_settings == [KEY]
