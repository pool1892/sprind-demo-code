"""The M3 recipe mechanics (docs/design/m3-recipe.md, section 2), on synthetic seeds.

CPU only. No world is composed and no real seed is drawn: the builders are given seeds of
the generator's form (bit 62 set) chosen here, and only the recipes are compared.
"""

from __future__ import annotations

from collections import Counter

import pytest

from fleetlab.worlds import loop_fleets as lf

SEEDS = [(1 << 62) + 7919 * k for k in range(300)]
IDS = {"world_id": "w90", "public_fleet_id": "w09"}


def _site_of(scope: str) -> str:
    if scope in lf.SITES:
        return scope
    return next(s for s in lf.SITES if scope in lf._robots_of(s))


def _parts(recipe: lf.LoopFleetRecipe) -> dict[str, tuple[object, ...]]:
    """Everything of a part but where it lands (and its prose, which names the place)."""
    return {
        p.handle: (p.kind, p.label, p.onset, p.part, p.version, p.nominal_version)
        for p in recipe.planted
    }


def _scope_kind(scope: str) -> str:
    return "fleet" if scope == "fleet" else "site" if scope in lf.SITES else "robot"


# ---- the permutation -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS[:50])
def test_the_placement_is_a_deterministic_bijection(seed: int) -> None:
    placement = lf.drawn_placement(seed)
    assert placement == lf.drawn_placement(seed)
    assert sorted(placement.sites) == sorted(placement.sites.values()) == list(lf.SITES)
    robots = [r for s in lf.SITES for r in lf._robots_of(s)]
    assert sorted(placement.robots) == sorted(placement.robots.values()) == robots
    for template, site in placement.sites.items():
        assert {placement.robots[r] for r in lf._robots_of(template)} == set(lf._robots_of(site))


def test_the_placement_depends_on_the_seed() -> None:
    orders = Counter(tuple(lf.drawn_placement(s).sites.values()) for s in SEEDS)
    assert len(orders) == 6  # every order of three sites occurs over 300 seeds
    robot_orders = {tuple(lf.drawn_placement(s).robots.values()) for s in SEEDS[:20]}
    assert len(robot_orders) > 1


# ---- the later fleet's condition site ------------------------------------------------------


def test_the_later_site_is_never_the_leads() -> None:
    reached: Counter[tuple[str, str]] = Counter()
    for lead_seed, later_seed in zip(SEEDS, reversed(SEEDS), strict=True):
        lead = lf.m3_lead(lead_seed, **IDS)
        lead_site = lf.condition_site(lead)
        later = lf.m3_later(later_seed, "w91", "w10", lead_site)
        site = lf.condition_site(later)
        assert site != lead_site
        assert site == lf.later_condition_site(lead_site, later_seed)
        reached[(lead_site, site)] += 1
    # Over these seeds every lead site occurs and each of its two other sites is drawn.
    assert {pair for pair in reached} == {(a, b) for a in lf.SITES for b in lf.SITES if a != b}


def test_the_later_rule_is_fixed() -> None:
    import hashlib

    for seed in SEEDS[:30]:
        for lead_site in lf.SITES:
            others = [s for s in lf.SITES if s != lead_site]
            byte = hashlib.sha256(f"m3-later-condition-site:{seed}".encode()).digest()[0]
            assert lf.later_condition_site(lead_site, seed) == others[byte % 2]
    with pytest.raises(ValueError):
        lf.later_condition_site("s4", SEEDS[0])


# ---- the variant ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS[:20])
def test_the_variant_drops_exactly_the_second_layers(seed: int) -> None:
    lead, variant = lf.m3_lead(seed, **IDS), lf.m3_variant(seed, **IDS)
    masked = set(lf.masked_layers(lf.RECIPES["LEAD"]))
    assert masked == {"f09", "f10", "f11"}
    assert {p.kind for p in lf.RECIPES["LEAD"].planted if p.handle in masked} == {
        "gripper_wear",
        "action_latency",
        "camera_pose_offset",
    }
    assert {p.handle for p in lead.planted} - {p.handle for p in variant.planted} == masked
    kept = {p.handle: p for p in lead.planted if p.handle not in masked}
    assert {p.handle: p for p in variant.planted} == kept  # the same parts, the same places
    assert variant.reveals == ()
    removed = {p.scope for p in lead.planted if p.handle in masked}
    # The removed parts by what the lead's summary calls them and by where they were.
    assert not [w for w in ("pads", "buffer", "waits", "second one", *removed) if w in variant.note]
    assert (variant.confounders, variant.decoys, variant.lasting) == (
        lead.confounders,
        lead.decoys,
        lead.lasting,
    )
    assert variant.site_suites == lead.site_suites


# ---- the parts and levels against the development recipes -----------------------------------


@pytest.mark.parametrize("seed", SEEDS[:20])
def test_the_leads_parts_and_levels_are_w70s(seed: int) -> None:
    from fleetlab.worlds.compose import dose_point

    template, lead = lf.RECIPES["LEAD"], lf.m3_lead(seed, **IDS)
    assert _parts(lead) == _parts(template)
    for before, after in zip(template.planted, lead.planted, strict=True):
        assert _scope_kind(before.scope) == _scope_kind(after.scope)
        if before.kind != "software_release":
            assert (
                dose_point(after.kind, after.label).level
                == dose_point(before.kind, before.label).level
            )
    assert (lead.confounders, lead.decoys, lead.reveals, lead.lasting) == (
        template.confounders,
        template.decoys,
        template.reveals,
        template.lasting,
    )
    assert (lead.windows, lead.episodes_per_robot_day) == (
        template.windows,
        template.episodes_per_robot_day,
    )
    # What sat together stays together, and the condition's site keeps its suite.
    site = {p.handle: _site_of(p.scope) for p in lead.planted}
    condition = lf.condition_site(lead)
    assert {site[h] for h in lead.lasting} == {condition} == {site["f09"]}
    assert site["f05"] == site["f06"] == site["f10"]
    assert site["f07"] == site["f08"] == site["f11"]
    assert len({condition, site["f05"], site["f07"]}) == 3
    assert lead.site_suites == ((condition, "libero_spatial"),)


@pytest.mark.parametrize("seed", SEEDS[:20])
def test_the_laters_parts_and_levels_are_w71s(seed: int) -> None:
    template = lf.RECIPES["LATER"]
    later = lf.m3_later(seed, **IDS, lead_condition_site="s2")
    assert _parts(later) == _parts(template)
    site = {p.handle: _site_of(p.scope) for p in later.planted}
    condition = lf.condition_site(later)
    assert condition != "s2"
    note_site = site["f01"]
    assert {site[h] for h in ("f01", "f02", "f03", "f04", "f09")} == {note_site}
    assert site["f10"] == site["f11"] == site["f12"]
    assert len({condition, note_site, site["f10"]}) == 3
    suites = dict(later.site_suites)
    assert suites[condition] == "libero_object"
    assert suites[note_site] == suites[site["f10"]] == "libero_spatial"


def test_the_m3_recipes_cannot_be_composed_from_the_command_line() -> None:
    assert not [name for name in lf.RECIPES if name.startswith("M3")]


# ---- the pending options -------------------------------------------------------------------


def test_the_pending_options_are_marked_and_inert() -> None:
    options = lf.PENDING_LATER_OPTIONS
    assert set(options) == {"release-blur", "arm-0.4"}
    assert all(o.status == "pending" for o in options.values())
    template = {p.handle: p for p in lf.RECIPES["LATER"].planted}
    assert template[options["release-blur"].handle].label == "camera_latency|24 steps scene lag"
    assert template[options["arm-0.4"].handle].label == "0.3 torque lost"
    later = lf.m3_later(SEEDS[0], **IDS, lead_condition_site="s1")
    held = {p.handle: p for p in later.planted}
    for option in options.values():
        assert held[option.handle].label == template[option.handle].label  # w71's part
    for name in options:
        with pytest.raises(NotImplementedError):
            lf.m3_later(SEEDS[0], **IDS, lead_condition_site="s1", options=[name])
    with pytest.raises(KeyError):
        lf.m3_later(SEEDS[0], **IDS, lead_condition_site="s1", options=["no-such-option"])


def test_every_placed_part_is_a_valid_fault() -> None:
    from fleetlab.worlds.compose import make_fault

    for recipe in (
        lf.m3_lead(SEEDS[1], **IDS),
        lf.m3_variant(SEEDS[2], **IDS),
        lf.m3_later(SEEDS[3], **IDS, lead_condition_site="s3"),
    ):
        for p in recipe.planted:
            make_fault(
                p.handle,
                p.kind,
                p.label,
                p.scope,
                p.onset,
                nominal_version=p.nominal_version,
                version=p.version,
            )


def test_the_prose_moves_with_the_parts() -> None:
    lead = lf.m3_lead(SEEDS[5], **IDS)
    placement = lf.drawn_placement(SEEDS[5])
    held = {p.handle: p for p in lead.planted}
    wear_robot = placement.robots["r03"]
    assert held["f09"].scope == wear_robot
    assert held["f09"].note.startswith(f"{wear_robot}'s gripper pads")
    blur_site = placement.sites["s2"]
    assert held["f05"].note.endswith(f"at {blur_site} blurs the scene camera image by 4 px")
    assert held["f05"].shows_in == f"change log (software.version at {blur_site}, day 1)"
    assert "17 control steps late" in held["f01"].note  # numbers and words are left alone
