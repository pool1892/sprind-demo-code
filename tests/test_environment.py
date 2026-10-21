"""The installed packages agree with what they declare to need, with known exceptions.

pyproject.toml forces numpy to 1.26.4 with a uv override. An override hides
every package's numpy requirement from the resolver, so nothing would notice a
package that really needs numpy 2. This test notices.

Requirements behind an extra count when something installed asks for that
extra, as this project does with `lerobot[smolvla]`.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# (package, requirement it declares) pairs that are knowingly not met.
# pyproject.toml gives the reason for each.
KNOWN = {
    ("lerobot", "numpy"),  # declares numpy>=2; verified on 1.26.4
    ("hf-libero", "opencv-python"),  # opencv-python-headless provides cv2
    ("robosuite", "opencv-python"),
    ("robomimic", "egl-probe"),  # hf-egl-probe provides egl_probe
}


def applies(requirement: Requirement, extras: Iterable[str]) -> bool:
    """Whether a requirement holds here: on this platform, plainly or through one of the extras."""
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras))


def requested_extras(requirements: dict[str, list[Requirement]]) -> dict[str, set[str]]:
    """The extras of each installed package that some installed package asks for."""
    requested: dict[str, set[str]] = {package: set() for package in requirements}
    grew = True
    while grew:  # an extra can ask for further extras, as lerobot[smolvla] does
        grew = False
        for package, declared in requirements.items():
            for requirement in declared:
                name = canonicalize_name(requirement.name)
                if name in requested and applies(requirement, requested[package]):
                    extras = {canonicalize_name(extra) for extra in requirement.extras}
                    if not extras <= requested[name]:
                        requested[name] |= extras
                        grew = True
    return requested


def unmet_requirements() -> dict[tuple[str, str], str]:
    installed: dict[str, str] = {
        canonicalize_name(dist.metadata["Name"]): dist.version for dist in metadata.distributions()
    }
    requirements = {
        package: [Requirement(line) for line in metadata.requires(package) or []]
        for package in installed
    }
    requested = requested_extras(requirements)

    unmet: dict[tuple[str, str], str] = {}
    for package, declared in requirements.items():
        for requirement in declared:
            if not applies(requirement, requested[package]):
                continue
            name = canonicalize_name(requirement.name)
            version = installed.get(name)
            if version is None or not requirement.specifier.contains(version, prereleases=True):
                unmet[(package, name)] = f"{package} needs {requirement}, installed: {version}"
    return unmet


def test_requested_extras_are_followed() -> None:
    # Guards the check itself: without extras it would skip what SmolVLA needs.
    requirements = {
        package: [Requirement(line) for line in metadata.requires(package) or []]
        for package in ("fleetlab", "lerobot", "torch")
    }
    requested = requested_extras(requirements)
    assert {"smolvla", "transformers-dep", "accelerate-dep"} <= requested["lerobot"]
    assert requested["torch"] == {"device-gfx1151"}


def test_declared_requirements_are_met_except_the_known_ones() -> None:
    unmet = unmet_requirements()
    unexpected = [message for key, message in unmet.items() if key not in KNOWN]
    assert not unexpected, "\n".join(unexpected)
    stale = KNOWN - unmet.keys()
    assert not stale, f"no longer unmet, remove from KNOWN: {sorted(stale)}"
