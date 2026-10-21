"""The cut's public rules are module state (`fleetlab.film.public.configure`): reset per test."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from fleetlab.film import public


@pytest.fixture(autouse=True)
def _no_cut_rules() -> Iterator[None]:
    public.configure()
    yield
    public.configure()
