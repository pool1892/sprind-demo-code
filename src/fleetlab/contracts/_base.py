"""What the contract models share: the pydantic base class, the type of a value,
and the visibility tags of principle 2.

Not part of the public interface; the names are re-exported where they belong.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

# A configuration value, a condition value or the value of an action. Typed
# rather than "any JSON" so that the agents' answer schemas stay strict
# (OpenAI's strict mode refuses an untyped value). A camera calibration is a
# list of numbers; the others are a number, a string or a flag.
ConfigValue = bool | int | float | str | list[float] | None


class ContractModel(BaseModel):
    """Base of every contract model: unknown fields are refused, instances are frozen.

    Frozen because these are records: a fault, an identity, an action. A changed
    record is a new instance (`model_copy(update=...)`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Visibility(StrEnum):
    """Whether the agents may see a stored column or a field of the system model."""

    visible = "visible"
    hidden = "hidden"


# Field tags for `Annotated[type, VISIBLE]`. The system description is built
# from the fields tagged VISIBLE, never by removing the hidden ones.
VISIBLE = Visibility.visible
HIDDEN = Visibility.hidden


def field_visibility(model: type[BaseModel], name: str) -> Visibility | None:
    """The tag of one field of a model class, or None if it carries none."""
    tags = [m for m in model.model_fields[name].metadata if isinstance(m, Visibility)]
    if len(tags) > 1:
        raise TypeError(f"{model.__name__}.{name} carries {len(tags)} visibility tags")
    return tags[0] if tags else None


def untagged_fields(model: type[BaseModel]) -> list[str]:
    """The fields of a model class without a visibility tag."""
    return [name for name in model.model_fields if field_visibility(model, name) is None]


def visible_dump(value: Any, drop: frozenset[str] = frozenset()) -> Any:
    """A model's JSON form with only its VISIBLE fields, recursively.

    Fields are taken from the allow-list of tags; an untagged field raises, so
    a new field cannot slip through unclassified. Names in `drop` are left out
    as well, at any depth, and so is a field whose `exclude_if` holds for its
    value, as in pydantic's own dump. Lists and dictionaries are walked; other
    values are dumped as pydantic dumps them in JSON mode.
    """
    if isinstance(value, BaseModel):
        cls = type(value)
        out: dict[str, Any] = {}
        for name, info in cls.model_fields.items():
            tag = field_visibility(cls, name)
            if tag is None:
                raise TypeError(f"{cls.__name__}.{name} has no visibility tag")
            if tag is VISIBLE and name not in drop:
                item = getattr(value, name)
                if info.exclude_if is not None and info.exclude_if(item):
                    continue
                out[name] = visible_dump(item, drop)
        return out
    if isinstance(value, dict):
        return {_key(k): visible_dump(v, drop) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [visible_dump(v, drop) for v in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _key(key: Any) -> str:
    return key.value if isinstance(key, StrEnum) else str(key)
