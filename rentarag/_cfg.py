"""Shared helpers for the frozen config dataclasses (presets, dict round trip)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any, TypeVar

T = TypeVar("T")


def config_from_dict(cls: type[T], data: Mapping[str, Any] | None, presets: Mapping[str, Mapping[str, Any]]) -> T:
    data = dict(data or {})
    preset = data.pop("preset", None)
    values: dict[str, Any] = {}
    if preset is not None:
        if preset not in presets:
            raise ValueError(f"unknown {cls.__name__} preset {preset!r}; choose from {', '.join(presets)}")
        values.update(presets[preset])
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} option(s) {unknown}; valid options: {', '.join(sorted(known))}")
    values.update(data)
    return cls(**values)


def config_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def coerce_config(cls: type[T], value: Any, presets: Mapping[str, Mapping[str, Any]]) -> T:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, str):
        return config_from_dict(cls, {"preset": value}, presets)
    if isinstance(value, Mapping):
        return config_from_dict(cls, value, presets)
    raise TypeError(f"expected {cls.__name__}, a preset name or a dict, got {type(value).__name__}")
