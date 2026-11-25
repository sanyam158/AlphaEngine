"""Feature registry. A feature is a pure, causal function ``(MarketData, **params) -> panel``.

Register with ``@feature("name", history=...)``; reference with
``FeatureSpec("name", {"param": value})`` (hashable, so panels are cached per spec).
``history`` is a callable on the params giving how many rows of history the feature needs
before it produces values; strategies use it to skip symbols with too little history.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.features.data import MarketData

FeatureFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    fn: FeatureFn
    defaults: Mapping[str, Any]
    history: Callable[[Mapping[str, Any]], int]
    description: str


_REGISTRY: dict[str, FeatureDef] = {}


def feature(
    name: str, *, history: Callable[[Mapping[str, Any]], int], **defaults: Any
) -> Callable[[FeatureFn], FeatureFn]:
    def deco(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} already registered")
        doc = (fn.__doc__ or "").strip()
        _REGISTRY[name] = FeatureDef(name, fn, dict(defaults), history, doc)
        return fn

    return deco


def registered_features() -> dict[str, FeatureDef]:
    import alphaengine.features.library  # noqa: F401, PLC0415 - registers built-ins

    return dict(_REGISTRY)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: tuple[tuple[str, Any], ...] = field(default=())

    @classmethod
    def of(cls, name: str, **params: Any) -> FeatureSpec:
        definition = get_definition(name)
        unknown = set(params) - set(definition.defaults)
        if unknown:
            raise StrategyError(f"feature {name!r} has no parameter(s) {sorted(unknown)}")
        merged = {**definition.defaults, **params}
        return cls(name, tuple(sorted(merged.items())))

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def history(self) -> int:
        return get_definition(self.name).history(self.kwargs)

    def __str__(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}({args})"


def get_definition(name: str) -> FeatureDef:
    import alphaengine.features.library  # noqa: F401, PLC0415 - registers built-ins

    try:
        return _REGISTRY[name]
    except KeyError:
        raise StrategyError(f"unknown feature {name!r}; known: {sorted(_REGISTRY)}") from None


def compute(spec: FeatureSpec, data: MarketData) -> pd.DataFrame:
    out = get_definition(spec.name).fn(data, **spec.kwargs)
    if not isinstance(out, pd.DataFrame):
        raise StrategyError(f"feature {spec} did not return a DataFrame")
    return out.reindex(index=data.close.index)
