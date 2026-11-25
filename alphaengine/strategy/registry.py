"""Persisted, immutable strategy configs. Changing a strategy means bumping its version."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from alphaengine.core.errors import StrategyError
from alphaengine.data.models import StrategyConfigRecord
from alphaengine.strategy.config import AnyConfig, config_hash, parse_config


def register_config(session: Session, cfg: AnyConfig) -> StrategyConfigRecord:
    """Store ``cfg`` (idempotent). Raises if the same name+version exists with other content."""
    h = config_hash(cfg)
    existing = session.scalars(
        select(StrategyConfigRecord).where(
            StrategyConfigRecord.name == cfg.name, StrategyConfigRecord.version == cfg.version
        )
    ).one_or_none()
    if existing is not None:
        if existing.config_hash != h:
            raise StrategyError(
                f"{cfg.name} v{cfg.version} is already registered with different content "
                f"({existing.config_hash} != {h}); bump the version"
            )
        return existing
    rec = StrategyConfigRecord(
        name=cfg.name, version=cfg.version, config_hash=h, config=cfg.model_dump(mode="json")
    )
    session.add(rec)
    session.flush()
    return rec


def get_config(session: Session, name: str, version: int | None = None) -> AnyConfig:
    stmt = select(StrategyConfigRecord).where(StrategyConfigRecord.name == name)
    if version is not None:
        stmt = stmt.where(StrategyConfigRecord.version == version)
    rec = session.scalars(stmt.order_by(StrategyConfigRecord.version.desc()).limit(1)).first()
    if rec is None:
        raise StrategyError(
            f"no registered strategy {name!r}" + (f" v{version}" if version else "")
        )
    return parse_config(rec.config)


def list_configs(session: Session) -> list[tuple[str, int, str]]:
    rows = session.execute(
        select(
            StrategyConfigRecord.name,
            StrategyConfigRecord.version,
            StrategyConfigRecord.config_hash,
        ).order_by(StrategyConfigRecord.name, StrategyConfigRecord.version)
    )
    return [(n, v, h) for n, v, h in rows]
