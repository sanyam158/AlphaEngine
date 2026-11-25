from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from alphaengine.core.errors import StrategyError
from alphaengine.data.db import session_scope
from alphaengine.data.store import PriceStore
from alphaengine.strategy.config import (
    BlendConfig,
    PipelineConfig,
    build_strategy,
    builtin_names,
    config_hash,
    load_config,
    load_strategy,
    parse_config,
)
from alphaengine.strategy.pipeline import Blend, PipelineStrategy
from alphaengine.strategy.registry import get_config, list_configs, register_config
from alphaengine.strategy.stages import RegimeFilter, TopN
from tests.helpers import make_bars

SF = sessionmaker[Session]

MINIMAL = {
    "name": "tiny",
    "universe": {"type": "static", "symbols": ["A", "B"]},
    "scorer": {"type": "feature", "feature": "momentum", "params": {"lookback": 3, "skip": 0}},
    "selector": {"type": "top_n", "n": 1},
}


def test_all_builtins_build() -> None:
    names = builtin_names()
    assert {"buy_hold_nifty", "dual_momentum", "momentum_top20", "core_satellite"} <= set(names)
    for n in names:
        s = load_strategy(n)
        assert s.name == n
    mom = load_strategy("momentum_top20")
    assert isinstance(mom, PipelineStrategy)
    assert isinstance(mom.selector, TopN)
    assert mom.selector.buffer == 40
    assert isinstance(mom.overlays[0], RegimeFilter)
    assert isinstance(load_strategy("core_satellite"), Blend)


def test_hash_is_stable_and_content_sensitive() -> None:
    a, b = parse_config(dict(MINIMAL)), parse_config(dict(MINIMAL))
    assert config_hash(a) == config_hash(b)
    changed = parse_config({**MINIMAL, "selector": {"type": "top_n", "n": 2}})
    assert config_hash(changed) != config_hash(a)
    assert isinstance(a, PipelineConfig)
    assert a.weighter.type == "equal"  # default
    assert a.schedule == "monthly"


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"name": "Bad Name"}, "lowercase"),
        ({"extra_key": 1}, "Extra inputs"),
        ({"schedule": "hourly"}, "schedule"),
        ({"universe": {"type": "nope"}}, "unknown universe type 'nope'"),
        ({"selector": {"type": "top_n", "n": "lots"}}, "selector"),
        ({"selector": {"type": "top_n", "n": 3, "buffer": 2}}, "buffer >= n"),
        ({"filters": [{"type": "min_price"}]}, r"filters\[0\]"),
        ({"scorer": {"type": "composite", "factors": []}}, "at least one factor"),
    ],
)
def test_config_errors(patch: dict[str, object], msg: str) -> None:
    with pytest.raises(StrategyError, match=msg):
        build_strategy(parse_config({**MINIMAL, **patch}))


def test_blend_inline_and_by_name() -> None:
    cfg = parse_config(
        {
            "name": "mix",
            "blend": [
                {"strategy": "buy_hold_nifty", "weight": 0.5},
                {"strategy": MINIMAL, "weight": 0.5},
            ],
        }
    )
    assert isinstance(cfg, BlendConfig)
    blend = build_strategy(cfg)
    assert isinstance(blend, Blend)
    assert [s.name for s, _ in blend.parts] == ["buy_hold_nifty", "tiny"]
    bad = parse_config(
        {
            "name": "mix",
            "blend": [{"strategy": {**MINIMAL, "universe": {"type": "x"}}, "weight": 1.0}],
        }
    )
    with pytest.raises(StrategyError, match=r"blend\[0\]"):
        build_strategy(bad)


def test_load_from_files(tmp_path: Path) -> None:
    import yaml

    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(MINIMAL))
    assert load_config(f).name == "tiny"
    with pytest.raises(StrategyError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(StrategyError, match="mapping"):
        load_config(lst)
    with pytest.raises(StrategyError, match="unknown strategy"):
        load_config("no_such_builtin")


def test_registry_is_immutable(session_factory: SF) -> None:
    cfg = parse_config(dict(MINIMAL))
    with session_scope(session_factory) as s:
        rec = register_config(s, cfg)
        assert register_config(s, cfg).id == rec.id  # idempotent
        changed = parse_config({**MINIMAL, "selector": {"type": "top_n", "n": 2}})
        with pytest.raises(StrategyError, match="bump the version"):
            register_config(s, changed)
        v2 = parse_config({**MINIMAL, "version": 2, "selector": {"type": "top_n", "n": 2}})
        register_config(s, v2)
        with pytest.raises(StrategyError, match="bump the version"):  # description is content
            register_config(s, parse_config({**MINIMAL, "description": "x"}))
    with session_scope(session_factory) as s:
        assert get_config(s, "tiny").version == 2
        assert get_config(s, "tiny", 1).version == 1
        assert [(n, v) for n, v, _ in list_configs(s)] == [("tiny", 1), ("tiny", 2)]
        with pytest.raises(StrategyError, match="no registered"):
            get_config(s, "ghost")


def test_strategy_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from alphaengine.cli.main import app

    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    r = runner.invoke(app, ["strategy", "list"])
    assert "dual_momentum" in r.output
    assert runner.invoke(app, ["strategy", "validate", "dual_momentum"]).exit_code == 0
    assert "top_n" in runner.invoke(app, ["strategy", "show", "dual_momentum"]).output
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nuniverse: {type: nope}\n")
    assert runner.invoke(app, ["strategy", "validate", str(bad)]).exit_code == 1
    assert "registered" in runner.invoke(app, ["strategy", "register", "fixed_mix"]).output
    assert "fixed_mix v1" in runner.invoke(app, ["strategy", "list"]).output

    # signals on a small synthetic store
    assert "no price data" in runner.invoke(app, ["strategy", "signals", "fixed_mix"]).output
    store = PriceStore(tmp_path / "data" / "curated", "equities")
    store.write(
        make_bars({"NIFTYBEES": [100.0] * 5, "JUNIORBEES": [50.0] * 5, "GOLDBEES": [40.0] * 5})
    )
    cur = tmp_path / "cur.csv"
    cur.write_text("symbol,weight\nniftybees,0.5\n")
    r = runner.invoke(app, ["strategy", "signals", "fixed_mix", "--current", str(cur)])
    assert r.exit_code == 0, r.output
    assert "NIFTYBEES" in r.output
    assert "60.00%" in r.output
    r = runner.invoke(app, ["strategy", "signals", "fixed_mix", "--as-of", "2000-01-01"])
    assert r.exit_code != 0
