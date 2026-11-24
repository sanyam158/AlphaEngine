from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphaengine.core.errors import OrderError
from alphaengine.core.types import Side
from alphaengine.data.prices import grandfather_fmv_lookup
from alphaengine.data.store import PriceStore
from alphaengine.market.costs import CostModel
from alphaengine.market.tax import (
    CarriedLoss,
    compute_fy_tax,
    compute_tax_history,
    financial_year,
    regime_on,
    taxable_gain,
)
from alphaengine.portfolio.lots import Disposal, LotBook, add_months, is_long_term
from tests.helpers import make_bars


def disp(
    qty: int, cost: float, proceeds: float, acquired: date, sold: date, sym: str = "X"
) -> Disposal:
    return Disposal(sym, qty, acquired, sold, D(str(cost)), D(str(proceeds)), D("0"), D("0"))


class TestHoldingPeriod:
    def test_more_than_twelve_months(self) -> None:
        assert not is_long_term(date(2023, 1, 10), date(2024, 1, 10))
        assert is_long_term(date(2023, 1, 10), date(2024, 1, 11))
        assert add_months(date(2020, 2, 29), 12) == date(2021, 2, 28)
        assert add_months(date(2024, 11, 30), 3) == date(2025, 2, 28)

    def test_financial_year_and_regimes(self) -> None:
        assert financial_year(date(2025, 3, 31)) == 2024
        assert financial_year(date(2025, 4, 1)) == 2025
        assert regime_on(date(2024, 7, 22)).stcg == D("0.15")
        assert regime_on(date(2024, 7, 23)).stcg == D("0.20")
        assert regime_on(date(2017, 1, 1)).ltcg is None
        with pytest.raises(ValueError):  # noqa: PT011
            regime_on(date(2000, 1, 1))


class TestLotBook:
    def test_fifo_partial_lots_and_charge_allocation(self) -> None:
        m = CostModel()
        book = LotBook()
        b1 = m.charges(Side.BUY, 10, D("100"), date(2024, 1, 1))
        book.buy("A", 10, D("100"), date(2024, 1, 1), b1)
        book.buy("A", 10, D("120"), date(2025, 1, 1))
        assert book.lots("A")[0].cost == D("1000") + b1.tax_deductible
        sc = m.charges(Side.SELL, 15, D("150"), date(2025, 3, 1))
        ds = book.sell("A", 15, D("150"), date(2025, 3, 1), sc)
        assert [(d.quantity, d.acquired, d.long_term) for d in ds] == [
            (10, date(2024, 1, 1), True),
            (5, date(2025, 1, 1), False),
        ]
        assert sum(d.sell_expenses for d in ds) == sc.tax_deductible
        assert sum(d.stt for d in ds) == sc.stt
        assert ds[1].cost == D("600")
        assert book.quantity("A") == 5
        assert book.cost_basis("A") == D("600")
        assert book.symbols() == ["A"]
        assert len(list(book)) == 1

    def test_split_and_bonus(self) -> None:
        book = LotBook()
        book.buy("A", 10, D("100"), date(2023, 1, 2))
        book.apply_split("A", 5)
        lot = book.lots("A")[0]
        assert (lot.quantity, lot.cost, lot.acquired) == (50, D("1000"), date(2023, 1, 2))
        new = book.apply_bonus("A", 2.0, date(2024, 6, 1))  # 1:1 bonus
        assert [(n.quantity, n.cost, n.source) for n in new] == [(50, D("0"), "BONUS")]
        assert book.quantity("A") == 100
        # FIFO sells the original lot first; the bonus lot keeps its own date and zero cost
        ds = book.sell("A", 60, D("30"), date(2024, 9, 1))
        assert [(d.quantity, d.cost, d.lot_source) for d in ds] == [
            (50, D("1000"), "BUY"),
            (10, D("0"), "BONUS"),
        ]
        assert not ds[1].long_term
        with pytest.raises(ValueError):  # noqa: PT011
            book.apply_bonus("A", 1.0, date(2024, 6, 1))
        with pytest.raises(ValueError):  # noqa: PT011
            book.apply_split("A", 0)

    def test_long_term_helpers_and_preview(self) -> None:
        book = LotBook()
        book.buy("A", 10, D("100"), date(2024, 1, 1))
        book.buy("A", 5, D("100"), date(2025, 6, 1))
        on = date(2025, 8, 1)
        assert book.long_term_quantity("A", on) == 10
        assert book.fifo_long_term_quantity("A", on) == 10
        assert book.next_long_term_date("A", on) == date(2026, 6, 1)
        assert book.next_long_term_date("A", date(2026, 7, 1)) is None
        preview = book.preview_sell("A", 12, D("110"), on)
        assert len(preview) == 2
        assert book.quantity("A") == 15  # unchanged

    def test_validation(self) -> None:
        book = LotBook()
        with pytest.raises(OrderError):
            book.buy("A", 0, D("1"), date(2024, 1, 1))
        with pytest.raises(OrderError, match="only 0 held"):
            book.sell("A", 1, D("1"), date(2024, 1, 1))
        with pytest.raises(OrderError):
            book.sell("A", 1, D("0"), date(2024, 1, 1))

    @given(
        st.lists(
            st.tuples(st.booleans(), st.integers(1, 500), st.decimals("1", "5000", places=2)),
            min_size=1,
            max_size=30,
        )
    )
    def test_cost_and_quantity_conservation(self, events: list[tuple[bool, int, D]]) -> None:
        m = CostModel()
        book = LotBook()
        bought_cost = D("0")
        disposed_cost = D("0")
        day = date(2024, 1, 1)
        for i, (is_buy, qty, price) in enumerate(events):
            d = add_months(day, i)
            if is_buy:
                c = m.charges(Side.BUY, qty, price, d)
                bought_cost += price * qty + c.tax_deductible
                book.buy("A", qty, price, d, c)
            else:
                q = min(qty, book.quantity("A"))
                if q == 0:
                    continue
                c = m.charges(Side.SELL, q, price, d)
                ds = book.sell("A", q, price, d, c)
                assert sum(x.quantity for x in ds) == q
                assert sum(x.gross_proceeds for x in ds) == price * q
                assert sum(x.sell_expenses for x in ds) == c.tax_deductible
                disposed_cost += sum((x.cost for x in ds), D("0"))
        assert abs(bought_cost - disposed_cost - book.cost_basis("A")) < D("1e-9")


class TestTax:
    def test_stcg_fy2025(self) -> None:
        res = compute_fy_tax(
            [disp(100, 100_000, 120_000, date(2025, 5, 1), date(2025, 9, 1))], 2025
        )
        assert res.stcg == {D("0.20"): D("20000")}
        assert (res.tax, res.cess, res.total_tax) == (D("4000"), D("160"), D("4160"))

    def test_ltcg_with_exemption(self) -> None:
        res = compute_fy_tax(
            [disp(1000, 1_000_000, 1_200_000, date(2023, 1, 2), date(2025, 6, 2))], 2025
        )
        assert res.exemption_used == D("125000")
        assert res.taxable_ltcg == D("75000")
        assert res.tax == D("9375")
        assert res.cess == D("375")

    def test_fy2024_rate_change_and_loss_to_highest_rate(self) -> None:
        a = date(2024, 3, 1)
        ds = [
            disp(1, 100_000, 110_000, a, date(2024, 6, 1)),  # STCG 10k @15%
            disp(1, 100_000, 110_000, a, date(2024, 8, 1)),  # STCG 10k @20%
            disp(1, 100_000, 95_000, a, date(2024, 9, 1)),  # STCL 5k
        ]
        res = compute_fy_tax(ds, 2024)
        assert res.stcg == {D("0.15"): D("10000"), D("0.20"): D("5000")}
        assert res.tax == D("2500")
        assert res.losses_used == D("5000")

    def test_set_off_rules_and_carry_forward(self) -> None:
        old = date(2020, 1, 1)
        ds = [
            disp(1, 100_000, 300_000, old, date(2025, 5, 1)),  # LTCG 200k
            disp(1, 100_000, 50_000, date(2025, 4, 2), date(2025, 6, 1)),  # STCL 50k -> vs LTCG
            disp(1, 100_000, 10_000, old, date(2025, 7, 1)),  # LTCL 90k -> vs LTCG
        ]
        res = compute_fy_tax(ds, 2025)
        # 200k - 50k - 90k = 60k LTCG, fully covered by the 1.25L exemption
        assert res.taxable_ltcg == 0
        assert res.exemption_used == D("60000")
        assert res.tax == 0
        # An LT loss can't offset STCG: it carries forward
        ds2 = [
            disp(1, 100_000, 150_000, date(2025, 4, 2), date(2025, 6, 1)),  # STCG 50k
            disp(1, 100_000, 70_000, old, date(2025, 7, 1)),  # LTCL 30k
        ]
        res2 = compute_fy_tax(ds2, 2025)
        assert res2.taxable_stcg == D("50000")
        assert res2.carried_forward == [CarriedLoss("LT", D("30000"), 2025)]

    def test_carried_losses_used_and_expire(self) -> None:
        ds = [disp(1, 100_000, 300_000, date(2020, 1, 1), date(2025, 5, 1))]  # LTCG 200k
        carried = [
            CarriedLoss("LT", D("20000"), 2016),  # 9 years old: expired
            CarriedLoss("ST", D("30000"), 2020),
            CarriedLoss("LT", D("10000"), 2023),
        ]
        res = compute_fy_tax(ds, 2025, carried)
        assert res.losses_used == D("40000")
        assert res.taxable_ltcg == D("35000")  # 200k - 40k - 125k
        assert res.carried_forward == []

    def test_pre_2018_ltcg_exempt_and_grandfathering(self) -> None:
        exempt = compute_fy_tax([disp(10, 1000, 5000, date(2010, 1, 1), date(2016, 1, 1))], 2015)
        assert exempt.exempt_ltcg == D("4000")
        assert exempt.tax == 0

        fmv = {"X": D("300")}.get
        d_low = disp(10, 1000, 2500, date(2015, 1, 1), date(2019, 1, 1))  # sold below FMV
        assert taxable_gain(d_low, fmv) == 0
        d_high = disp(10, 1000, 4000, date(2015, 1, 1), date(2019, 1, 1))
        assert taxable_gain(d_high, fmv) == D("1000")  # cost stepped up to FMV 3000
        assert taxable_gain(d_high, None) == D("3000")
        assert taxable_gain(d_high, lambda s: None) == D("3000")
        after_cutoff = disp(10, 1000, 4000, date(2018, 2, 1), date(2019, 6, 1))
        assert taxable_gain(after_cutoff, fmv) == D("3000")

    def test_history_carries_losses(self) -> None:
        ds = [
            disp(1, 100_000, 60_000, date(2023, 5, 1), date(2023, 6, 1)),  # FY23 STCL 40k
            disp(1, 100_000, 150_000, date(2025, 5, 1), date(2025, 6, 1)),  # FY25 STCG 50k
        ]
        hist = compute_tax_history(ds)
        assert sorted(hist) == [2023, 2024, 2025]
        assert hist[2023].carried_forward == [CarriedLoss("ST", D("40000"), 2023)]
        assert hist[2025].taxable_stcg == D("10000")
        assert hist[2025].tax == D("2000")
        assert compute_tax_history([]) == {}


def test_grandfather_fmv_lookup_adjusts_for_later_splits(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = PriceStore(tmp_path)
    bars = make_bars({"AAA": [100.0], "BBB": [50.0], "CCC": [10.0]}, start="2018-01-31")
    store.write(bars)
    acts = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "ex_date": date(2020, 1, 1),
                "kind": "SPLIT",
                "ratio": 5.0,
                "amount": None,
            },
            {
                "symbol": "BBB",
                "ex_date": date(2021, 1, 1),
                "kind": "BONUS",
                "ratio": 2.0,
                "amount": None,
            },
            {
                "symbol": "CCC",
                "ex_date": date(2022, 1, 1),
                "kind": "SPLIT_OR_BONUS",
                "ratio": 2.0,
                "amount": None,
            },
        ]
    )
    f = grandfather_fmv_lookup(store, acts)
    assert f("AAA") == D("101") / 5  # high = close * 1.01 in make_bars
    assert f("BBB") == D("50.5")
    assert f("CCC") is None  # unclassified: no grandfathering (conservative)
    assert f("ZZZ") is None
