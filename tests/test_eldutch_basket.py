# Added on the hades branch: mechanical sanity checks for the vanilla
# EL-DUTCH port (mapping doc: mapping-eldutch-nautilus-2026-07-14.md).
# Distributed under the GNU Lesser General Public License Version 3.0 or later.
# See the repository NOTICE file for provenance and licensing scope.

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from nautilus_trader.model.identifiers import InstrumentId

from strategies.eldutch_basket import (
    ElDutchBucketConfig,
    ElDutchEventPortfolio,
    _fee_for_leg,
    considered_price,
)


def _level(price: float, size: float) -> SimpleNamespace:
    return SimpleNamespace(price=price, size=size)


def _instrument_id(value: str) -> InstrumentId:
    return InstrumentId.from_str(f"{value}.POLYMARKET")


def _stub_bucket(
    *,
    instrument: str,
    commit: float,
    shares: float = 0.0,
    avg_entry: float | None = None,
    best_bid: float | None = None,
    fee_rate: float = 0.05,
    flash_pct: float = 0.10,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            instrument_id=_instrument_id(instrument),
            fee_rate=fee_rate,
            flash_profit_pct=flash_pct,
        ),
        is_fresh=False,
        commit_price=lambda: commit,
        position_size=lambda: shares,
        avg_entry_price=lambda: avg_entry,
        last_best_bid=lambda: best_bid,
        flash_calls=[],
        liquidate_for_flash=lambda: None,
    )


# ---- fee curve ----


def test_fee_for_leg_matches_polymarket_curve() -> None:
    assert _fee_for_leg(0.5, 0.05) == pytest.approx(0.05 * 0.5 * 0.5)
    assert _fee_for_leg(0.2, 0.05) == pytest.approx(0.05 * 0.2 * 0.8)
    assert _fee_for_leg(0.0, 0.05) == 0.0
    assert _fee_for_leg(1.0, 0.05) == 0.0


# ---- considered-bid cumulative walk (SM _cumulative_side + 270.3 bar) ----


def test_considered_price_walks_past_dust() -> None:
    # $250 dollar bar binds (touch 0.25 -> share bar 1000*0.25=$250 too);
    # the first two levels are dust, the third crosses the bar.
    levels = [_level(0.25, 100.0), _level(0.24, 200.0), _level(0.23, 2000.0)]
    liq, price, status = considered_price(levels, dollar_bar=250.0, share_bar=1000.0)
    assert status == "ok"
    assert price == pytest.approx(0.23)
    assert liq > 250.0


def test_considered_price_share_bar_binds_on_cheap_tails() -> None:
    # touch 0.01 -> effective bar = min(250, 1000*0.01=$10): the touch level's
    # $11 alone crosses it, so the walk stays at the touch instead of going
    # 10-20 ticks deep chasing the $250 dollar bar.
    levels = [_level(0.010, 1_100.0), _level(0.009, 50_000.0)]
    _liq, price, status = considered_price(levels, dollar_bar=250.0, share_bar=1000.0)
    assert status == "ok"
    assert price == pytest.approx(0.010)


def test_considered_price_insufficient_and_empty() -> None:
    levels = [_level(0.25, 10.0)]
    _liq, price, status = considered_price(levels, dollar_bar=250.0, share_bar=1000.0)
    assert status == "insufficient"
    assert price == pytest.approx(0.25)
    assert considered_price([], dollar_bar=250.0, share_bar=1000.0)[2] == "none"


# ---- fee-net entry gate (SM _entry_net_gate formula) ----


def test_entry_gate_blocks_at_one_dollar_net() -> None:
    portfolio = ElDutchEventPortfolio("gate-test")
    committed = _stub_bucket(instrument="A", commit=0.50)
    candidate = _stub_bucket(instrument="B", commit=0.0)
    portfolio._buckets = {
        committed.config.instrument_id: committed,
        candidate.config.instrument_id: candidate,
    }
    # 0.50 + fee(0.50) = 0.5125; candidate 0.40 + fee(0.40)=0.412 -> 0.9245 OK
    assert portfolio.entry_gate_ok(candidate, 0.40, max_net=1.0)
    # candidate 0.48 + fee = 0.492{48} -> 1.0057 >= 1.0 BLOCKED
    assert not portfolio.entry_gate_ok(candidate, 0.48, max_net=1.0)


def test_entry_gate_substitutes_candidate_price() -> None:
    portfolio = ElDutchEventPortfolio("gate-sub")
    candidate = _stub_bucket(instrument="C", commit=0.99)  # stale commit ignored
    portfolio._buckets = {candidate.config.instrument_id: candidate}
    assert portfolio.entry_gate_ok(candidate, 0.10, max_net=1.0)


# ---- flash profit (SM check_flash_profit fee-net math + throttle) ----


def _flash_portfolio(bid: float) -> tuple[ElDutchEventPortfolio, list[str]]:
    portfolio = ElDutchEventPortfolio("flash-test")
    calls: list[str] = []
    held = _stub_bucket(instrument="H", commit=0.20, shares=50.0, avg_entry=0.20, best_bid=bid)
    held.liquidate_for_flash = lambda: calls.append("H")
    empty = _stub_bucket(instrument="E", commit=0.15)
    empty.liquidate_for_flash = lambda: calls.append("E")
    portfolio._buckets = {
        held.config.instrument_id: held,
        empty.config.instrument_id: empty,
    }
    return portfolio, calls


def test_flash_triggers_on_fee_net_roi_and_latches() -> None:
    # entry 0.20, bid 0.24: net = 0.24 - 0.05*0.24*0.76 = 0.23088 -> ROI 15.4%
    portfolio, calls = _flash_portfolio(bid=0.24)
    portfolio.maybe_flash(ts_event_ns=100_000_000_000)
    assert calls == ["H", "E"] or sorted(calls) == ["E", "H"]
    assert portfolio.liquidating
    # latched: further checks never re-fire
    portfolio.maybe_flash(ts_event_ns=200_000_000_000)
    assert len(calls) == 2


def test_flash_respects_throttle_and_threshold() -> None:
    # entry 0.20, bid 0.21: net ROI ~= 0.9% < 10% -> no trigger
    portfolio, calls = _flash_portfolio(bid=0.21)
    portfolio.maybe_flash(ts_event_ns=100_000_000_000)
    assert not portfolio.liquidating and not calls
    # bid improves but the 60s throttle absorbs the next check
    for bucket in portfolio._buckets.values():
        bucket.last_best_bid = lambda: 0.30
    portfolio.maybe_flash(ts_event_ns=100_000_000_000 + int(30e9))
    assert not portfolio.liquidating
    # past the throttle it fires
    portfolio.maybe_flash(ts_event_ns=100_000_000_000 + int(61e9))
    assert portfolio.liquidating


# ---- config validation ----


def test_config_validation_rejects_bad_dials() -> None:
    base = dict(
        instrument_id=_instrument_id("X"),
        event_key="ev",
    )
    ElDutchBucketConfig(**base)  # defaults valid
    with pytest.raises(ValueError):
        ElDutchBucketConfig(**base, target_shares=Decimal(0))
    with pytest.raises(ValueError):
        ElDutchBucketConfig(**base, stop_loss_pct=1.5)
    with pytest.raises(ValueError):
        ElDutchBucketConfig(**base, fee_rate=-0.1)


def test_registry_returns_same_portfolio_per_event() -> None:
    a = ElDutchEventPortfolio.for_event("same-key")
    b = ElDutchEventPortfolio.for_event("same-key")
    c = ElDutchEventPortfolio.for_event("other-key")
    assert a is b and a is not c
