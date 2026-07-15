from __future__ import annotations

import pandas as pd
import pytest

from prediction_market_extensions.backtesting._result_policies import (
    apply_binary_settlement_pnl,
    apply_joint_portfolio_settlement_pnl,
)


def test_settlement_pnl_is_not_applied_when_resolution_occurs_after_replay() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -1.25,
            "realized_outcome": 1.0,
            "fill_events": [{"action": "buy", "price": 0.90, "quantity": 25.0, "commission": 0.0}],
            "simulated_through": "2026-04-01T00:00:00+00:00",
            "settlement_observable_time": "2026-04-10T00:00:00+00:00",
        }
    )

    assert result["pnl"] == -1.25
    assert result["settlement_pnl_applied"] is False
    assert "warnings" in result
    assert "mark-to-market PnL" in result["warnings"][0]


def test_settlement_pnl_is_not_applied_when_simulated_through_is_missing() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": 1.25,
            "realized_outcome": 1.0,
            "fill_events": [{"action": "buy", "price": 0.90, "quantity": 25.0, "commission": 0.0}],
            "settlement_observable_time": "2026-04-10T00:00:00+00:00",
        }
    )

    assert result["pnl"] == 1.25
    assert result["settlement_pnl_applied"] is False
    assert "simulated_through is missing" in result["warnings"][0]


def test_settlement_pnl_updates_summary_series_at_resolution_time() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                }
            ],
            "simulated_through": "2026-04-01T00:05:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
            ],
        }
    )

    assert result["pnl"] == pytest.approx(0.22625)
    assert result["equity_series"][-1][0] == "2026-04-01T00:05:00+00:00"
    assert result["equity_series"][-1][1] == pytest.approx(1000.22625)
    assert result["cash_series"][-1][0] == "2026-04-01T00:05:00+00:00"
    assert result["cash_series"][-1][1] == pytest.approx(1000.22625)
    assert result["pnl_series"][-1][0] == "2026-04-01T00:05:00+00:00"
    assert result["pnl_series"][-1][1] == pytest.approx(0.22625)
    assert result["settlement_equity_adjustment"] == pytest.approx(0.25)
    assert result["settlement_cash_adjustment"] == pytest.approx(5.0)


def test_settlement_series_prefers_market_close_when_expiration_precedes_replay() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                }
            ],
            "simulated_through": "2026-04-26T18:05:00+00:00",
            "settlement_observable_time": "2026-04-26T00:00:00+00:00",
            "market_close_time_ns": pd.Timestamp("2026-04-26T18:05:00+00:00").value,
            "equity_series": [
                ("2026-04-26T18:04:30+00:00", 1000.0),
                ("2026-04-26T18:04:59.996000+00:00", 999.97625),
            ],
            "cash_series": [
                ("2026-04-26T18:04:30+00:00", 1000.0),
                ("2026-04-26T18:04:59.996000+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-26T18:04:30+00:00", 0.0),
                ("2026-04-26T18:04:59.996000+00:00", -0.02375),
            ],
        }
    )

    assert result["settlement_series_time"] == "2026-04-26T18:05:00+00:00"
    assert result["equity_series"][-2][1] == pytest.approx(999.97625)
    assert result["equity_series"][-1] == ("2026-04-26T18:05:00+00:00", pytest.approx(1000.22625))
    assert result["cash_series"][-1] == ("2026-04-26T18:05:00+00:00", pytest.approx(1000.22625))
    assert result["pnl_series"][-1] == ("2026-04-26T18:05:00+00:00", pytest.approx(0.22625))
    assert result["settlement_equity_adjustment"] == pytest.approx(0.25)
    assert result["settlement_cash_adjustment"] == pytest.approx(5.0)


def test_joint_portfolio_series_receive_settlement_adjustments() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                }
            ],
            "simulated_through": "2026-04-01T00:05:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
            ],
        }
    )

    results = apply_joint_portfolio_settlement_pnl([result])

    assert results[0]["joint_portfolio_equity_series"][-1][0] == ("2026-04-01T00:05:00+00:00")
    assert results[0]["joint_portfolio_equity_series"][-1][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_cash_series"][-1][0] == ("2026-04-01T00:05:00+00:00")
    assert results[0]["joint_portfolio_cash_series"][-1][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_pnl_series"][-1][0] == "2026-04-01T00:05:00+00:00"
    assert results[0]["joint_portfolio_pnl_series"][-1][1] == pytest.approx(0.22625)


def test_joint_portfolio_equity_carries_cash_payout_after_settlement() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                }
            ],
            "simulated_through": "2026-04-01T00:06:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
                ("2026-04-01T00:06:00+00:00", -4.77375),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
                ("2026-04-01T00:06:00+00:00", -4.77375),
            ],
        }
    )

    results = apply_joint_portfolio_settlement_pnl([result])

    assert results[0]["joint_portfolio_equity_series"][-2][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_equity_series"][-1][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_cash_series"][-1][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_pnl_series"][-2][1] == pytest.approx(0.22625)
    assert results[0]["joint_portfolio_pnl_series"][-1][1] == pytest.approx(0.22625)


def test_joint_portfolio_settlement_keeps_marked_position_single_counted() -> None:
    # Replay runners hold positions to resolution: engine equity KEEPS the
    # position marked (~$1) through the window end and settlement is applied
    # post-hoc. The post-settlement delta must then be the small mark->
    # settlement correction, NOT the full cash payout on top of the
    # surviving mark (which inflated equity curves by the position value).
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                    "timestamp": "2026-04-01T00:04:50+00:00",
                }
            ],
            "simulated_through": "2026-04-01T00:06:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "market_close_time_ns": pd.Timestamp("2026-04-01T00:05:00+00:00").value,
            "price_series": [
                ("2026-04-01T00:04:30+00:00", 0.95),
                ("2026-04-01T00:05:00+00:00", 0.99),
            ],
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 1000.17625),
                ("2026-04-01T00:06:00+00:00", 1000.17625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.17625),
                ("2026-04-01T00:06:00+00:00", 0.17625),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                # position marked at 0.99 into settlement and KEPT after it
                ("2026-04-01T00:05:00+00:00", 1000.17625),
                ("2026-04-01T00:06:00+00:00", 1000.17625),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.17625),
                ("2026-04-01T00:06:00+00:00", 0.17625),
            ],
        }
    )

    # settlement 5.0 - 0.02375 commission = 4.97625 pnl; mark at 0.99 ->
    # equity adj = 0.05, cash adj = 5.0
    assert result["settlement_equity_adjustment"] == pytest.approx(0.05)
    assert result["settlement_cash_adjustment"] == pytest.approx(5.0)

    results = apply_joint_portfolio_settlement_pnl([result])

    equity = dict(results[0]["joint_portfolio_equity_series"])
    # settled account value = 1000 - 0.02375 + (5.0 - 4.75) = 1000.22625
    assert equity["2026-04-01T00:05:00+00:00"] == pytest.approx(1000.22625)
    assert equity["2026-04-01T00:06:00+00:00"] == pytest.approx(1000.22625)
    assert max(value for _, value in results[0]["joint_portfolio_equity_series"]) == pytest.approx(
        1000.22625
    )
    cash = dict(results[0]["joint_portfolio_cash_series"])
    assert cash["2026-04-01T00:06:00+00:00"] == pytest.approx(1000.22625)


def test_joint_portfolio_settlement_pins_kept_position_despite_post_mark_drift() -> None:
    # The engine keeps the position marked after resolution and the mark then
    # FADES (post-resolution noise). The settled value is frozen: every point
    # after the settlement timestamp must equal the settled account value,
    # cancelling the drift point by point (a constant correction computed at
    # the settlement moment would leak the drift into the curve).
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.02375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                    "timestamp": "2026-04-01T00:04:50+00:00",
                }
            ],
            "simulated_through": "2026-04-01T00:07:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "market_close_time_ns": pd.Timestamp("2026-04-01T00:05:00+00:00").value,
            "price_series": [
                ("2026-04-01T00:04:30+00:00", 0.95),
                ("2026-04-01T00:05:00+00:00", 0.99),
                ("2026-04-01T00:06:00+00:00", 0.60),
                ("2026-04-01T00:07:00+00:00", 0.90),
            ],
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 1000.17625),
                ("2026-04-01T00:06:00+00:00", 998.22625),
                ("2026-04-01T00:07:00+00:00", 999.72625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
                ("2026-04-01T00:07:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.17625),
                ("2026-04-01T00:06:00+00:00", -1.77375),
                ("2026-04-01T00:07:00+00:00", -0.27375),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 1000.17625),
                # mark fades to 0.60 then recovers to 0.90 AFTER resolution
                ("2026-04-01T00:06:00+00:00", 998.22625),
                ("2026-04-01T00:07:00+00:00", 999.72625),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
                ("2026-04-01T00:07:00+00:00", 995.22625),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.17625),
                ("2026-04-01T00:06:00+00:00", -1.77375),
                ("2026-04-01T00:07:00+00:00", -0.27375),
            ],
        }
    )

    results = apply_joint_portfolio_settlement_pnl([result])

    equity = dict(results[0]["joint_portfolio_equity_series"])
    assert equity["2026-04-01T00:05:00+00:00"] == pytest.approx(1000.22625)
    assert equity["2026-04-01T00:06:00+00:00"] == pytest.approx(1000.22625)
    assert equity["2026-04-01T00:07:00+00:00"] == pytest.approx(1000.22625)
    pnl = dict(results[0]["joint_portfolio_pnl_series"])
    assert pnl["2026-04-01T00:06:00+00:00"] == pytest.approx(0.22625)
    assert pnl["2026-04-01T00:07:00+00:00"] == pytest.approx(0.22625)


def test_joint_portfolio_settlement_does_not_double_count_stale_position_value() -> None:
    result = apply_binary_settlement_pnl(
        {
            "pnl": -4.77375,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.95,
                    "quantity": 5.0,
                    "commission": 0.02375,
                    "timestamp": "2026-04-01T00:04:50+00:00",
                }
            ],
            "simulated_through": "2026-04-01T00:06:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "market_close_time_ns": pd.Timestamp("2026-04-01T00:05:00+00:00").value,
            "price_series": [
                ("2026-04-01T00:04:30+00:00", 0.95),
                ("2026-04-01T00:05:00+00:00", 0.95),
            ],
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -4.77375),
                ("2026-04-01T00:06:00+00:00", -4.77375),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.97625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 995.22625),
                ("2026-04-01T00:06:00+00:00", 995.22625),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", -0.02375),
                ("2026-04-01T00:06:00+00:00", -4.77375),
            ],
        }
    )

    assert result["settlement_equity_adjustment"] == pytest.approx(0.25)
    assert result["settlement_cash_adjustment"] == pytest.approx(5.0)

    results = apply_joint_portfolio_settlement_pnl([result])

    assert results[0]["joint_portfolio_equity_series"][-2][1] == pytest.approx(1000.22625)
    assert results[0]["joint_portfolio_equity_series"][-1][1] == pytest.approx(1000.22625)
    assert max(value for _, value in results[0]["joint_portfolio_equity_series"]) == pytest.approx(
        1000.22625
    )


def test_joint_portfolio_settlement_pins_tail_when_engine_drops_position_late() -> None:
    # Observed in pilot replays (april-28-may-5 week): the engine keeps the
    # settled position marked PAST the settlement timestamp (so the boundary
    # discriminator picks the kept-marks per-point path) and then drops it to
    # cash-only equity partway through the remaining window. Neither the
    # per-point correction (assumes marks survive to the end) nor the flat
    # cash restore (assumes the drop is at the boundary) covers that; once
    # every filled result is settled the tail must simply be pinned flat at
    # the settled account value.
    result = apply_binary_settlement_pnl(
        {
            "pnl": -0.76875,
            "realized_outcome": 1.0,
            "fill_events": [
                {
                    "action": "buy",
                    "side": "yes",
                    "price": 0.15,
                    "quantity": 5.0,
                    "commission": 0.01875,
                    "timestamp": "2026-04-01T00:04:50+00:00",
                }
            ],
            "simulated_through": "2026-04-01T00:07:00+00:00",
            "settlement_observable_time": "2026-04-01T00:05:00+00:00",
            "market_close_time_ns": pd.Timestamp("2026-04-01T00:05:00+00:00").value,
            "price_series": [
                ("2026-04-01T00:04:30+00:00", 0.15),
                ("2026-04-01T00:05:00+00:00", 0.30),
                ("2026-04-01T00:06:00+00:00", 0.60),
                ("2026-04-01T00:07:00+00:00", 0.60),
            ],
            "equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 1000.73125),
                ("2026-04-01T00:06:00+00:00", 1002.23125),
                ("2026-04-01T00:07:00+00:00", 999.23125),
            ],
            "cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.23125),
                ("2026-04-01T00:06:00+00:00", 999.23125),
                ("2026-04-01T00:07:00+00:00", 999.23125),
            ],
            "pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.73125),
                ("2026-04-01T00:06:00+00:00", 2.23125),
                ("2026-04-01T00:07:00+00:00", -0.76875),
            ],
            "joint_portfolio_equity_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                # mark carried through settlement and beyond...
                ("2026-04-01T00:05:00+00:00", 1000.73125),
                ("2026-04-01T00:06:00+00:00", 1002.23125),
                # ...then the position leaves account equity with NO payout
                ("2026-04-01T00:07:00+00:00", 999.23125),
            ],
            "joint_portfolio_cash_series": [
                ("2026-04-01T00:04:30+00:00", 1000.0),
                ("2026-04-01T00:05:00+00:00", 999.23125),
                ("2026-04-01T00:06:00+00:00", 999.23125),
                ("2026-04-01T00:07:00+00:00", 999.23125),
            ],
            "joint_portfolio_pnl_series": [
                ("2026-04-01T00:04:30+00:00", 0.0),
                ("2026-04-01T00:05:00+00:00", 0.73125),
                ("2026-04-01T00:06:00+00:00", 2.23125),
                ("2026-04-01T00:07:00+00:00", -0.76875),
            ],
        }
    )

    # settlement 5.0 - 0.75 cost - 0.01875 commission = 4.23125 pnl;
    # mark at 0.30 -> equity adj = 3.5, cash adj = 5.0
    assert result["settlement_equity_adjustment"] == pytest.approx(3.5)
    assert result["settlement_cash_adjustment"] == pytest.approx(5.0)

    results = apply_joint_portfolio_settlement_pnl([result])

    # settled account value = 1000 + 4.23125, frozen from settlement on
    equity = dict(results[0]["joint_portfolio_equity_series"])
    assert equity["2026-04-01T00:05:00+00:00"] == pytest.approx(1004.23125)
    assert equity["2026-04-01T00:06:00+00:00"] == pytest.approx(1004.23125)
    assert equity["2026-04-01T00:07:00+00:00"] == pytest.approx(1004.23125)
    pnl = dict(results[0]["joint_portfolio_pnl_series"])
    assert pnl["2026-04-01T00:06:00+00:00"] == pytest.approx(4.23125)
    assert pnl["2026-04-01T00:07:00+00:00"] == pytest.approx(4.23125)


def test_final_settlement_timestamp_requires_every_filled_result_settled() -> None:
    from prediction_market_extensions.backtesting._result_policies import (
        _final_settlement_timestamp_if_fully_settled,
    )

    fill = {
        "action": "buy",
        "side": "yes",
        "price": 0.5,
        "quantity": 1.0,
        "timestamp": "2026-04-01T00:00:00+00:00",
    }
    settled_early = {
        "fill_events": [fill],
        "settlement_pnl_applied": True,
        "settlement_series_time": "2026-04-01T00:05:00+00:00",
    }
    settled_late = {
        "fill_events": [fill],
        "settlement_pnl_applied": True,
        "settlement_series_time": "2026-04-02T00:00:00+00:00",
    }
    unsettled = {"fill_events": [fill], "settlement_pnl_applied": False}
    no_fills = {"fill_events": [], "settlement_pnl_applied": False}

    # Fully settled: the LATEST settlement wins; no-fill stragglers are inert.
    timestamp = _final_settlement_timestamp_if_fully_settled(
        [settled_early, settled_late, no_fills]
    )
    assert timestamp == pd.Timestamp("2026-04-02T00:00:00+00:00")

    # A filled-but-unsettled result blocks the pin (mixed window).
    assert _final_settlement_timestamp_if_fully_settled([settled_early, unsettled]) is None
