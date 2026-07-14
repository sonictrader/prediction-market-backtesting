# Added on the hades branch: bounded PMXT smoke runner for environment
# verification. Derived from polymarket_pmxt_book_100_replay_runner.py.
# Distributed under the GNU Lesser General Public License Version 3.0 or later.
# See the repository NOTICE file for provenance and licensing scope.

from __future__ import annotations

from decimal import Decimal

if __package__ in {None, ""}:
    from _script_helpers import ensure_repo_root
else:
    from ._script_helpers import ensure_repo_root

ensure_repo_root(__file__)

# Deliberately small: a cold PMXT archive path downloads full hourly files
# (~225-450 MB each), so the smoke window stays at a handful of hours.
WINDOW_START = "2026-07-13T00:00:00Z"
WINDOW_END = "2026-07-13T05:59:59.999999999Z"
WINDOW_START_NS = 1783900800000000000
WINDOW_END_NS = 1783922399999999999

SMOKE_MARKET_SLUGS = (
    "will-china-invade-taiwan-before-2027",
    "will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568",
    "will-elon-musk-win-the-2028-us-presidential-election",
)

PMXT_LOCAL_MIRROR = "local:C:/hades-data/pmxt-mirror"


def run() -> None:
    from prediction_market_extensions.backtesting._execution_config import (
        ExecutionModelConfig,
        StaticLatencyConfig,
    )
    from prediction_market_extensions.backtesting._experiments import (
        build_replay_experiment,
        run_experiment,
    )
    from prediction_market_extensions.backtesting._prediction_market_backtest import (
        MarketReportConfig,
    )
    from prediction_market_extensions.backtesting._prediction_market_runner import (
        MarketDataConfig,
    )
    from prediction_market_extensions.backtesting._replay_specs import BookReplay
    from prediction_market_extensions.backtesting._timing_harness import timing_harness
    from prediction_market_extensions.backtesting.data_sources import Book, Polymarket, PMXT

    replays = tuple(
        BookReplay(
            market_slug=slug,
            token_index=0,
            start_time=WINDOW_START,
            end_time=WINDOW_END,
            metadata={
                "sim_label": slug,
                "replay_window_start_ns": WINDOW_START_NS,
                "replay_window_end_ns": WINDOW_END_NS,
            },
        )
        for slug in SMOKE_MARKET_SLUGS
    )

    @timing_harness
    def _run() -> None:
        run_experiment(
            build_replay_experiment(
                name="polymarket_pmxt_book_smoke_runner",
                description=(
                    "Bounded PMXT book smoke backtest over 3 Polymarket markets / 6 hours"
                ),
                data=MarketDataConfig(
                    platform=Polymarket,
                    data_type=Book,
                    vendor=PMXT,
                    sources=(
                        PMXT_LOCAL_MIRROR,
                        "archive:r2v2.pmxt.dev",
                        "archive:r2.pmxt.dev",
                    ),
                ),
                replays=replays,
                strategy_configs=[
                    {
                        "strategy_path": "strategies:BookMicropriceImbalanceStrategy",
                        "config_path": "strategies:BookMicropriceImbalanceConfig",
                        "config": {
                            "trade_size": Decimal(5),
                            "depth_levels": 3,
                            "entry_imbalance": 0.62,
                            "exit_imbalance": 0.48,
                            "min_microprice_edge": 0.0015,
                            "max_spread": 0.08,
                            "max_entry_price": 0.95,
                            "max_expected_slippage": 0.02,
                            "min_holding_updates": 0,
                            "reentry_cooldown_updates": 0,
                            "min_holding_seconds": 30.0,
                            "reentry_cooldown_seconds": 60.0,
                            "take_profit": 0.02,
                            "stop_loss": 0.025,
                        },
                    }
                ],
                initial_cash=1_000.0,
                probability_window=30,
                min_book_events=25,
                min_price_range=0.0,
                execution=ExecutionModelConfig(
                    queue_position=True,
                    latency_model=StaticLatencyConfig(
                        base_latency_ms=75.0,
                        insert_latency_ms=10.0,
                        update_latency_ms=5.0,
                        cancel_latency_ms=5.0,
                    ),
                ),
                report=MarketReportConfig(
                    count_key="book_events",
                    count_label="Book Events",
                    pnl_label="PnL (pUSD)",
                    market_key="sim_label",
                    summary_report=True,
                    summary_report_path=(
                        "output/polymarket_pmxt_book_smoke_runner_joint_portfolio.html"
                    ),
                    summary_plot_panels=(
                        "total_equity",
                        "equity",
                        "market_pnl",
                        "periodic_pnl",
                        "yes_price",
                        "allocation",
                        "total_drawdown",
                        "drawdown",
                        "total_rolling_sharpe",
                        "rolling_sharpe",
                        "total_cash_equity",
                        "cash_equity",
                        "monthly_returns",
                        "total_brier_advantage",
                        "brier_advantage",
                    ),
                ),
                empty_message=("No PMXT smoke-runner windows met the book requirements."),
                partial_message=("Completed {completed} of {total} PMXT smoke replays."),
                return_summary_series=True,
            )
        )

    _run()


if __name__ == "__main__":
    run()
