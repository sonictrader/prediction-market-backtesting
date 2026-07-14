# Added on the hades branch: vanilla EL-DUTCH pilot runner (270.10).
# Replays precomputed basket selections (pilot_select.py in the 1337-HADES
# monorepo) through the ElDutchBucketStrategy on real PMXT L2 books, one
# engine per weekly event, and writes per-event HTML plus an aggregate JSON.
#
# Multiple events run in one subprocess per event: the Nautilus Rust logging
# singleton panics when a second BacktestEngine initializes in the same
# process, so the parent fans events out to child processes (the same
# pattern the repo optimizer uses) and merges the per-event result files.
# Distributed under the GNU Lesser General Public License Version 3.0 or later.
# See the repository NOTICE file for provenance and licensing scope.

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    _HELPER_PATH = Path(__file__).resolve().parents[1] / "_script_helpers.py"
    _SPEC = importlib.util.spec_from_file_location("_script_helpers", _HELPER_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"Unable to load script helper from {_HELPER_PATH}")
    _HELPER = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_HELPER)
    ensure_repo_root = _HELPER.ensure_repo_root
else:
    from backtests._script_helpers import ensure_repo_root

REPO_ROOT = ensure_repo_root(__file__)

DEFAULT_SELECTION_DIR = (
    "C:/Users/ivo/Desktop/vscode/cline/1337-HADES-mono/memory-bank/analysis/eldutch/data/pilot"
)
SELECTION_DIR = Path(os.environ.get("ELDUTCH_SELECTION_DIR", DEFAULT_SELECTION_DIR))
WINDOW_DAY = int(os.environ.get("ELDUTCH_WINDOW_DAY", "3"))
EVENTS_FILTER = {
    slug.strip() for slug in os.environ.get("ELDUTCH_EVENTS", "").split(",") if slug.strip()
}
LATENCY_MS = float(os.environ.get("ELDUTCH_LATENCY_MS", "75"))
USE_STOP_LOSS = os.environ.get("ELDUTCH_USE_STOP_LOSS", "1").strip() not in {"0", "false"}
USE_FLASH = os.environ.get("ELDUTCH_USE_FLASH", "1").strip() not in {"0", "false"}
TARGET_SHARES = float(os.environ.get("TARGET_SHARES", "50"))
CASH_MULT = float(os.environ.get("ELDUTCH_INITIAL_CASH_MULT", "2.0"))
OUTPUT_DIR = Path("output/eldutch_pilot")
PMXT_LOCAL_MIRROR = "local:C:/hades-data/pmxt-mirror"

MODE = f"wd{WINDOW_DAY}_sl{int(USE_STOP_LOSS)}_fl{int(USE_FLASH)}_lat{int(LATENCY_MS)}"


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_selections() -> list[dict]:
    selections = []
    for path in sorted(SELECTION_DIR.glob(f"*__wd{WINDOW_DAY}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if EVENTS_FILTER and payload["event_slug"] not in EVENTS_FILTER:
            continue
        selections.append(payload)
    return selections


def _event_results_path(event_slug: str) -> Path:
    return OUTPUT_DIR / f"result_{event_slug}__{MODE}.json"


def _run_single_event(sel: dict) -> None:
    """Run one weekly event in THIS process (child mode)."""
    from decimal import Decimal

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
    from prediction_market_extensions.backtesting.data_sources import Book, Polymarket, PMXT

    event_slug = sel["event_slug"]
    event_key = f"{event_slug}__{MODE}"
    replays = tuple(
        BookReplay(
            market_slug=bucket["market_slug"],
            outcome="Yes",
            start_time=_iso(sel["entry_ts"]),
            end_time=_iso(sel["end_ts"]),
            metadata={
                "sim_label": bucket["market_slug"],
                "replay_window_start_ns": sel["activation_ns"],
                "replay_window_end_ns": sel["close_ns"],
            },
        )
        for bucket in sel["buckets"]
    )
    basket_cost = sum(bucket["entry_price_ref"] * TARGET_SHARES for bucket in sel["buckets"])
    initial_cash = max(100.0, CASH_MULT * basket_cost)
    report_path = OUTPUT_DIR / f"{event_slug}__{MODE}.html"

    experiment = build_replay_experiment(
        name=f"eldutch_pilot_{event_slug}_{MODE}",
        description="Vanilla EL-DUTCH weekly-basket pilot on PMXT books",
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
                "strategy_path": "strategies:ElDutchBucketStrategy",
                "config_path": "strategies:ElDutchBucketConfig",
                "config": {
                    "event_key": event_key,
                    "target_shares": Decimal(int(TARGET_SHARES)),
                    "use_stop_loss": USE_STOP_LOSS,
                    "use_flash": USE_FLASH,
                    "fee_rate": sel.get("fee_rate", 0.05),
                    "activation_start_time_ns": sel["activation_ns"],
                    "market_close_time_ns": sel["close_ns"],
                },
            }
        ],
        initial_cash=initial_cash,
        probability_window=30,
        min_book_events=1,
        min_price_range=0.0,
        execution=ExecutionModelConfig(
            queue_position=True,
            latency_model=StaticLatencyConfig(
                base_latency_ms=LATENCY_MS,
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
            summary_report_path=report_path.as_posix(),
        ),
        empty_message=f"No replay windows met requirements for {event_slug}.",
        partial_message=(f"{event_slug}: completed {{completed}} of {{total}} bucket replays."),
        return_summary_series=True,
    )
    results = run_experiment(experiment)
    payload = {
        "event_slug": event_slug,
        "mode": MODE,
        "window_day": sel["window_day"],
        "n_buckets": len(sel["buckets"]),
        "sum_entry_prices": sel["sum_entry_prices"],
        "basket_cost_ref": round(basket_cost, 4),
        "initial_cash": round(initial_cash, 2),
        "results": results,
    }
    path = _event_results_path(event_slug)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"Wrote {path}")


def run() -> None:
    selections = _load_selections()
    if not selections:
        print(f"No wd{WINDOW_DAY} selections found in {SELECTION_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"EL-DUTCH pilot: {len(selections)} events, window-day {WINDOW_DAY}, "
        f"stop_loss={USE_STOP_LOSS}, flash={USE_FLASH}, latency={LATENCY_MS}ms"
    )

    if len(selections) == 1:
        _run_single_event(selections[0])
        return

    # Parent mode: one subprocess per event (fresh Nautilus logging singleton
    # per engine), then merge the per-event payloads.
    completed: list[str] = []
    failed: list[str] = []
    for sel in selections:
        event_slug = sel["event_slug"]
        env = dict(os.environ)
        env["ELDUTCH_EVENTS"] = event_slug
        env.setdefault("PYTHONUTF8", "1")
        print(f"\n=== {event_slug} ({MODE}) ===", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        if proc.returncode == 0 and _event_results_path(event_slug).exists():
            completed.append(event_slug)
        else:
            failed.append(event_slug)
            print(f"FAILED: {event_slug} (exit {proc.returncode})", flush=True)

    aggregate = [
        json.loads(_event_results_path(slug).read_text(encoding="utf-8")) for slug in completed
    ]
    results_path = OUTPUT_DIR / f"results_{MODE}.json"
    results_path.write_text(json.dumps(aggregate, indent=1, default=str), encoding="utf-8")
    print(f"\nWrote {results_path} ({len(completed)} events, {len(failed)} failed)")
    if failed:
        print("Failed events: " + ", ".join(failed))


if __name__ == "__main__":
    run()
