# Added on the hades branch: vanilla EL-DUTCH basket strategy port.
# Mapping doc: 1337-HADES-mono/memory-bank/analysis/eldutch/
#   mapping-eldutch-nautilus-2026-07-14.md (Ivo-approved 2026-07-14).
# Ports deployments/el-dutch/strategy-manager/logic.py behavior dial-for-dial:
# patient maker entries at considered_bid + tick (cumulative-liquidity walk),
# reprice hysteresis (tolerance ticks + dwell), fee-net portfolio gate,
# per-bucket stop-loss (taker at bid), portfolio flash liquidation at fee-net
# ROI, hold winners to resolution. NO take-profit layer (vanilla scope).
# Distributed under the GNU Lesser General Public License Version 3.0 or later.
# See the repository NOTICE file for provenance and licensing scope.

from __future__ import annotations

import math
from decimal import Decimal

from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.enums import BookType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

FLASH_CHECK_THROTTLE_SECS = 60.0  # SM check_flash_profit 60s throttle


def _fee_for_leg(price: float, fee_rate: float) -> float:
    """Polymarket taker fee per share: rate * p * (1 - p)."""
    clamped = min(max(price, 0.0), 1.0)
    return fee_rate * clamped * (1.0 - clamped)


def _level_price(level: object) -> float | None:
    price = getattr(level, "price", None)
    if price is None:
        return None
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _level_size(level: object) -> float:
    size = getattr(level, "size", None)
    if callable(size):
        size = size()
    try:
        value = float(size)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def considered_price(
    levels: list[object], *, dollar_bar: float, share_bar: float
) -> tuple[float, float, str]:
    """SM `_cumulative_side` + the 270.3 price-scaled bar.

    Walk one side best-first accumulating price*size dollars; the considered
    price is the first level where the cumulative amount STRICTLY exceeds the
    effective bar = min(dollar_bar, share_bar * side touch price).
    Returns (cumulative_dollars, considered_price, status).
    """
    touch = _level_price(levels[0]) if levels else None
    if touch is None:
        return 0.0, 0.0, "none"
    bar = min(dollar_bar, share_bar * touch)
    cumulative = 0.0
    last_price = 0.0
    for level in levels:
        price = _level_price(level)
        if price is None:
            continue
        last_price = price
        cumulative += price * _level_size(level)
        if cumulative > bar:
            return cumulative, price, "ok"
    if cumulative > 0.0:
        return cumulative, last_price, "insufficient"
    return 0.0, 0.0, "none"


class ElDutchEventPortfolio:
    """Shared per-event state: fee-net entry gate + flash-profit trigger.

    The SM computes both from one `_portfolio_math()`; bucket strategies here
    push their latest view (commit price, position, avg entry, best bid) on
    every book event and the portfolio answers the same two questions with the
    same formulas.
    """

    _REGISTRY: dict[str, ElDutchEventPortfolio] = {}

    def __init__(self, event_key: str) -> None:
        self.event_key = event_key
        self._buckets: dict[InstrumentId, ElDutchBucketStrategy] = {}
        self.liquidating = False
        self._last_flash_check_ns = 0

    @classmethod
    def for_event(cls, event_key: str) -> ElDutchEventPortfolio:
        portfolio = cls._REGISTRY.get(event_key)
        if portfolio is None:
            portfolio = cls(event_key)
            cls._REGISTRY[event_key] = portfolio
        return portfolio

    def register(self, strategy: ElDutchBucketStrategy) -> None:
        # A fresh engine re-registers every bucket; replacing stale instances
        # (from a previous run of the same event in this process) also resets
        # the portfolio flags when the roster turns over completely.
        existing = self._buckets.get(strategy.config.instrument_id)
        if existing is not None and existing is not strategy:
            self._buckets[strategy.config.instrument_id] = strategy
            if all(s.is_fresh for s in self._buckets.values()):
                self.liquidating = False
                self._last_flash_check_ns = 0
        else:
            self._buckets[strategy.config.instrument_id] = strategy

    # ---- fee-net entry gate (SM _entry_net_gate / _portfolio_math) ----

    def entry_gate_ok(
        self,
        candidate: ElDutchBucketStrategy,
        candidate_price: float,
        *,
        max_net: float,
    ) -> bool:
        projected_cost = 0.0
        projected_fees = 0.0
        for strategy in self._buckets.values():
            if strategy is candidate:
                px = candidate_price
            else:
                px = strategy.commit_price()
            if px > 0.0:
                projected_cost += px
                projected_fees += _fee_for_leg(px, strategy.config.fee_rate)
        return (projected_cost + projected_fees) < max_net

    # ---- flash profit (SM check_flash_profit fee-net math) ----

    def maybe_flash(self, ts_event_ns: int) -> None:
        if self.liquidating:
            return
        throttle_ns = int(FLASH_CHECK_THROTTLE_SECS * 1e9)
        if ts_event_ns - self._last_flash_check_ns < throttle_ns:
            return
        self._last_flash_check_ns = ts_event_ns

        holdings_cost = 0.0
        net_proceeds = 0.0
        n_holding = 0
        min_filled_frac = 0.0
        for strategy in self._buckets.values():
            min_filled_frac = float(strategy.config.flash_min_filled_frac)
            shares = strategy.position_size()
            if shares <= 0.0:
                continue
            avg_entry = strategy.avg_entry_price()
            best_bid = strategy.last_best_bid()
            if avg_entry is None:
                continue
            n_holding += 1
            holdings_cost += shares * avg_entry
            if best_bid is not None and best_bid > 0.0:
                net_proceeds += (
                    shares * best_bid - _fee_for_leg(best_bid, strategy.config.fee_rate) * shares
                )
        if n_holding == 0 or holdings_cost <= 0.0:
            return
        # Flash coverage gate (Ivo, 270.10): a +10% ROI on one stray bucket is
        # noise, not a locked dutch spread - only arm the flash once at least
        # this fraction of the selected basket is actually held. 0 = the
        # as-implemented SM behavior (no gate).
        if min_filled_frac > 0.0 and n_holding < min_filled_frac * len(self._buckets):
            return
        net_roi = (net_proceeds - holdings_cost) / holdings_cost
        flash_pct = None
        for strategy in self._buckets.values():
            flash_pct = float(strategy.config.flash_profit_pct)
            break
        if flash_pct is None or net_roi < flash_pct:
            return

        self.liquidating = True
        for strategy in self._buckets.values():
            strategy.liquidate_for_flash()


class ElDutchBucketConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    event_key: str
    # SM dials (deployments/el-dutch/strategy-manager/config.py names)
    target_shares: Decimal = Decimal(50)
    min_order_size: Decimal = Decimal(5)
    min_liquidity_per_side: float = 250.0
    min_liquidity_shares: float = 1000.0
    maker_reprice_tolerance_ticks: int = 1
    maker_reprice_dwell_secs: float = 15.0
    stop_loss_pct: float = 0.60
    flash_profit_pct: float = 0.10
    flash_min_filled_frac: float = 0.0
    use_stop_loss: bool = True
    use_flash: bool = True
    fee_rate: float = 0.05
    max_portfolio_net: float = 1.0
    activation_start_time_ns: int = 0
    market_close_time_ns: int = 0

    def __post_init__(self) -> None:
        if self.target_shares <= 0:
            raise ValueError(f"target_shares must be > 0, got {self.target_shares}")
        if self.min_order_size <= 0:
            raise ValueError(f"min_order_size must be > 0, got {self.min_order_size}")
        if not 0.0 < float(self.stop_loss_pct) < 1.0:
            raise ValueError(f"stop_loss_pct must be in (0, 1), got {self.stop_loss_pct}")
        if float(self.flash_profit_pct) <= 0.0:
            raise ValueError(f"flash_profit_pct must be > 0, got {self.flash_profit_pct}")
        if not 0.0 <= float(self.flash_min_filled_frac) <= 1.0:
            raise ValueError(
                f"flash_min_filled_frac must be in [0, 1], got {self.flash_min_filled_frac}"
            )
        if not 0.0 <= float(self.fee_rate) < 1.0:
            raise ValueError(f"fee_rate must be in [0, 1), got {self.fee_rate}")


class ElDutchBucketStrategy(Strategy):
    """One selected bucket of a vanilla EL-DUTCH basket (see module docstring)."""

    def __init__(self, config: ElDutchBucketConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._order_book: OrderBook | None = None
        # working entry order state
        self._working_order_id = None
        self._working_price: float | None = None
        self._pending: bool = False  # submit/cancel in flight
        self._mispriced_since_ns: int | None = None
        # position state (SM store semantics: fills are authoritative)
        self._entry_qty_sum = Decimal("0")
        self._entry_cost_sum = Decimal("0")
        # exit guards (SM sell_attempted / portfolio_liquidating)
        self._sell_attempted = False
        self._last_best_bid: float | None = None
        self._last_ts_event_ns: int = 0
        self.is_fresh = True
        self._portfolio: ElDutchEventPortfolio | None = None

    # ---- portfolio-facing views ----

    def position_size(self) -> float:
        return float(self._entry_qty_sum)

    def avg_entry_price(self) -> float | None:
        if self._entry_qty_sum > 0:
            return float(self._entry_cost_sum / self._entry_qty_sum)
        return None

    def last_best_bid(self) -> float | None:
        return self._last_best_bid

    def commit_price(self) -> float:
        """SM `_bucket_commit_price`: avg entry > resting BUY price > best ask."""
        avg = self.avg_entry_price()
        if avg is not None and avg > 0.0:
            return avg
        if self._working_price is not None:
            return self._working_price
        if self._order_book is not None:
            ask = self._order_book.best_ask_price()
            if ask is not None:
                return float(ask)
        return 0.0

    # ---- lifecycle ----

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found - stopping.")
            self.stop()
            return
        self._portfolio = ElDutchEventPortfolio.for_event(self.config.event_key)
        self._portfolio.register(self)
        self.subscribe_order_book_deltas(
            instrument_id=self.config.instrument_id,
            book_type=BookType.L2_MBP,
        )

    def on_order_book_deltas(self, deltas) -> None:  # type: ignore[no-untyped-def]
        if self._order_book is None:
            self._order_book = OrderBook(self.config.instrument_id, book_type=BookType.L2_MBP)
        self._order_book.apply_deltas(deltas)
        ts_event_ns = int(getattr(deltas, "ts_event", 0) or 0)
        self._last_ts_event_ns = ts_event_ns
        self.is_fresh = False
        bid = self._order_book.best_bid_price()
        self._last_best_bid = float(bid) if bid is not None else None

        # SM _handle_orderbook order: manage -> entry -> stop-loss -> flash
        self._manage_buy_order(ts_event_ns)
        self._patient_entry(ts_event_ns)
        if self.config.use_stop_loss:
            self._monitor_stop_loss()
        if self.config.use_flash and self._portfolio is not None:
            self._portfolio.maybe_flash(ts_event_ns)

    # ---- entry management (SM manage_buy_orders + execute_patient_entry) ----

    def _entry_window_open(self, ts_event_ns: int) -> bool:
        start_ns = int(self.config.activation_start_time_ns)
        if start_ns > 0 and ts_event_ns < start_ns:
            return False
        close_ns = int(self.config.market_close_time_ns)
        if close_ns > 0 and ts_event_ns > close_ns:
            return False
        return True

    def _ideal_entry_price(self) -> tuple[float | None, str]:
        assert self._order_book is not None
        _liq, considered_bid, status = considered_price(
            self._order_book.bids(),
            dollar_bar=float(self.config.min_liquidity_per_side),
            share_bar=float(self.config.min_liquidity_shares),
        )
        if status != "ok":
            return None, status
        tick = float(self._instrument.price_increment)
        ideal = considered_bid + tick
        best_ask = self._order_book.best_ask_price()
        # Maker safety: never cross - join the considered bid on tight spreads
        if best_ask is not None and ideal >= float(best_ask):
            ideal = considered_bid
        return ideal, "ok"

    def _manage_buy_order(self, ts_event_ns: int) -> None:
        if self._working_order_id is None or self._pending:
            return
        if self._order_book is None:
            return
        ideal, status = self._ideal_entry_price()
        if status != "ok":
            self._cancel_working("insufficient_liquidity")
            return
        assert ideal is not None
        our_price = self._working_price or 0.0
        deviation = abs(our_price - ideal)
        if deviation <= 1e-9:
            self._mispriced_since_ns = None
            return
        tick = float(self._instrument.price_increment)
        tolerance = float(self.config.maker_reprice_tolerance_ticks) * tick
        if deviation <= tolerance + 1e-12:
            if self._mispriced_since_ns is None:
                self._mispriced_since_ns = ts_event_ns
                return
            dwell_ns = int(float(self.config.maker_reprice_dwell_secs) * 1e9)
            if ts_event_ns - self._mispriced_since_ns < dwell_ns:
                return
        self._cancel_working("price_improvement")

    def _patient_entry(self, ts_event_ns: int) -> None:
        if self._pending or self._working_order_id is not None:
            return
        if self._portfolio is not None and self._portfolio.liquidating:
            return
        if not self._entry_window_open(ts_event_ns):
            return
        if self._order_book is None:
            return

        shortfall = self.config.target_shares - self._entry_qty_sum
        if shortfall < self.config.min_order_size:
            return

        ideal, status = self._ideal_entry_price()
        if status != "ok" or ideal is None or ideal <= 0.0:
            return

        # Fee-net portfolio gate (SM _entry_net_gate)
        if self._portfolio is not None and not self._portfolio.entry_gate_ok(
            self, ideal, max_net=float(self.config.max_portfolio_net)
        ):
            return

        # $1.00 CLOB minimum notional (SM _below_min_notional)
        if ideal * float(shortfall) < 1.0:
            return

        try:
            quantity = self._instrument.make_qty(float(shortfall), round_down=True)
        except ValueError:
            return
        if quantity.as_double() <= 0.0:
            return
        order = self.order_factory.limit(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            price=self._instrument.make_price(ideal),
            time_in_force=TimeInForce.GTC,
        )
        self._pending = True
        self._working_price = ideal
        self._mispriced_since_ns = None
        try:
            self.submit_order(order)
        except Exception:
            self._pending = False
            self._working_price = None
            raise

    def _cancel_working(self, reason: str) -> None:
        if self._working_order_id is None or self._pending:
            return
        order = self.cache.order(self._working_order_id)
        if order is None or order.is_closed:
            self._clear_working()
            return
        self._pending = True
        self.cancel_order(order)

    def _clear_working(self) -> None:
        self._working_order_id = None
        self._working_price = None
        self._mispriced_since_ns = None

    # ---- exits ----

    def _monitor_stop_loss(self) -> None:
        if self._sell_attempted or self._pending:
            return
        if self._portfolio is not None and self._portfolio.liquidating:
            return  # flash liquidation owns the exit
        shares = self.position_size()
        if shares <= 0.0:
            return
        avg_entry = self.avg_entry_price()
        if avg_entry is None or avg_entry <= 0.0:
            return
        best_bid = self._last_best_bid
        if best_bid is None or best_bid <= 0.0:
            return
        threshold = avg_entry * (1.0 - float(self.config.stop_loss_pct))
        if best_bid > threshold:
            return
        sell_size = math.floor(shares)
        if sell_size <= 0:
            return
        self.log.warning(
            f"STOP LOSS: bid {best_bid:.4f} <= threshold {threshold:.4f} "
            f"(entry {avg_entry:.4f}) - taker exit {sell_size} shares"
        )
        self._sell_attempted = True
        self._submit_taker_sell(sell_size, best_bid)

    def liquidate_for_flash(self) -> None:
        """Portfolio flash: cancel entry, taker-sell the whole position at bid."""
        if self._working_order_id is not None and not self._pending:
            self._cancel_working("flash_liquidation")
        shares = math.floor(self.position_size())
        best_bid = self._last_best_bid
        if shares <= 0 or best_bid is None or best_bid <= 0.0:
            return
        self.log.info(f"FLASH LIQUIDATION: selling {shares} shares at bid {best_bid:.4f}")
        self._submit_taker_sell(shares, best_bid)

    def _submit_taker_sell(self, size: int, bid_price: float) -> None:
        try:
            quantity = self._instrument.make_qty(float(size), round_down=True)
        except ValueError:
            return
        if quantity.as_double() <= 0.0:
            return
        order = self.order_factory.limit(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=quantity,
            price=self._instrument.make_price(bid_price),
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )
        self.submit_order(order)

    # ---- order events ----

    def on_order_accepted(self, event) -> None:  # type: ignore[no-untyped-def]
        order = self.cache.order(getattr(event, "client_order_id", None))
        if order is not None and order.side == OrderSide.BUY:
            self._working_order_id = order.client_order_id
        self._pending = False

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        fill_px = Decimal(str(event.last_px))
        fill_qty = Decimal(str(event.last_qty))
        if event.order_side == OrderSide.BUY:
            self._entry_cost_sum += fill_px * fill_qty
            self._entry_qty_sum += fill_qty
            # SM partial-fill semantics: cancel the remainder on any fill;
            # the entry path re-places for the new shortfall next book event.
            if event.client_order_id == self._working_order_id:
                order = self.cache.order(event.client_order_id)
                if order is not None and not order.is_closed:
                    if not self._pending:
                        self._pending = True
                        self.cancel_order(order)
                else:
                    self._clear_working()
                    self._pending = False
        else:
            sold = min(fill_qty, self._entry_qty_sum)
            if self._entry_qty_sum > 0:
                average_cost = self._entry_cost_sum / self._entry_qty_sum
                self._entry_cost_sum -= average_cost * sold
            self._entry_qty_sum -= sold
            if self._entry_qty_sum <= 0:
                self._entry_qty_sum = Decimal("0")
                self._entry_cost_sum = Decimal("0")
            # SM resets both guards on a SELL fill (re-entry is possible)
            self._sell_attempted = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.client_order_id == self._working_order_id:
            self._clear_working()
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.client_order_id == self._working_order_id:
            self._clear_working()
        self._pending = False

    def on_order_denied(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.client_order_id == self._working_order_id:
            self._clear_working()
        self._pending = False

    def on_order_expired(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.client_order_id == self._working_order_id:
            self._clear_working()
        self._pending = False

    def on_stop(self) -> None:
        # Hold to resolution: cancel resting orders only, keep positions for
        # the runner's settlement marking.
        self.cancel_all_orders(self.config.instrument_id)

    def on_reset(self) -> None:
        self._order_book = None
        self._clear_working()
        self._pending = False
        self._entry_qty_sum = Decimal("0")
        self._entry_cost_sum = Decimal("0")
        self._sell_attempted = False
        self._last_best_bid = None
        self._last_ts_event_ns = 0
        self.is_fresh = True
