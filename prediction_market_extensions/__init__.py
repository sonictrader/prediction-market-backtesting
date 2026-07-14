"""Prediction market extensions for NautilusTrader."""

from __future__ import annotations

_COMMISSION_PATCH_INSTALLED = False


def install_commission_patch() -> None:
    """
    Install the repo's Polymarket commission rounding policy.

    Nautilus 1.226 uses the current curved fee formula and pUSD currency
    model. This startup hook keeps this repository's fee rounding centralized
    while targeting the 1.226 function signature directly.
    """
    global _COMMISSION_PATCH_INSTALLED
    if _COMMISSION_PATCH_INSTALLED:
        return

    import nautilus_trader.adapters.polymarket.common.parsing as upstream_parsing

    from prediction_market_extensions.adapters.polymarket import parsing as pm_parsing

    upstream_parsing.calculate_commission = pm_parsing.calculate_commission
    _COMMISSION_PATCH_INSTALLED = True


_NETWORK_SHIMS_INSTALLED = False


def install_network_shims() -> None:
    """
    Install OS trust-store TLS and an honest default urllib User-Agent.

    Some environments (observed on Windows with OS-level trust stores) fail
    Python TLS verification against r2*.pmxt.dev, gamma-api, and data-api,
    and those Cloudflare-fronted hosts also reject the default
    ``Python-urllib/x.y`` User-Agent with 403. Use the OS certificate store
    when ``truststore`` is installed, and send a tool-identifying User-Agent
    on urllib openers by default.
    """
    global _NETWORK_SHIMS_INSTALLED
    if _NETWORK_SHIMS_INSTALLED:
        return

    try:
        import truststore
    except ImportError:
        truststore = None
    if truststore is not None:
        truststore.inject_into_ssl()

    from urllib.request import build_opener, install_opener

    opener = build_opener()
    opener.addheaders = [
        (
            "User-Agent",
            "prediction-market-backtesting/4.1 "
            "(research; +https://github.com/sonictrader/prediction-market-backtesting)",
        )
    ]
    install_opener(opener)
    _NETWORK_SHIMS_INSTALLED = True


install_commission_patch()
install_network_shims()
