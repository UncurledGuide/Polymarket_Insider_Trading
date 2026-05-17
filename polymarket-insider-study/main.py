"""
main.py — Polymarket Insider Trading Detection: Van Dyke + Two Unnamed Wallets

Ground truth: US v. Gannon Ken Van Dyke (SDNY, 2026). Van Dyke was a DoD contractor
who traded Polymarket contracts tied to a Venezuela military operation he had advance
knowledge of. He is the only confirmed insider in this dataset.

We run the same four statistical signals on:
  1. Van Dyke (hardcoded from DOJ indictment)
  2. wallet_2_a72D  — flagged by on-chain analysts, not prosecuted
  3. wallet_3_SBet365 — flagged by on-chain analysts, not prosecuted

Usage:
    python main.py          # reads POLYGONSCAN_API_KEY from .env automatically
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory
import sys
import textwrap
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# Ensure the src/ package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from signals import (
    TradeRecord,
    AccountFlags,
    run_all_signals,
)


# ===========================================================================
# 1.  VAN DYKE — hardcoded from DOJ indictment
# ===========================================================================

VANDYKE_TRADES_RAW = [
    {"date": "2025-12-27", "market": "US Forces Venezuela by Jan 31",  "shares": 634,    "price": 0.15, "cost_usdc": 96},
    {"date": "2025-12-30", "market": "Maduro Out by Jan 31",            "shares": 13769,  "price": 0.09, "cost_usdc": 1238},
    {"date": "2025-12-30", "market": "Maduro Out by Jan 31",            "shares": 850,    "price": 0.10, "cost_usdc": 85},
    {"date": "2026-01-01", "market": "US Invade Venezuela by Jan 31",   "shares": 17858,  "price": 0.06, "cost_usdc": 1071},
    {"date": "2026-01-01", "market": "Trump War Powers by Jan 31",      "shares": 1752,   "price": 0.06, "cost_usdc": 105},
    {"date": "2026-01-01", "market": "Maduro Out by Jan 31",            "shares": 73685,  "price": 0.07, "cost_usdc": 5158},
    {"date": "2026-01-02", "market": "Trump War Powers by Jan 31",      "shares": 3618,   "price": 0.04, "cost_usdc": 145},
    {"date": "2026-01-02", "market": "Maduro Out by Jan 31",            "shares": 90347,  "price": 0.07, "cost_usdc": 6150},
    {"date": "2026-01-02", "market": "Maduro Out by Jan 31",            "shares": 82421,  "price": 0.07, "cost_usdc": 6000},
    {"date": "2026-01-02", "market": "US Forces Venezuela by Jan 31",   "shares": 564,    "price": 0.07, "cost_usdc": 39},
    {"date": "2026-01-02", "market": "Maduro Out by Jan 31",            "shares": 87500,  "price": 0.08, "cost_usdc": 7050},
    {"date": "2026-01-02", "market": "Maduro Out by Jan 31",            "shares": 88187,  "price": 0.08, "cost_usdc": 7215},
]

VANDYKE_ACCOUNT_CREATED = "2025-12-26"
EVENT_TIMESTAMP         = "2026-01-03 04:21:00"   # Trump TruthSocial post (EST)
VANDYKE_TOTAL_PROFIT    = 409_881                  # USD, from indictment

VANDYKE_TRADES = [TradeRecord(**t) for t in VANDYKE_TRADES_RAW]


# ===========================================================================
# 2.  ADDITIONAL WALLETS — Polygonscan live fetch with hardcoded fallback
# ===========================================================================

WALLETS = {
    "wallet_2_a72D":    "0xa72DB1749e9AC2379D49A3c12708325ED17FeBd4",
    "wallet_3_SBet365": "0x6baf05d193692bb208d616709e27442c910a94c5",
}
KNOWN_PROFITS = {
    "wallet_2_a72D":    75_000,
    "wallet_3_SBet365": 145_600,
}
KNOWN_INVESTED = {
    "wallet_2_a72D":    5_800,
    "wallet_3_SBet365": 25_000,
}
POLYMARKET_CTF = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# Polygonscan deprecated V1; the V2 unified endpoint requires chainid=137 for Polygon
POLYGONSCAN_API = "https://api.etherscan.io/v2/api"
POLYGON_CHAIN_ID = 137

# Hardcoded fallback trades for each wallet (based on public reporting).
# wallet_2: deposited ~$5,800 Dec 28 2025, withdrew ~$75k Jan 3 2026
# wallet_3: deposited ~$25,000 Dec 28 2025, withdrew ~$145,600 Jan 3 2026
#
# Share counts are back-calculated from known profits: since each resolved YES
# share pays $1 USDC, total_shares = invested + profit.  The implied purchase
# price = invested / total_shares, which gives us the market-odds baseline
# used in Signal 1 and Signal 2.
_w2_shares = 5_800 + 75_000   # = 80,800  → implied price ≈ 0.0718
_w3_shares = 25_000 + 145_600  # = 170,600 → implied price ≈ 0.1466

FALLBACK_TRADES: dict[str, list[TradeRecord]] = {
    "wallet_2_a72D": [
        TradeRecord(date="2025-12-28", market="Maduro Out by Jan 31",
                    shares=_w2_shares, price=round(5_800 / _w2_shares, 4),
                    cost_usdc=5_800, resolved_yes=True),
    ],
    "wallet_3_SBet365": [
        TradeRecord(date="2025-12-28", market="Maduro Out by Jan 31",
                    shares=_w3_shares, price=round(25_000 / _w3_shares, 4),
                    cost_usdc=25_000, resolved_yes=True),
    ],
}
FALLBACK_ACCOUNT_CREATED = {
    "wallet_2_a72D":    "2025-12-27",
    "wallet_3_SBet365": "2025-12-20",
}


def _polygonscan_tokentx(wallet_address: str, api_key: str) -> list[dict] | None:
    """Fetch all ERC-20 token transfers for a wallet via Etherscan V2 (Polygon)."""
    params = {
        "chainid": POLYGON_CHAIN_ID,
        "module":  "account",
        "action":  "tokentx",
        "address": wallet_address,
        "apikey":  api_key,
        "sort":    "asc",
    }
    try:
        resp = requests.get(POLYGONSCAN_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [WARN] Polygonscan request failed: {exc}")
        return None

    if data.get("status") != "1":
        print(f"    [WARN] Polygonscan returned status={data.get('status')}: {data.get('message')}")
        return None

    return data["result"]


def fetch_wallet_trades(
    wallet_address: str,
    api_key: str,
    event_ts: int,
) -> tuple[list[TradeRecord], str] | None:
    """
    Pull ERC-20 transfers for *wallet_address* and reconstruct Polymarket trades.

    Strategy:
      1. Identify CTF deposits  (wallet → CTF, before event) — these are the buys.
      2. Identify payout inflows (any large USDC inflow arriving after the event
         that is NOT from a known funding relay) — these are the resolved winnings.
      3. Back-calculate shares per trade proportionally from total payout.
         Since each resolved YES share pays $1 USDC, total_payout ≈ total_shares.
      4. Derive implied market price per trade = cost / allocated_shares.
      5. Use exact UTC datetimes (to HH:MM precision) for the timing signal.

    Returns (trades, account_created_date) or None on failure.
    """
    wallet_lower = wallet_address.lower()
    ctf_lower = POLYMARKET_CTF.lower()

    txns = _polygonscan_tokentx(wallet_address, api_key)
    if txns is None:
        return None

    # Only process USDC tokens (native or bridged)
    usdc_txns = [
        t for t in txns
        if t.get("tokenSymbol", "").upper() in ("USDC", "USDC.E")
    ]

    if not usdc_txns:
        return None

    # Account creation proxy: date of the very first ERC-20 transaction
    import calendar as _cal
    first_tx_ts = int(usdc_txns[0]["timeStamp"])
    # Use UTC explicitly to avoid local-timezone issues
    first_tx_dt = datetime(1970, 1, 1) + __import__("datetime").timedelta(seconds=first_tx_ts)
    from datetime import timedelta
    account_created = (first_tx_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Step 1: CTF deposits before the event ──────────────────────────────
    ctf_deposits: list[dict] = []
    for tx in usdc_txns:
        ts = int(tx["timeStamp"])
        if tx["from"].lower() == wallet_lower and tx["to"].lower() == ctf_lower:
            if ts < event_ts:  # only pre-event buys are relevant
                ctf_deposits.append(tx)

    if not ctf_deposits:
        print(f"    [INFO] No pre-event CTF deposits found for {wallet_address[:10]}…")
        return None

    total_invested_usdc = sum(int(d["value"]) / 1e6 for d in ctf_deposits)

    # ── Step 2: Identify payout inflows arriving after the event ──────────
    # The Polymarket settlement contract (or a market-maker) sends USDC back
    # after resolution. These are large inflows arriving AFTER the event.
    # We treat any inflow > 10% of invested capital arriving after event as payout.
    payout_threshold = total_invested_usdc * 0.10
    total_payout_usdc = 0.0
    for tx in usdc_txns:
        ts = int(tx["timeStamp"])
        value = int(tx["value"]) / 1e6
        if (tx["to"].lower() == wallet_lower
                and ts >= event_ts
                and value >= payout_threshold):
            total_payout_usdc += value

    # Fallback: if payout detection fails, use known profit from public reporting
    if total_payout_usdc < total_invested_usdc:
        print(f"    [INFO] Payout not detected on-chain; using known profit from reports.")
        return None  # triggers fallback in caller

    # ── Step 3: Build TradeRecord per CTF deposit ──────────────────────────
    # Shares allocated proportionally: each deposit gets its fraction of total payout.
    # Implied price = cost / shares (lower price → larger potential return → more suspicious).
    trades: list[TradeRecord] = []
    for dep in ctf_deposits:
        ts = int(dep["timeStamp"])
        cost = int(dep["value"]) / 1e6
        # Convert UTC blockchain timestamp to EST (UTC-5) to match EVENT_TIMESTAMP
        # which is expressed in EST. This keeps timing comparisons in the same timezone.
        from datetime import timedelta as _td
        # Convert UTC blockchain Unix timestamp → EST without local-tz ambiguity
        dt_utc = datetime(1970, 1, 1) + _td(seconds=ts)
        dt_est = dt_utc - _td(hours=5)
        dt_est_str = dt_est.strftime("%Y-%m-%d")
        dt_est_full = dt_est.strftime("%Y-%m-%d %H:%M:%S")

        allocated_shares = (cost / total_invested_usdc) * total_payout_usdc
        implied_price = cost / allocated_shares if allocated_shares > 0 else 0.07

        trades.append(TradeRecord(
            date=dt_est_str,
            market=f"Venezuela/Maduro (on-chain, {dt_est_full} EST)",
            shares=round(allocated_shares, 2),
            price=round(implied_price, 4),
            cost_usdc=round(cost, 2),
            resolved_yes=True,
            datetime_utc=dt_est_full,  # stored as EST for consistent comparison
        ))

    print(f"    [INFO] Found {len(trades)} pre-event CTF trade(s). "
          f"Invested: ${total_invested_usdc:,.2f}  Payout: ${total_payout_usdc:,.2f}")

    return trades, account_created


def get_wallet_data(
    wallet_key: str, api_key: str | None
) -> tuple[list[TradeRecord], str, bool]:
    """
    Returns (trades, account_created_date, used_fallback).
    Tries live Polygonscan fetch; falls back to hardcoded data if unavailable.
    """
    from datetime import timedelta
    wallet_address = WALLETS[wallet_key]

    # Convert EST event timestamp to a UTC Unix timestamp for blockchain comparison.
    # Use calendar.timegm which always interprets the struct as UTC, so we avoid
    # the local-timezone pitfall of naive_datetime.timestamp().
    import calendar
    event_dt_est = datetime.strptime(EVENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S")
    event_dt_utc = event_dt_est + timedelta(hours=5)  # EST → UTC
    event_ts_utc = calendar.timegm(event_dt_utc.timetuple())

    if api_key:
        print(f"  Fetching {wallet_key} ({wallet_address[:10]}…) from Polygonscan…")
        result = fetch_wallet_trades(wallet_address, api_key, event_ts_utc)
        time.sleep(0.25)  # respect rate limit

        if result is not None:
            live_trades, account_created = result
            return live_trades, account_created, False

    print(f"  [FALLBACK] Using hardcoded data for {wallet_key}")
    return (
        FALLBACK_TRADES[wallet_key],
        FALLBACK_ACCOUNT_CREATED[wallet_key],
        True,
    )


# ===========================================================================
# 3.  REPORTING UTILITIES
# ===========================================================================

def print_section(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def fmt_pval(p: float) -> str:
    """Format a p-value with significance stars."""
    stars = ""
    if p < 0.001:
        stars = " ***"
    elif p < 0.01:
        stars = " **"
    elif p < 0.05:
        stars = " *"
    return f"{p:.4f}{stars}"


def print_subject_report(
    label: str,
    trades: list[TradeRecord],
    account_created: str,
    total_invested: float,
    total_profit: float,
    results: dict,
    used_fallback: bool = False,
) -> None:
    """Print a detailed per-subject signal report."""
    sig1 = results["signal_1_binomial"]
    sig2 = results["signal_2_bootstrap"]
    sig3 = results["signal_3_timing"]
    sig4: AccountFlags = results["signal_4_flags"]

    fallback_note = " [HARDCODED FALLBACK — no live API data]" if used_fallback else ""
    print_section(f"{label}{fallback_note}")

    print(f"\n  Portfolio summary:")
    print(f"    Trades:          {len(trades)}")
    print(f"    Total invested:  ${total_invested:>10,.0f}")
    print(f"    Total profit:    ${total_profit:>10,.0f}")
    print(f"    Return multiple: {(total_invested + total_profit) / total_invested:.1f}x")

    print(f"\n  Signal 1 — Binomial test (win rate vs market odds):")
    print(f"    N trades:               {sig1['n_trades']}")
    print(f"    N wins:                 {sig1['n_wins']}")
    print(f"    Market-implied avg p:   {sig1['weighted_avg_market_price']:.4f}")
    print(f"    Observed win rate:      {sig1['observed_win_rate']:.4f}")
    print(f"    p-value (one-tailed):   {fmt_pval(sig1['pvalue'])}")

    print(f"\n  Signal 2 — Bootstrap return test ({sig2['n_simulations']:,} simulations):")
    print(f"    Observed return multiple: {sig2['observed_multiple']:.2f}x")
    print(f"    Simulated median:         {sig2['sim_median_multiple']:.3f}x")
    print(f"    Simulated 95th pct:       {sig2['sim_p95_multiple']:.3f}x")
    print(f"    Simulated 99th pct:       {sig2['sim_p99_multiple']:.3f}x")
    print(f"    Empirical p-value:        {fmt_pval(sig2['pvalue'])}")

    print(f"\n  Signal 3 — Timing (hours before triggering event):")
    print(f"    Hours before event (per trade): {sig3.get('hours_before_event_per_trade', 'N/A')}")
    print(f"    Mean:                   {sig3.get('mean_hours_before', 'N/A')} h")
    print(f"    Median:                 {sig3.get('median_hours_before', 'N/A')} h")
    print(f"    Trades within 48h:      {sig3.get('frac_within_48h', 0)*100:.1f}%")
    print(f"    Trades within 7 days:   {sig3.get('frac_within_7d', 0)*100:.1f}%")
    print(f"    Concentration z-score:  {sig3.get('z_score_48h_concentration', 'N/A'):.2f}")
    print(f"    p-value (timing):       {fmt_pval(sig3.get('pvalue_timing', 1.0))}")

    print(f"\n  Signal 4 — Account & concentration flags:")
    print(f"    Account age at 1st trade: {sig4.account_age_days} day(s)")
    print(f"    Markets traded:           {sig4.n_markets_traded}")
    print(f"    Top-market capital pct:   {sig4.top_market_pct*100:.1f}%")
    print(f"    Days active:              {sig4.days_active_trading}")
    flags_str = ", ".join(sig4.flags) if sig4.flags else "NONE"
    print(f"    Flags raised ({sig4.flag_count}):          {flags_str}")


# ===========================================================================
# 4.  COMPARISON TABLE
# ===========================================================================

def build_comparison_table(
    subjects: list[dict],
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame.
    Each subject dict must have keys: label, invested, profit, results, prosecuted.
    """
    rows = []
    for s in subjects:
        res = s["results"]
        sig1 = res["signal_1_binomial"]
        sig2 = res["signal_2_bootstrap"]
        sig3 = res["signal_3_timing"]
        sig4: AccountFlags = res["signal_4_flags"]

        rows.append({
            "Subject":             s["label"],
            "Invested ($)":        f"{s['invested']:,.0f}",
            "Profit ($)":          f"{s['profit']:,.0f}",
            "Return multiple":     f"{(s['invested']+s['profit'])/s['invested']:.1f}x",
            "Sig1 p-value":        fmt_pval(sig1["pvalue"]),
            "Sig2 p-value":        fmt_pval(sig2["pvalue"]),
            "Sig3 median h before":str(sig3.get("median_hours_before", "N/A")),
            "Sig3 p-value":        fmt_pval(sig3.get("pvalue_timing", 1.0)),
            "Sig4 flags":          ", ".join(sig4.flags) if sig4.flags else "NONE",
            "Prosecuted":          "YES" if s["prosecuted"] else "NO",
        })

    df = pd.DataFrame(rows).set_index("Subject").T
    return df


# ===========================================================================
# 5.  MAIN
# ===========================================================================

def main() -> None:
    api_key = os.getenv("POLYGONSCAN_API_KEY")
    if not api_key:
        print("[INFO] POLYGONSCAN_API_KEY not set — wallet data will use hardcoded fallback.")

    # ---- Van Dyke ----
    print_section("Running signals on Van Dyke (confirmed insider, SDNY 2026)")
    vandyke_results = run_all_signals(
        trades=VANDYKE_TRADES,
        account_created=VANDYKE_ACCOUNT_CREATED,
        event_timestamp=EVENT_TIMESTAMP,
    )
    vd_invested = sum(t.cost_usdc for t in VANDYKE_TRADES)

    print_subject_report(
        label="Van Dyke (confirmed insider)",
        trades=VANDYKE_TRADES,
        account_created=VANDYKE_ACCOUNT_CREATED,
        total_invested=vd_invested,
        total_profit=VANDYKE_TOTAL_PROFIT,
        results=vandyke_results,
        used_fallback=False,
    )

    # ---- Additional wallets ----
    wallet_results = {}
    wallet_meta   = {}

    print_section("Fetching additional wallet data")
    for wkey in WALLETS:
        trades, acct_created, used_fallback = get_wallet_data(wkey, api_key)
        results = run_all_signals(
            trades=trades,
            account_created=acct_created,
            event_timestamp=EVENT_TIMESTAMP,
        )
        wallet_results[wkey] = results

        # Derive invested/profit from the actual trade records so the numbers
        # are consistent whether we used live data or the hardcoded fallback.
        # Each resolved YES share pays $1 USDC, so payout = sum(shares).
        actual_invested = sum(t.cost_usdc for t in trades)
        actual_payout   = sum(t.shares for t in trades if t.resolved_yes)
        actual_profit   = actual_payout - actual_invested

        wallet_meta[wkey] = {
            "trades":          trades,
            "account_created": acct_created,
            "used_fallback":   used_fallback,
            "actual_invested": actual_invested,
            "actual_profit":   actual_profit,
        }
        print_subject_report(
            label=wkey,
            trades=trades,
            account_created=acct_created,
            total_invested=actual_invested,
            total_profit=actual_profit,
            results=results,
            used_fallback=used_fallback,
        )

    # ---- Comparison table ----
    print_section("SIDE-BY-SIDE COMPARISON TABLE")

    subjects = [
        {
            "label":       "Van Dyke",
            "invested":    vd_invested,
            "profit":      VANDYKE_TOTAL_PROFIT,
            "results":     vandyke_results,
            "prosecuted":  True,
        },
        {
            "label":       "wallet_2_a72D",
            "invested":    wallet_meta["wallet_2_a72D"]["actual_invested"],
            "profit":      wallet_meta["wallet_2_a72D"]["actual_profit"],
            "results":     wallet_results["wallet_2_a72D"],
            "prosecuted":  False,
        },
        {
            "label":       "wallet_3_SBet365",
            "invested":    wallet_meta["wallet_3_SBet365"]["actual_invested"],
            "profit":      wallet_meta["wallet_3_SBet365"]["actual_profit"],
            "results":     wallet_results["wallet_3_SBet365"],
            "prosecuted":  False,
        },
    ]

    df = build_comparison_table(subjects)

    # Pretty-print the transposed DataFrame
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 120)
    print()
    print(df.to_string())

    print("\n  Significance key: *** p<0.001  ** p<0.01  * p<0.05")
    print("  Signal 1: binomial test (win rate vs market odds)")
    print("  Signal 2: bootstrap return test (100k simulations)")
    print("  Signal 3: timing concentration (trades within 48h of event)")
    print("  Signal 4: account age / concentration / burst flags\n")

    # ---- Save comparison table ----
    out_path = os.path.join(os.path.dirname(__file__), "data", "comparison_table.csv")
    df.to_csv(out_path)
    print(f"  [INFO] Comparison table saved to {out_path}")


if __name__ == "__main__":
    main()
