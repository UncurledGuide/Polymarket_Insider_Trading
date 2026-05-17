"""
signals.py — Four statistical signals for detecting insider trading on Polymarket.

Each signal is designed to be interpretable as evidence in a research or legal context:
  1. Binomial p-value:    Is the win rate statistically consistent with chance given market odds?
  2. Bootstrap return:    Is the aggregate return achievable by a random trader with equal capital?
  3. Timing signal:       How far ahead of the resolution event did trades cluster?
  4. Account flags:       Hard rule checks on account age and trade concentration.

All signals accept a list of trade dicts with standard keys; see TradeRecord below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Shared trade schema
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Canonical representation of a single Polymarket trade."""
    date: str                       # ISO date "YYYY-MM-DD" (always required)
    market: str                     # Human-readable market name
    shares: float                   # Number of YES shares purchased
    price: float                    # Price per share (= implied probability, 0–1)
    cost_usdc: float                # Actual USDC spent
    resolved_yes: bool = True       # Did the YES side win?
    datetime_utc: Optional[str] = None  # Full UTC datetime "YYYY-MM-DD HH:MM:SS" when known


# ---------------------------------------------------------------------------
# Signal 1 — Binomial p-value on win rate vs market-implied odds
# ---------------------------------------------------------------------------

def signal_binomial_pvalue(trades: list[TradeRecord]) -> dict:
    """
    Treat each trade as a Bernoulli trial where the "null" success probability
    equals the market price at time of purchase (the crowd's implied probability).

    H0: the trader wins at the rate the market predicted.
    H1: the trader wins at a higher rate than the market predicted.

    We weight trials by USDC invested so that a $7k bet counts more than a $96 bet.
    The effective number of trials and the weighted success probability are derived
    from the portfolio of trades.

    Returns a one-tailed p-value; values below 0.05 are conventionally significant.
    """
    if not trades:
        return {"pvalue": 1.0, "n_trades": 0, "weighted_avg_price": None,
                "win_rate": None, "note": "no trades"}

    total_cost = sum(t.cost_usdc for t in trades)
    n = len(trades)

    # Weighted average market-implied probability (null hypothesis win rate)
    weights = np.array([t.cost_usdc / total_cost for t in trades])
    null_p = float(np.dot(weights, [t.price for t in trades]))

    # Observed wins (weighted)
    wins = sum(t.cost_usdc for t in trades if t.resolved_yes)
    observed_win_rate = wins / total_cost  # = 1.0 if all won

    # Effective number of independent trials via the "portfolio binomial" approach:
    # treat total capital as n_eff independent unit bets each winning at null_p.
    # We use the actual number of distinct trades as n (conservative).
    n_eff = n
    k_eff = sum(1 for t in trades if t.resolved_yes)

    # One-tailed binomial test: P(X >= k | n, p)
    pvalue = float(stats.binomtest(k_eff, n_eff, null_p, alternative="greater").pvalue)

    return {
        "pvalue": pvalue,
        "n_trades": n,
        "n_wins": k_eff,
        "weighted_avg_market_price": round(null_p, 4),
        "observed_win_rate": round(observed_win_rate, 4),
    }


# ---------------------------------------------------------------------------
# Signal 2 — Bootstrap return test (100 k simulations)
# ---------------------------------------------------------------------------

def signal_bootstrap_return(
    trades: list[TradeRecord],
    n_simulations: int = 100_000,
    rng_seed: int = 42,
) -> dict:
    """
    Simulate 100,000 counterfactual traders who each deploy the same total USDC
    across the same number of trades but pick random market prices (drawn from a
    realistic distribution of Polymarket prices).

    The empirical p-value is the fraction of simulations that achieve a return
    multiple >= the observed return multiple.

    Price distribution: uniform over [0.03, 0.97] clipped to observed market range,
    reflecting that prediction markets span the full probability range.
    Each simulated "trade" wins with probability equal to its drawn price.

    Returns the empirical p-value and descriptive stats of the null distribution.
    """
    if not trades:
        return {"pvalue": 1.0, "note": "no trades"}

    total_cost = sum(t.cost_usdc for t in trades)
    # Observed return: sum of (shares * $1) for winning trades / total invested
    observed_pnl = sum(t.shares for t in trades if t.resolved_yes) - total_cost
    observed_multiple = (observed_pnl + total_cost) / total_cost  # e.g. 12.4

    rng = np.random.default_rng(rng_seed)
    n = len(trades)

    # Weight each trade by its fraction of total investment
    cost_fractions = np.array([t.cost_usdc / total_cost for t in trades])
    # Absolute USDC per simulated trade (same allocation as observed)
    usdc_per_trade = cost_fractions * total_cost

    sim_multiples = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Draw a random implied probability for each trade slot
        sim_prices = rng.uniform(0.03, 0.97, size=n)
        # Each trade wins with probability = sim_price
        outcomes = rng.random(size=n) < sim_prices
        # Shares purchased = cost / price (same as real trader)
        sim_shares = usdc_per_trade / sim_prices
        # PnL: winning shares pay $1 each; losing trades lose the stake
        sim_pnl = np.sum(sim_shares * outcomes) - total_cost
        sim_multiples[i] = (sim_pnl + total_cost) / total_cost

    empirical_pvalue = float(np.mean(sim_multiples >= observed_multiple))

    return {
        "pvalue": empirical_pvalue,
        "observed_multiple": round(observed_multiple, 2),
        "sim_median_multiple": round(float(np.median(sim_multiples)), 3),
        "sim_p95_multiple": round(float(np.percentile(sim_multiples, 95)), 3),
        "sim_p99_multiple": round(float(np.percentile(sim_multiples, 99)), 3),
        "n_simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# Signal 3 — Time-before-event distribution
# ---------------------------------------------------------------------------

def signal_timing(
    trades: list[TradeRecord],
    event_timestamp: str,
    event_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> dict:
    """
    Compute how many hours before the resolution-triggering event each trade occurred.

    Insider trading characteristically clusters in a narrow window immediately before
    the event — random traders spread uniformly over the market lifetime.

    Returns:
      - hours_before_event per trade
      - mean and median hours before event
      - fraction of trades occurring within 48h (the "last-day concentration")
      - a concentration z-score vs. the null hypothesis that trades are uniform
        over the market lifetime (approximated as 30 days = 720 hours).
    """
    if not trades:
        return {"note": "no trades"}

    event_dt = datetime.strptime(event_timestamp, event_fmt)
    MARKET_LIFETIME_HOURS = 720  # ~30 days; used as null distribution width

    hours_before = []
    for t in trades:
        # Prefer full UTC datetime when available (blockchain data); fall back to date-only
        if t.datetime_utc:
            trade_dt = datetime.strptime(t.datetime_utc, "%Y-%m-%d %H:%M:%S")
        else:
            trade_dt = datetime.strptime(t.date, "%Y-%m-%d")
        delta = event_dt - trade_dt
        hours_before.append(delta.total_seconds() / 3600)

    hours_arr = np.array(hours_before)

    # Fraction within 48 hours of the event
    within_48h = float(np.mean(hours_arr <= 48))
    # Fraction within 7 days (168h)
    within_7d = float(np.mean(hours_arr <= 168))

    # Under H0 (uniform), expected fraction within 48h = 48/720 ≈ 0.067
    null_frac_48h = 48 / MARKET_LIFETIME_HOURS
    n = len(hours_arr)
    # Binomial z-score for concentration in the 48h window
    se = math.sqrt(null_frac_48h * (1 - null_frac_48h) / n)
    z_48h = (within_48h - null_frac_48h) / se if se > 0 else float("inf")
    pvalue_timing = float(stats.norm.sf(z_48h))  # one-tailed

    return {
        "hours_before_event_per_trade": [round(h, 1) for h in hours_before],
        "mean_hours_before": round(float(np.mean(hours_arr)), 1),
        "median_hours_before": round(float(np.median(hours_arr)), 1),
        "min_hours_before": round(float(np.min(hours_arr)), 1),
        "max_hours_before": round(float(np.max(hours_arr)), 1),
        "frac_within_48h": round(within_48h, 3),
        "frac_within_7d": round(within_7d, 3),
        "z_score_48h_concentration": round(z_48h, 2),
        "pvalue_timing": round(pvalue_timing, 6),
    }


# ---------------------------------------------------------------------------
# Signal 4 — Account age + trade concentration flags
# ---------------------------------------------------------------------------

@dataclass
class AccountFlags:
    """Results of the rule-based account and concentration checks."""
    account_age_days: Optional[int]
    days_active_trading: int
    n_markets_traded: int
    top_market_pct: float          # fraction of capital in single market
    account_new_flag: bool         # account < 7 days old at first trade
    concentration_flag: bool       # >80% capital in a single market
    burst_flag: bool               # >50% of capital deployed in ≤3 days
    flags: list[str] = field(default_factory=list)
    flag_count: int = 0

    def __post_init__(self):
        if self.account_new_flag:
            self.flags.append("NEW_ACCOUNT")
        if self.concentration_flag:
            self.flags.append("HIGH_CONCENTRATION")
        if self.burst_flag:
            self.flags.append("BURST_TRADING")
        self.flag_count = len(self.flags)


def signal_account_flags(
    trades: list[TradeRecord],
    account_created: str,
    event_timestamp: str,
    date_fmt: str = "%Y-%m-%d",
) -> AccountFlags:
    """
    Rule-based checks inspired by exchange surveillance systems:

    NEW_ACCOUNT:        Account was created ≤7 days before the first trade.
                        Burner/throwaway accounts are a classic evasion technique.
    HIGH_CONCENTRATION: >80% of total USDC invested in a single market.
                        Legitimate diversified traders rarely bet everything on one event.
    BURST_TRADING:      >50% of total capital deployed within a 3-day window.
                        Sudden large bets before a specific event are a timing red flag.
    """
    if not trades:
        return AccountFlags(
            account_age_days=None, days_active_trading=0, n_markets_traded=0,
            top_market_pct=0.0, account_new_flag=False,
            concentration_flag=False, burst_flag=False,
        )

    created_dt = datetime.strptime(account_created, date_fmt)
    event_dt = datetime.strptime(event_timestamp.split()[0], date_fmt)

    trade_dates = sorted(set(t.date for t in trades))
    first_trade_dt = datetime.strptime(trade_dates[0], date_fmt)
    last_trade_dt = datetime.strptime(trade_dates[-1], date_fmt)

    account_age_days = (first_trade_dt - created_dt).days
    days_active = (last_trade_dt - first_trade_dt).days + 1

    # Market concentration
    market_costs: dict[str, float] = {}
    for t in trades:
        market_costs[t.market] = market_costs.get(t.market, 0) + t.cost_usdc
    total_cost = sum(market_costs.values())
    top_market_pct = max(market_costs.values()) / total_cost if total_cost > 0 else 0

    # Burst: find the 3-day window with the highest capital concentration
    all_dates = sorted(set(t.date for t in trades))
    max_burst_frac = 0.0
    for start_date_str in all_dates:
        start_dt = datetime.strptime(start_date_str, date_fmt)
        end_dt = start_dt + timedelta(days=2)
        window_cost = sum(
            t.cost_usdc for t in trades
            if start_dt <= datetime.strptime(t.date, date_fmt) <= end_dt
        )
        max_burst_frac = max(max_burst_frac, window_cost / total_cost if total_cost > 0 else 0)

    return AccountFlags(
        account_age_days=account_age_days,
        days_active_trading=days_active,
        n_markets_traded=len(market_costs),
        top_market_pct=round(top_market_pct, 3),
        account_new_flag=(account_age_days <= 7),
        concentration_flag=(top_market_pct > 0.80),
        burst_flag=(max_burst_frac > 0.50),
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_all_signals(
    trades: list[TradeRecord],
    account_created: str,
    event_timestamp: str,
    n_bootstrap: int = 100_000,
) -> dict:
    """Run all four signals and return a combined result dict."""
    return {
        "signal_1_binomial": signal_binomial_pvalue(trades),
        "signal_2_bootstrap": signal_bootstrap_return(trades, n_simulations=n_bootstrap),
        "signal_3_timing": signal_timing(trades, event_timestamp),
        "signal_4_flags": signal_account_flags(trades, account_created, event_timestamp),
    }
