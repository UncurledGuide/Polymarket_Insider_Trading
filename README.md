# Polymarket Insider Trading Detection

A four-signal statistical framework for detecting informed trading on Polymarket,
validated against a real, publicly indicted case.

## Background

On-chain prediction markets create a paper trail that traditional insider trading
doesn't: every trade, timestamp, and wallet is public. This project asks whether
that paper trail is enough to flag informed trading using statistics alone, no
inside information about who the trader is.

Ground truth: **US v. Gannon Ken Van Dyke (SDNY, 2026)**. Van Dyke, a DoD contractor,
traded Polymarket contracts tied to a Venezuela military operation he had advance
knowledge of. He is the only trader in this dataset confirmed to be an insider, which
makes him the benchmark every signal is validated against.

## Methodology

Four independent signals, each designed to be interpretable as evidence on its own:

| Signal | What it tests |
|---|---|
| **Binomial test** | Is the win rate too high to be explained by the market's own implied odds? |
| **Bootstrap return (100K simulations)** | Could a random trader deploying the same capital across the same number of trades plausibly hit this return? |
| **Timing z-score** | Do trades cluster suspiciously close to the resolving event, versus a uniform null over the market's lifetime? |
| **Account flags** | Rule-based checks: new account, capital concentrated in one market, capital burst-deployed in a short window |

## Results

| | Van Dyke | wallet_2_a72D | wallet_3_SBet365 |
|---|---|---|---|
| Invested | $34,352 | $5,783 | $25,090 |
| Profit | $409,881 | $74,982 | $145,620 |
| Return multiple | 12.9x | 14.0x | 6.8x |
| Binomial p-value | 0.0000 *** | 0.0716 | 0.0216 * |
| Bootstrap p-value | 0.0000 *** | 0.0023 ** | 0.0060 ** |
| Median hours before event | 40.4 | 11.6 | 5.6 |
| Timing p-value | 0.0000 *** | 0.0001 *** | 0.0000 *** |
| Account flags | NEW_ACCOUNT, HIGH_CONCENTRATION, BURST_TRADING | HIGH_CONCENTRATION, BURST_TRADING | NEW_ACCOUNT, BURST_TRADING |
| Prosecuted | **YES** | No | No |

Van Dyke is significant across all four signals, which is exactly what you'd want
from a framework being validated against a known case. The other two wallets show
partial significance and have not been prosecuted, useful as a reminder that
statistical suspicion isn't proof.

## What this is part of

This framework is the validation layer for a larger project,
[`PredictionMarkets`](https://github.com/UncurledGuide/PredictionMarkets), which
scans thousands of resolved markets to test whether these same signals predict
systematic resolution bias at scale.

## Stack

Python, NumPy, SciPy (`stats.binomtest`, `stats.norm`), live on-chain data via
Etherscan V2 (Polygon), with hardcoded fallback for reproducibility.

## Usage

\`\`\`bash
python main.py   # reads POLYGONSCAN_API_KEY from .env automatically
\`\`\`
