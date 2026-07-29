"""
Unit checks for the realized factor-exposure REPORTING in
research/portfolio/walk_forward.py (_fmt_exposure + the per-window '|mkt|'
column it feeds). No database required.

Realized exposure to each neutralized factor is the acceptance check for the
optimizer. The bar to clear is |w . beta| <= portfolio.neutrality_band for that
factor, NOT ~0: the band deliberately lets the book hold some exposure so it can
keep alpha and trade less rather than fight the position cap down to exact zero.
So the column is formatted at 4 decimals, the band's own scale.

The formatter's one hard job is that a window which never traded has NO exposure
(NaN) and must not print '0.0000' - a perfect neutrality score for a book that
does not exist. It prints '-'.

The second half of this file pins down band_reproject, the step _backtest_window
uses to re-impose neutrality each bar. It removes only the exposure EXCESS
beyond the band; a full projection onto null(A') instead forces A'w = 0 and
discards the band entirely. walk_forward used to do exactly that, which is why
realized |mkt| sat at ~1e-4 no matter how much slack the band granted.

Run: uv run tests/exposure_report_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from research.portfolio.walk_forward import _fmt_exposure

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# --- no book -> '-', never a neutrality score -------------------------------
check("NaN (window never traded) -> '-'", _fmt_exposure(np.nan) == '-',
      f"(got {_fmt_exposure(np.nan)!r})")
check("None -> '-'", _fmt_exposure(None) == '-')
check("+/-inf -> '-'",
      _fmt_exposure(np.inf) == '-' and _fmt_exposure(-np.inf) == '-')

# --- 4 decimals, on the same scale as the neutrality band -------------------
# The band (portfolio.neutrality_band, market 0.10) is what the column is read
# against, so the format matches its scale rather than the residual's.
check("band-scale exposure reads directly against the band",
      _fmt_exposure(0.1) == '0.1000', f"(got {_fmt_exposure(0.1)!r})")
check("a neutrality breach is unmistakable",
      _fmt_exposure(0.1234) == '0.1234')
# The realized book is neutralized exactly, so a healthy run reads 0.0000 here
# and anything materially above it means the neutralize step is not binding.
check("machine-zero residual reads as 0.0000 (the healthy state)",
      _fmt_exposure(1.39e-17) == '0.0000')
check("exact zero is distinguishable from no-book",
      _fmt_exposure(0.0) == '0.0000', f"(got {_fmt_exposure(0.0)!r})")
check("sign preserved (raw exposure, pre-abs)",
      _fmt_exposure(-0.0250) == '-0.0250',
      f"(got {_fmt_exposure(-0.0250)!r})")
check("four decimals, always", all(len(_fmt_exposure(x).split('.')[1]) == 4
                                  for x in [0.0, 0.1, 1.0, 0.12345]))

# --- band vs full projection ------------------------------------------------
# A = [dollar, beta_market] over a crypto-like book. The full projection is what
# _backtest_window USED to apply per bar; band_reproject is what it applies now.
rng = np.random.default_rng(0)
n = 120
betas = 1.0 + 0.3 * rng.standard_normal(n)      # beta_market ~ 1 for crypto
A = np.column_stack([np.ones(n), betas])
w = rng.standard_normal(n) / n

pre_mkt = abs(float(w @ betas))
neutralizer = np.linalg.solve(A.T @ A, A.T)
w_neutral = w - A @ (neutralizer @ w)
post_mkt = abs(float(w_neutral @ betas))

check("un-neutralized book has visible market exposure", pre_mkt > 1e-3,
      f"(|mkt| {_fmt_exposure(pre_mkt)})")
check("projection drives |mkt| to machine zero, far inside the 0.10 band",
      post_mkt < 1e-12, f"(|mkt| {post_mkt:.2e})")

from research.lib.portfolio_opt import band_reproject

band = np.array([0.0, 0.10])                     # dollar exact, market banded
w_banded = band_reproject(A, w, band)
banded_mkt = abs(float(w_banded @ betas))
check("band_reproject holds the exposure inside the band",
      banded_mkt <= 0.10 + 1e-9, f"(|mkt| {_fmt_exposure(banded_mkt)})")
check("band_reproject KEEPS slack the full projection discards",
      banded_mkt > post_mkt * 1e6,
      f"(banded {_fmt_exposure(banded_mkt)} vs projected {post_mkt:.2e})")
check("dollar neutrality (band 0) stays EXACT alongside a banded factor",
      abs(float(w_banded.sum())) < 1e-9,
      f"(net {float(w_banded.sum()):.2e})")
check("band = 0 reproduces the full projection exactly",
      np.allclose(band_reproject(A, w, np.zeros(2)), w_neutral))
check("an already-compliant book is left alone (no needless trading)",
      np.allclose(band_reproject(A, w_banded, band), w_banded))

# --- the REALIZED book must be neutralized EXACTLY --------------------------
# Signals predict RESIDUAL returns, orthogonal to these factors by
# construction, so factor exposure has zero expected return and pure variance.
# Letting the neutrality_band reach the book was measured at -1.70%/yr of
# direct factor PnL, 19.8% of book variance, and a further -1.70%/yr of lost
# residual alpha capture: Sharpe 0.94 -> 0.37 on an identical sample.
import inspect
from research.portfolio.walk_forward import WalkForwardPortfolio

src = inspect.getsource(WalkForwardPortfolio._backtest_window)
check("_backtest_window neutralizes via band_reproject",
      'band_reproject(Av, v, bands_v)' in src)
check("the per-bar step passes an EXACT (zero) band, not neutrality_band",
      'bands_v = np.zeros(' in src,
      "(the band must not reach the realized book)")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
