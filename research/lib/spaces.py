"""
Cross-sectional statistical-arbitrage SPACES.

A "space" is one economic hypothesis: relative cross-sectional displacement in
space S predicts idiosyncratic (residual) returns. evaluate.py z-scores the raw
value cross-sectionally; walk_forward selects which spaces have live edge each
training window. Outcome-agnostic - no returns are inspected here.

Add a space = add one `_S(...)` line. Sign is resolved from in-sample IC, so the
formula is direction-neutral.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from config import get


@dataclass(frozen=True)
class SpaceDef:
    name: str
    columns: Tuple[str, ...]      # feature columns read (column-projected loading)
    theme: str                    # economic family (drives walk_forward family cap)
    rationale: str
    op: str = 'direct'            # direct | spread | ratio | diff
    lag: int = 0                  # for op='diff' (in BASE bars, not screening bars)
    halflife: Optional[float] = None  # per-space EWM smoothing; None -> global default

    @property
    def signal_type(self) -> str:
        return f'space_{self.op}'

    @property
    def category(self) -> str:
        return self.theme

    @property
    def direction(self) -> int:
        return 1


def _S(name, col, theme, rationale, op='direct', lag=0, col2=None, halflife=None):
    cols = (col,) if col2 is None else (col, col2)
    return SpaceDef(name=name, columns=cols, theme=theme, rationale=rationale,
                    op=op, lag=lag, halflife=halflife)


# =============================================================================
# The space library. Each line is one hypothesis. Grouped by economic theme.
# =============================================================================
# The hand-curated signal library is RETIRED: discovery
# (research/signals/) is the only signal source - promoted
# candidates enter the registry as disc_* entries via
# research/lib/discovered.py. The SpaceDef machinery below is kept
# because those entries reuse it (compute_space_raw's 'dsl' op), and a
# hand-written hypothesis can always be re-added as one _S(...) line.
SPACES = []


# NOTE: the explicit smoothing-halflife variant families (space_*_h0/h12/h36)
# were retired: evaluation now smooths every signal at a halflife matched to
# the scored lag (signals.lag_smoothing; see evaluate.smoothing_halflife_for_lag),
# which covers the speed dimension for the WHOLE library instead of five
# hand-picked bases - and removes 15 near-duplicate entries from the
# multiple-testing budget. Per-space `halflife=` overrides remain supported
# (they act as a floor under the per-lag halflife).


def compute_space_raw(space: SpaceDef, features: pd.DataFrame) -> pd.Series:
    """Raw (pre-normalization) value of a space - a vectorized expression over
    feature columns. evaluate.py applies the cross-sectional z-score + neutrality.
    """
    c = space.columns
    if space.op == 'direct':
        return features[c[0]]
    if space.op == 'spread':
        return features[c[0]] - features[c[1]]
    if space.op == 'ratio':
        return features[c[0]] / (features[c[1]].abs() + 1e-9)
    if space.op == 'diff':
        return features[c[0]] - features.groupby('symbol', sort=False)[c[0]].shift(space.lag)
    if space.op == 'dsl':
        # Promoted discovery candidate (research/lib/discovered.py): compile
        # its DSL program on the feature frame. compile_candidate requires
        # (symbol, timestamp)-sorted input (rolling operators) - both callers
        # (evaluate._load_features, walk_forward.composite_scores) sort that
        # way. Its output is already cross-sectionally z-scored; evaluate's
        # final z-score on top is idempotent in distribution, so a discovered
        # signal's numbers mean the same thing as a curated space's.
        from research.signals.generation import compile_candidate
        sig = compile_candidate(space.candidate,
                                features[['timestamp', 'symbol'] + list(c)])
        aligned = features[['timestamp', 'symbol']].merge(
            sig, on=['timestamp', 'symbol'], how='left')
        return pd.Series(aligned['signal'].values, index=features.index)
    raise ValueError(f"unknown space op: {space.op}")


def build_registry_entries(smoothing_halflife: Optional[float] = None) -> dict:
    """Registry entries {name: info} consumed by research.lib.signal_eval."""
    if smoothing_halflife is None:
        smoothing_halflife = get('signals.spaces.smoothing_halflife',
                                 get('signals.smoothing_halflife', 3))
    entries = {}
    for sp in SPACES:
        # A per-space halflife (smoothing-speed variants) overrides the global
        # default; None falls back to it.
        hl = sp.halflife if sp.halflife is not None else smoothing_halflife
        entries[f'space_{sp.name}'] = {
            'signal_def': sp,
            'description': sp.rationale,
            'category': sp.theme,
            'direction': 1,
            'kind': 'space',
            'smoothing_halflife': hl,
            'family': sp.theme,
        }
    return entries
