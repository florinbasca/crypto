"""
Bridge: promoted discovery candidates -> first-class production signals.

discovery.py promotes candidates (bounded-DSL programs) through its gates
and stores them in the discovery promotions table(s). This module turns those
rows into registry entries of the SAME shape as the curated spaces
(research/lib/spaces.py), so the ordinary pipeline picks them up with no
manual translation step:

    discovery.py        # promotes candidates
    walk_forward.py     # scores disc_* in memory, selects, backtests

Timing honesty: a candidate's EXPRESSION was chosen by a search that saw data
up to its promotion roll, so letting the walk-forward select it in windows
BEFORE that date is time travel (the signal definition did not exist yet).
Every entry therefore carries valid_from = the promotion roll's OOS start,
and the walk-forward selector drops discovered signals from windows whose
training end precedes it.

Disable the whole bridge with signals.include_discovered: false.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd

from config import get


@dataclass(frozen=True)
class DiscoveredDef:
    """SpaceDef-shaped wrapper around a discovery Candidate (op='dsl').

    compute_space_raw dispatches on op; signal_feature_columns reads
    .columns - the same interface the curated spaces expose, so evaluate.py
    and walk_forward.py need no special-casing beyond the 'dsl' op branch.
    """
    name: str
    columns: Tuple[str, ...]
    theme: str
    rationale: str
    candidate: object = field(compare=False)      # generation.Candidate
    op: str = 'dsl'
    lag: int = 0
    halflife: Optional[float] = None

    @property
    def signal_type(self) -> str:
        return 'space_dsl'

    @property
    def category(self) -> str:
        return self.theme

    @property
    def direction(self) -> int:
        return 1


def promotion_tables() -> list:
    """The promotions table(s) the bridge reads."""
    return [get('discovery', {})['tables']['promotions']]


def entries_from_promotions(promos: pd.DataFrame,
                            valid_from_by_roll: Optional[Dict[int, pd.Timestamp]] = None,
                            smoothing_halflife: Optional[float] = None) -> dict:
    """Registry entries {name: info} from promotion rows (pure, testable).

    Deduped by cand_hash (the content hash of the program): the row with the
    strongest |select_ic_tstat| wins the direction/metadata, and valid_from is
    the EARLIEST promotion date seen for that hash. Names are hash-stable
    (disc_<family>_<hash>) so scored stats stay consistent across runs.
    """
    from research.signals.agent.generation import Candidate, candidate_columns
    import json

    if promos is None or promos.empty:
        return {}
    if smoothing_halflife is None:
        smoothing_halflife = get('signals.spaces.smoothing_halflife',
                                 get('signals.smoothing_halflife', 3))
    valid_from_by_roll = valid_from_by_roll or {}

    entries: dict = {}
    strongest: Dict[str, float] = {}
    for _, row in promos.iterrows():
        try:
            cand = Candidate.from_dict(json.loads(row['candidate_json']))
        except Exception as e:
            logging.warning(f"discovered: unparseable candidate_json "
                            f"({row.get('cand_hash', '?')}): {e}")
            continue
        name = f"disc_{cand.family}_{cand.hash[:10]}"
        tstat = abs(float(row.get('select_ic_tstat', 0.0) or 0.0))
        vf = valid_from_by_roll.get(int(row.get('roll_id', -1)))

        if name in entries:
            prior_vf = entries[name]['valid_from']
            if vf is not None and (prior_vf is None or vf < prior_vf):
                entries[name]['valid_from'] = vf
            if tstat <= strongest[name]:
                continue

        strongest[name] = tstat
        prior_vf = entries[name]['valid_from'] if name in entries else None
        if prior_vf is not None and (vf is None or prior_vf < vf):
            vf = prior_vf
        half_life = float(row.get('half_life_bars', 0) or 0) or None
        sdef = DiscoveredDef(
            name=name,
            columns=tuple(sorted(candidate_columns(cand))),
            theme=f"disc_{cand.family}",
            rationale=cand.rationale or cand.name,
            candidate=cand,
            lag=int(row.get('target_lag', 0) or 0),
            halflife=half_life,
        )
        entries[name] = {
            'signal_def': sdef,
            'description': sdef.rationale,
            'category': sdef.theme,
            'direction': int(row.get('direction', 1) or 1),
            'kind': 'discovered',
            'smoothing_halflife': smoothing_halflife,
            'family': sdef.theme,
            'valid_from': vf,
            # Fitted alpha half-life (bars) from the discovery train profile:
            # caps the walk-forward's turnover-implied holding period, so a
            # fast-decaying signal is never aim-discounted as if its alpha
            # outlived its own term structure.
            'half_life_bars': half_life,
        }
    return entries


def load_discovered_entries() -> dict:
    """Load every promoted candidate from the discovery promotions tables.

    Returns {} when the bridge is disabled (signals.include_discovered),
    no promotions exist yet, or the DB is unavailable - the curated library
    keeps working either way.
    """
    if not get('signals.include_discovered', True):
        return {}
    try:
        from dbutil import load_data, table_exists
        frames = []
        for table in promotion_tables():
            if table_exists(table):
                df = load_data(table)
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            return {}
        promos = pd.concat(frames, ignore_index=True)

        from research.signals.agent.data import make_rolls
        rolls = make_rolls(get('discovery'))
        valid_from = {r.roll_id: r.oos_start for r in rolls}

        entries = entries_from_promotions(promos, valid_from)
        if entries:
            logging.info(f"discovered bridge: {len(entries)} promoted "
                         f"candidates registered as disc_* signals")
        return entries
    except Exception as e:
        logging.warning(f"discovered bridge unavailable: {e}")
        return {}
