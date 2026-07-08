"""
GENERATION: everything that creates and compiles candidate signals.

1. The bounded DSL - Candidate programs, the operator registry, validation
   (only allowed columns/ops/windows/size caps get through).
2. The compiler - Candidate -> cross-sectional signal panel with the same
   per-timestamp demean + z-score + clip (+-3) that research/lib/signal_eval.py
   applies to every registered signal, plus the truncation spot-check
   (causality on full compiled programs).
3. The proposers - the ONLY components allowed to invent candidates:
   RandomProposer (no-API baseline and control experiment) and LLMProposer
   (untrusted; sees compressed diagnostics only, emits DSL JSON; everything it
   returns is re-validated and re-scored by fixed code).

A Candidate is an immutable, JSON-serializable program: an expression tree over
allowed feature columns plus optional gate conditions. Expressions are nested
lists (S-expressions):

    ["col", "res_zscore"]
    ["neg", ["mul", ["sign", ["col", "res_zscore"]],
                    ["square", ["col", "res_zscore"]]]]
    ["roll_mean", ["col", "cs_rel_volume"], 144]
    ["where", ["lt", ["col", "cs_rel_volume"], 0.0], expr_a, expr_b]

Conditions are ["gt"|"lt"|"abs_gt"|"abs_lt", expression, threshold].

Causality is per-OPERATOR, not per-candidate: every operator here only reads
data through bar t inclusive (repo timing convention - targets start at t+1),
which tests/discovery_checks.py proves with a truncation test over the whole
registry. A candidate built from causal operators over real feature columns is
causal by construction; compile-time validation guarantees the second half
(only allowed columns, ops, windows, and size caps).
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from config import get

_EPS = 1e-10

CONDITION_OPS = ('gt', 'lt', 'abs_gt', 'abs_lt')


# =============================================================================
# Operator registry
# =============================================================================
# spec: n_args = expression arguments; extra = ('window'|'number', ...) trailing
# scalar parameters. fn signatures:
#   elementwise:      fn(*series) -> series
#   rolling:          fn(grouped_series, window) -> series  (per-symbol, causal)
#   cross_sectional:  fn(df, series) -> series              (per-timestamp)

def _sym_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def _sym_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _safe_div(a, b):
    return a / (np.abs(b) + _EPS)


def _roll(g, window, op):
    out = getattr(g.rolling(window, min_periods=max(2, window // 2)), op)()
    # groupby-rolling returns a (group, original) MultiIndex; realign
    return out.droplevel(0)


def _roll_zscore(g, window):
    m = _roll(g, window, 'mean')
    s = _roll(g, window, 'std')
    return (g.obj - m) / (s + _EPS)


def _roll_delta(g, window):
    return g.obj - g.shift(window)


def _cs_rank(df, x):
    # Centered percentile rank in [-0.5, 0.5] at each timestamp
    return x.groupby(df['timestamp']).rank(pct=True) - 0.5


def _cs_zscore(df, x):
    g = x.groupby(df['timestamp'])
    return (x - g.transform('mean')) / (g.transform('std') + _EPS)


def _cs_demean(df, x):
    return x - x.groupby(df['timestamp']).transform('mean')


OPERATORS: Dict[str, Dict] = {
    # elementwise unary
    'neg':       {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': lambda x: -x},
    'abs':       {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': np.abs},
    'sign':      {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': np.sign},
    'square':    {'kind': 'elementwise', 'n_args': 1, 'extra': (),
                  'fn': lambda x: np.sign(x) * x * x},   # sign-preserving x|x|
    'sqrt':      {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': _sym_sqrt},
    'log1p':     {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': _sym_log1p},
    'tanh':      {'kind': 'elementwise', 'n_args': 1, 'extra': (), 'fn': np.tanh},
    # elementwise binary
    'add':       {'kind': 'elementwise', 'n_args': 2, 'extra': (), 'fn': lambda a, b: a + b},
    'sub':       {'kind': 'elementwise', 'n_args': 2, 'extra': (), 'fn': lambda a, b: a - b},
    'mul':       {'kind': 'elementwise', 'n_args': 2, 'extra': (), 'fn': lambda a, b: a * b},
    'div':       {'kind': 'elementwise', 'n_args': 2, 'extra': (), 'fn': _safe_div},
    # per-symbol rolling (window from discovery.dsl.windows; data through t only)
    'roll_mean':   {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                    'fn': lambda g, w: _roll(g, w, 'mean')},
    'roll_std':    {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                    'fn': lambda g, w: _roll(g, w, 'std')},
    'roll_sum':    {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                    'fn': lambda g, w: _roll(g, w, 'sum')},
    'roll_zscore': {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                    'fn': _roll_zscore},
    'roll_delta':  {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                    'fn': _roll_delta},
    # cross-sectional (per timestamp)
    'cs_rank':   {'kind': 'cross_sectional', 'n_args': 1, 'extra': (), 'fn': _cs_rank},
    'cs_zscore': {'kind': 'cross_sectional', 'n_args': 1, 'extra': (), 'fn': _cs_zscore},
    'cs_demean': {'kind': 'cross_sectional', 'n_args': 1, 'extra': (), 'fn': _cs_demean},
    # conditional: ["where", condition, expr_if_true, expr_if_false]
    'where':     {'kind': 'where', 'n_args': 2, 'extra': ()},
}


# =============================================================================
# Candidate
# =============================================================================

def _to_tuple(node):
    if isinstance(node, (list, tuple)):
        return tuple(_to_tuple(x) for x in node)
    return node


def _to_list(node):
    if isinstance(node, tuple):
        return [_to_list(x) for x in node]
    return node


@dataclass(frozen=True)
class Candidate:
    """One signal program. Immutable; hash identifies it across rolls/runs."""
    name: str
    family: str
    expression: tuple                      # nested tuples (S-expression)
    conditions: tuple = ()                 # of (op, expression, threshold)
    rationale: str = ''

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'family': self.family,
            'expression': _to_list(self.expression),
            'conditions': _to_list(self.conditions),
            'rationale': self.rationale,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))

    @classmethod
    def from_dict(cls, d: dict) -> 'Candidate':
        return cls(
            name=str(d.get('name', '')),
            family=str(d.get('family', '')),
            expression=_to_tuple(d['expression']),
            conditions=_to_tuple(d.get('conditions', [])),
            rationale=str(d.get('rationale', '')),
        )

    @property
    def hash(self) -> str:
        """Content hash of the PROGRAM (name/rationale excluded): the same
        expression+conditions is the same candidate wherever it reappears."""
        payload = json.dumps(
            {'expression': _to_list(self.expression),
             'conditions': _to_list(self.conditions)},
            separators=(',', ':'))
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _count(node) -> int:
    """Nodes in an expression tree (operators + leaves)."""
    if not isinstance(node, tuple):
        return 0
    n = 1
    for arg in node[1:]:
        if isinstance(arg, tuple):
            n += _count(arg)
    return n


def complexity(cand: Candidate) -> int:
    """Total program size: expression nodes + condition nodes (each condition
    counts its expression plus one for the comparison)."""
    total = _count(cand.expression)
    for cond in cand.conditions:
        total += 1 + _count(cond[1])
    return total


def expression_columns(node, out: Optional[set] = None) -> set:
    """Feature columns referenced anywhere in an expression."""
    if out is None:
        out = set()
    if isinstance(node, tuple):
        if node and node[0] == 'col':
            out.add(node[1])
        else:
            for arg in node[1:]:
                expression_columns(arg, out)
    return out


def candidate_columns(cand: Candidate) -> set:
    cols = expression_columns(cand.expression)
    for cond in cand.conditions:
        expression_columns(cond[1], cols)
    return cols


def _depth(node) -> int:
    if not isinstance(node, tuple):
        return 0
    if node and node[0] == 'col':
        return 1
    return 1 + max((_depth(a) for a in node[1:] if isinstance(a, tuple)),
                   default=0)


# =============================================================================
# Structural (AST) similarity - diversity beyond output correlation
# =============================================================================

def _subtree_nodes(node, out=None) -> list:
    """Every sub-expression node of a tree (each tuple is a canonical,
    hashable structural key: operator + children + columns/windows)."""
    if out is None:
        out = []
    if isinstance(node, tuple):
        out.append(node)
        for arg in node[1:]:
            if isinstance(arg, tuple):
                _subtree_nodes(arg, out)
    return out


def candidate_subtrees(cand: Candidate, min_depth: int = 2) -> set:
    """A candidate's structural building blocks: sub-expressions of at least
    min_depth (bare column leaves excluded as too generic). Condition
    expressions count too."""
    nodes = _subtree_nodes(cand.expression)
    for cond in cand.conditions:
        nodes += _subtree_nodes(cond[1])
    return {n for n in nodes if _depth(n) >= min_depth}


def ast_similarity(a: Candidate, b: Candidate) -> float:
    """Jaccard overlap of two candidates' building blocks: 1.0 = identical
    structure, 0.0 = nothing shared. Catches near-clones that slip under
    output-correlation de-dup (same recipe, one column/window swapped)."""
    sa, sb = candidate_subtrees(a), candidate_subtrees(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# =============================================================================
# Validation (boundedness): only allowed columns, ops, windows, and size caps
# =============================================================================

class ValidationError(ValueError):
    pass


def _validate_expr(node, allowed_columns: set, windows: set, where_ok=True):
    if not isinstance(node, tuple) or len(node) < 2:
        raise ValidationError(f"malformed node: {node!r}")
    op = node[0]
    if op == 'col':
        if len(node) != 2 or node[1] not in allowed_columns:
            raise ValidationError(f"unknown column: {node[1]!r}")
        return
    if op == 'where':
        if not where_ok:
            raise ValidationError("nested 'where' not allowed inside a condition")
        if len(node) != 4:
            raise ValidationError("'where' takes (condition, expr_a, expr_b)")
        _validate_condition(node[1], allowed_columns, windows)
        _validate_expr(node[2], allowed_columns, windows)
        _validate_expr(node[3], allowed_columns, windows)
        return
    spec = OPERATORS.get(op)
    if spec is None:
        raise ValidationError(f"unknown operator: {op!r}")
    args = node[1:]
    n_expr, extra = spec['n_args'], spec['extra']
    if len(args) != n_expr + len(extra):
        raise ValidationError(f"{op}: expected {n_expr + len(extra)} args, "
                              f"got {len(args)}")
    for a in args[:n_expr]:
        _validate_expr(a, allowed_columns, windows)
    for kind, value in zip(extra, args[n_expr:]):
        if kind == 'window':
            if value not in windows:
                raise ValidationError(f"{op}: window {value!r} not in allowed "
                                      f"set {sorted(windows)}")
        elif not isinstance(value, (int, float)):
            raise ValidationError(f"{op}: scalar parameter expected, got {value!r}")


def _validate_condition(cond, allowed_columns: set, windows: set):
    if not isinstance(cond, tuple) or len(cond) != 3:
        raise ValidationError(f"malformed condition: {cond!r}")
    op, expr, threshold = cond
    if op not in CONDITION_OPS:
        raise ValidationError(f"unknown condition op: {op!r}")
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"condition threshold must be a number, "
                              f"got {threshold!r}")
    _validate_expr(expr, allowed_columns, windows, where_ok=False)


def validate_candidate(cand: Candidate, allowed_columns: Sequence[str],
                       dsl_cfg: Optional[dict] = None) -> None:
    """Raise ValidationError unless the candidate is inside the bounded space."""
    cfg = dsl_cfg or get('discovery.dsl', {})
    windows = set(cfg.get('windows', []))
    allowed = set(allowed_columns)

    _validate_expr(cand.expression, allowed, windows)
    if len(cand.conditions) > int(cfg.get('max_conditions', 2)):
        raise ValidationError(
            f"too many conditions: {len(cand.conditions)}")
    for cond in cand.conditions:
        _validate_condition(cond, allowed, windows)
    if _depth(cand.expression) > int(cfg.get('max_depth', 4)):
        raise ValidationError("expression too deep")
    if complexity(cand) > int(cfg.get('max_nodes', 24)):
        raise ValidationError("program too large")


# =============================================================================
# Evaluation
# =============================================================================

def eval_expr(node, df: pd.DataFrame) -> pd.Series:
    """Evaluate an expression to a float Series aligned to df's index.

    df: panel SORTED by (symbol, timestamp) with 'timestamp', 'symbol' and the
    referenced feature columns. Rolling operators group by symbol; cross-
    sectional operators group by timestamp. Fully vectorized.
    """
    op = node[0]
    if op == 'col':
        return df[node[1]].astype(float)
    if op == 'where':
        mask = eval_condition(node[1], df)
        a = eval_expr(node[2], df)
        b = eval_expr(node[3], df)
        return a.where(mask, b)
    spec = OPERATORS[op]
    n_expr = spec['n_args']
    args = [eval_expr(a, df) for a in node[1:1 + n_expr]]
    extra = node[1 + n_expr:]
    if spec['kind'] == 'elementwise':
        return spec['fn'](*args)
    if spec['kind'] == 'rolling':
        g = args[0].groupby(df['symbol'])
        return spec['fn'](g, int(extra[0]))
    if spec['kind'] == 'cross_sectional':
        return spec['fn'](df, args[0])
    raise ValidationError(f"cannot evaluate operator {op!r}")


def eval_condition(cond, df: pd.DataFrame) -> pd.Series:
    """Boolean Series for one gate condition. NaN inputs -> False."""
    op, expr, threshold = cond
    x = eval_expr(expr, df)
    if op == 'gt':
        out = x > threshold
    elif op == 'lt':
        out = x < threshold
    elif op == 'abs_gt':
        out = np.abs(x) > threshold
    elif op == 'abs_lt':
        out = np.abs(x) < threshold
    else:
        raise ValidationError(f"unknown condition op {op!r}")
    return out.fillna(False)


# =============================================================================
# Compiler: Candidate -> cross-sectional signal panel
# =============================================================================

_CLIP = 3.0


def compile_candidate(cand: Candidate, panel: pd.DataFrame,
                      allowed_columns=None,
                      dsl_cfg: Optional[dict] = None) -> pd.DataFrame:
    """Compile one candidate into [timestamp, symbol, signal].

    panel must be sorted by (symbol, timestamp) - rolling operators depend on
    it. Validation runs first when allowed_columns is given (the search always
    passes it; tests may skip). Pipeline (hard requirement from agent.md):
    evaluate the expression, apply gate conditions (gated-off rows are neutral
    0, keeping the cross-section intact), then per-timestamp cross-sectional
    demean + z-score + clip (+-3) - the same normalization
    research/lib/signal_eval.py applies to every registered signal, so a
    discovered signal's numbers are on the same scale as any other's.
    """
    if allowed_columns is not None:
        validate_candidate(cand, allowed_columns, dsl_cfg)

    raw = eval_expr(cand.expression, panel)

    if cand.conditions:
        mask = pd.Series(True, index=panel.index)
        for cond in cand.conditions:
            mask &= eval_condition(cond, panel)
        # Outside the gate the signal is NEUTRAL (0), not missing: the
        # cross-section keeps its full width and the book simply holds no
        # position in gated-off names.
        raw = raw.where(mask | raw.isna(), 0.0)

    out = panel[['timestamp', 'symbol']].copy()
    out['signal'] = raw.astype(float).replace([np.inf, -np.inf], np.nan)

    g = out.groupby('timestamp')['signal']
    out['signal'] = (out['signal'] - g.transform('mean')) / (g.transform('std') + _EPS)
    out['signal'] = out['signal'].clip(-_CLIP, _CLIP)
    return out.dropna(subset=['signal']).reset_index(drop=True)


def truncation_check(cand: Candidate, panel: pd.DataFrame,
                     cut_frac: float = 0.75, atol: float = 1e-8) -> bool:
    """True when the compiled signal is causal at the cut timestamp.

    Compiles on the full panel and on the panel truncated at a cut timestamp;
    the LAST truncated cross-section must be identical in both. Any operator
    peeking past t shifts those values. The per-operator guarantee lives in
    tests/discovery_checks.py; this is the belt-and-suspenders sample applied
    to full programs.
    """
    stamps = np.sort(panel['timestamp'].unique())
    if len(stamps) < 8:
        raise ValueError("panel too short for a truncation check")
    cut_ts = stamps[int(len(stamps) * cut_frac)]

    full = compile_candidate(cand, panel)
    trunc_panel = panel[panel['timestamp'] <= cut_ts]
    trunc = compile_candidate(cand, trunc_panel.reset_index(drop=True))

    a = full[full['timestamp'] == cut_ts].set_index('symbol')['signal']
    b = trunc[trunc['timestamp'] == cut_ts].set_index('symbol')['signal']
    if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
        return False
    b = b.reindex(a.index)
    return bool(np.allclose(a.values, b.values, atol=atol, equal_nan=True))


# =============================================================================
# Proposers
# =============================================================================

class Proposer(ABC):
    """A proposer reads compressed diagnostics + the current parents and
    returns Candidate objects. It never touches data, never runs code, never
    scores anything - the harness re-validates and re-scores everything it
    returns. Implementations must be swappable: the harness runs identically
    with the random-mutation baseline (no API) and the LLM proposer."""

    @abstractmethod
    def propose(self, n: int, family: str, diagnostics: dict,
                parents: Sequence[Candidate],
                family_columns: Dict[str, list],
                rng: np.random.Generator,
                parent_scores: Optional[dict] = None,
                failures: Optional[list] = None,
                overused: Optional[list] = None) -> list:
        """Return up to n candidates for one family. Invalid programs are fine
        to return - the harness validates and drops them.

        parent_scores: {candidate_hash: {reward, ic_tstat, half_life_bars}} so
          an API proposer can see HOW WELL each parent did, not just its form.
        failures: [{expression, conditions, reward}] of recently-culled
          low-scoring candidates - things to avoid re-proposing.
        overused: over-mined structural building blocks (sub-expressions) to
          vary away from - the frequent-subtree-avoidance hint. All three are
          hints for API proposers; the random baseline ignores them."""

    def usage_snapshot(self) -> dict:
        """Cumulative API usage. Zero for non-API proposers; API proposers
        override with their live accumulator (see _ApiProposer)."""
        return {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}


_UNARY = ['neg', 'abs', 'sign', 'square', 'sqrt', 'log1p', 'tanh']
_BINARY = ['add', 'sub', 'mul', 'div']
_ROLLING = ['roll_mean', 'roll_std', 'roll_sum', 'roll_zscore', 'roll_delta']
_CS = ['cs_rank', 'cs_zscore', 'cs_demean']
_GT_LT_THRESHOLDS = [-1.0, -0.5, 0.0, 0.5, 1.0]
_ABS_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


class RandomProposer(Proposer):
    """Random-mutation proposer: samples fresh programs from the DSL grammar
    and point-mutates parents (replace a subtree, wrap in a unary, swap a
    column/window, add/drop a gate). Deterministic given the rng. If the
    search cannot beat this with the LLM plugged in, the LLM is not adding
    value - that control experiment is the point of keeping this first-class."""

    def __init__(self, dsl_cfg: Optional[dict] = None,
                 mutation_prob: Optional[float] = None):
        self.dsl_cfg = dsl_cfg or get('discovery.dsl', {})
        if mutation_prob is None:
            mutation_prob = get('discovery.search.mutation_prob', 0.5)
        self.mutation_prob = float(mutation_prob)

    # -- grammar sampling ---------------------------------------------------

    def _leaf(self, rng, cols) -> tuple:
        return ('col', cols[int(rng.integers(len(cols)))])

    def _random_expr(self, rng, cols, depth_left: int) -> tuple:
        if depth_left <= 1 or rng.random() < 0.35:
            return self._leaf(rng, cols)
        kind = rng.random()
        if kind < 0.35:
            op = _UNARY[int(rng.integers(len(_UNARY)))]
            return (op, self._random_expr(rng, cols, depth_left - 1))
        if kind < 0.60:
            op = _BINARY[int(rng.integers(len(_BINARY)))]
            return (op, self._random_expr(rng, cols, depth_left - 1),
                    self._random_expr(rng, cols, depth_left - 1))
        if kind < 0.85:
            op = _ROLLING[int(rng.integers(len(_ROLLING)))]
            windows = self.dsl_cfg['windows']
            w = int(windows[int(rng.integers(len(windows)))])
            return (op, self._random_expr(rng, cols, depth_left - 1), w)
        op = _CS[int(rng.integers(len(_CS)))]
        return (op, self._random_expr(rng, cols, depth_left - 1))

    def _random_condition(self, rng, cols) -> tuple:
        op = CONDITION_OPS[int(rng.integers(len(CONDITION_OPS)))]
        expr = ('col', cols[int(rng.integers(len(cols)))])
        if rng.random() < 0.5:
            windows = self.dsl_cfg['windows']
            w = int(windows[int(rng.integers(len(windows)))])
            expr = ('roll_zscore', expr, w)
        pool = _ABS_THRESHOLDS if op.startswith('abs') else _GT_LT_THRESHOLDS
        threshold = float(pool[int(rng.integers(len(pool)))])
        return (op, expr, threshold)

    def _fresh(self, rng, family: str, cols) -> Candidate:
        max_depth = int(self.dsl_cfg['max_depth'])
        expr = self._random_expr(rng, cols, max_depth)
        conditions = []
        max_cond = int(self.dsl_cfg['max_conditions'])
        while len(conditions) < max_cond and rng.random() < 0.4:
            conditions.append(self._random_condition(rng, cols))
        return _named(family, expr, tuple(conditions))

    # -- mutation -----------------------------------------------------------

    def _subtree_paths(self, node, path=()) -> list:
        if not isinstance(node, tuple) or node[0] == 'col':
            return [path]
        paths = [path]
        for i, arg in enumerate(node[1:], start=1):
            if isinstance(arg, tuple):
                paths.extend(self._subtree_paths(arg, path + (i,)))
        return paths

    def _get(self, node, path):
        for i in path:
            node = node[i]
        return node

    def _set(self, node, path, new):
        if not path:
            return new
        i = path[0]
        return node[:i] + (self._set(node[i], path[1:], new),) + node[i + 1:]

    def _mutate(self, rng, parent: Candidate, cols) -> Candidate:
        expr = parent.expression
        conditions = list(parent.conditions)
        move = rng.random()
        if move < 0.35:
            # replace a random subtree with a fresh one
            paths = self._subtree_paths(expr)
            path = paths[int(rng.integers(len(paths)))]
            depth_budget = max(2, int(self.dsl_cfg['max_depth']) - len(path))
            expr = self._set(expr, path,
                             self._random_expr(rng, cols, depth_budget))
        elif move < 0.55:
            # wrap the whole expression in a unary
            op = _UNARY[int(rng.integers(len(_UNARY)))]
            expr = (op, expr)
        elif move < 0.75:
            # swap a random leaf column
            paths = [p for p in self._subtree_paths(expr)
                     if self._get(expr, p)[0] == 'col']
            if paths:
                path = paths[int(rng.integers(len(paths)))]
                expr = self._set(expr, path, self._leaf(rng, cols))
        elif move < 0.9 or not conditions:
            # add / replace a gate
            cond = self._random_condition(rng, cols)
            if conditions and len(conditions) >= int(self.dsl_cfg['max_conditions']):
                conditions[int(rng.integers(len(conditions)))] = cond
            else:
                conditions.append(cond)
        else:
            # drop a gate
            conditions.pop(int(rng.integers(len(conditions))))
        return _named(parent.family, expr, tuple(conditions))

    # -- Proposer interface --------------------------------------------------

    def propose(self, n: int, family: str, diagnostics: dict,
                parents: Sequence[Candidate],
                family_columns: Dict[str, list],
                rng: np.random.Generator,
                parent_scores: Optional[dict] = None,
                failures: Optional[list] = None,
                overused: Optional[list] = None) -> list:
        # The random baseline mutates the grammar blindly; scores/failures/
        # overused are hints only an LLM can use.
        cols = list(family_columns.get(family) or [])
        if not cols:
            return []
        family_parents = [p for p in parents if p.family == family]
        out = []
        for _ in range(n):
            if family_parents and rng.random() < self.mutation_prob:
                parent = family_parents[int(rng.integers(len(family_parents)))]
                out.append(self._mutate(rng, parent, cols))
            else:
                out.append(self._fresh(rng, family, cols))
        return out


def _named(family: str, expression: tuple, conditions: tuple,
           rationale: str = '') -> Candidate:
    """Build a candidate named after its content hash."""
    probe = Candidate(name='', family=family, expression=expression,
                      conditions=conditions)
    return Candidate(name=f'{family}_{probe.hash[:8]}', family=family,
                     expression=expression, conditions=conditions,
                     rationale=rationale)


# The fixed LLM system prompt (identical on every call) lives in prompt.md so it
# can be read and edited as prose. Loaded once at import; the file holds ONLY
# prompt text (no markdown title) because it is sent to the model verbatim.
_LLM_SYSTEM = (Path(__file__).parent / 'prompt.md').read_text(
    encoding='utf-8').strip()


class _ApiProposer(Proposer):
    """Shared machinery for API-backed proposers: prompt building, response
    parsing, and token-usage accounting. Untrusted and swappable; an
    unusable response (API error, truncation with no salvageable prefix) is
    RETRIED ONCE, then degrades to an empty batch (the search continues on
    parents).

    Usage accounting: every completed call adds to self.usage. `calls` counts
    ATTEMPTS (a failed call may still bill; a retry is a second attempt);
    token counts come from the provider's usage metadata on successful
    responses.
    """

    provider = ''   # set by subclasses; selects model + price config entries

    def __init__(self, llm_cfg: Optional[dict] = None,
                 dsl_cfg: Optional[dict] = None):
        self.llm_cfg = llm_cfg or get('discovery.llm', {})
        self.dsl_cfg = dsl_cfg or get('discovery.dsl', {})
        self._client = None
        self.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}

    @property
    def model(self) -> str:
        m = self.llm_cfg['model']
        return m[self.provider] if isinstance(m, dict) else str(m)

    def _api_key(self) -> str:
        """API key from the generic .env variable (discovery.llm.key_name).
        Provider-agnostic: switching LLMs never renames the key."""
        return load_api_key(self.llm_cfg.get('key_name', 'LLM_KEY'))

    def usage_snapshot(self) -> dict:
        return dict(self.usage)

    def _prompt(self, n, family, diagnostics, parents, family_columns,
                parent_scores=None, failures=None, overused=None) -> str:
        from research.signals.data import describe_column
        # name -> what it measures, so the LLM reasons about the mechanism
        # rather than guessing from the abbreviation.
        allowed = family_columns.get(family, [])
        all_cols = sorted({c for cols in family_columns.values() for c in cols})
        scores = parent_scores or {}
        # Parents WITH their scores, best first, so the model can lean into
        # what is working and away from what barely survived.
        ranked = sorted(parents, key=lambda p: -(scores.get(p.hash, {})
                                                  .get('reward', 0.0)))
        parents_scored = [{**p.to_dict(), 'score': scores.get(p.hash, {})}
                          for p in ranked][:10]
        payload = {
            'task': (f"Propose {n} new candidates for family '{family}' "
                     f"predicting the forward residual target "
                     f"'{diagnostics.get('target')}'"),
            'notes': ("ev_/mx_/tm_ columns are CROSS-SECTIONALLY CONSTANT "
                      "(events / macro state / calendar): identical for every "
                      "coin at a timestamp, so a direct expression of them "
                      "z-scores to zero. Use them ONLY inside conditions "
                      "(regime gates) or multiplied against a cross-sectional "
                      "column. Classic price-TA columns (mq_/ma_/rs_/bb_/mc_/"
                      "dm_/at_/ch_) are weak as standalone predictors of "
                      "RESIDUAL returns (momentum is already a stripped "
                      "factor) — prefer them as GATES/interactions, e.g. "
                      "'residual reversal only when the trend is exhausted or "
                      "choppy (low efficiency ratio)'. "
                      "LIQUIDITY-CONDITIONING (documented): short-horizon price "
                      "autocorrelation FLIPS with liquidity — illiquid/low-volume "
                      "coins REVERSE (liquidity-provision premium), liquid coins "
                      "trend. A pooled price signal blends the two and cancels; "
                      "gate reversal/momentum on a liquidity or volume column "
                      "(e.g. cs_rel_volume, lq_*, cs_mcap_rank). Order-flow "
                      "columns (of_/ms_ofi/signed) are largely ORTHOGONAL to "
                      "price signals — prized independent breadth. "
                      "current_parents carry their 'score' (reward, IC t-stat, "
                      "alpha half-life in bars) — build on the strong ones, "
                      "abandon directions that scored near zero. "
                      "avoid_these already scored poorly — do not re-propose "
                      "them or trivial variants. overused_building_blocks are "
                      "sub-expressions the search has already tried heavily — "
                      "reach for a DIFFERENT structure/mechanism, not another "
                      "formula built on them."),
            'columns': {c: describe_column(c) for c in allowed},
            'other_columns': {c: describe_column(c) for c in all_cols
                              if c not in allowed},
            'ALLOWED_WINDOWS': self.dsl_cfg['windows'],
            'max_depth': self.dsl_cfg['max_depth'],
            'max_conditions': self.dsl_cfg['max_conditions'],
            'diagnostics': {
                c: diagnostics['features'][c]
                for c in diagnostics.get('top_by_family', {}).get(family, [])
                if c in diagnostics.get('features', {})
            },
            'current_parents': parents_scored,
            'avoid_these': (failures or [])[:6],
            'overused_building_blocks': (overused or [])[:8],
        }
        return json.dumps(payload, separators=(',', ':'))

    def _complete(self, prompt: str) -> str:
        """One API call: prompt -> response text. Must add token counts to
        self.usage from the provider's usage metadata."""
        raise NotImplementedError

    def propose(self, n: int, family: str, diagnostics: dict,
                parents: Sequence[Candidate],
                family_columns: Dict[str, list],
                rng: np.random.Generator,
                parent_scores: Optional[dict] = None,
                failures: Optional[list] = None,
                overused: Optional[list] = None) -> list:
        n = min(n, int(self.llm_cfg['candidates_per_call']))
        prompt = self._prompt(n, family, diagnostics, parents, family_columns,
                              parent_scores=parent_scores, failures=failures,
                              overused=overused)
        items = None
        last_err = None
        for attempt in (1, 2):   # one retry: truncation/parse flakes are
            self.usage['calls'] += 1   # transient; twice in a row is real
            try:
                items = _parse_json_array(self._complete(prompt))
                break
            except RuntimeError:
                raise   # missing/invalid .env key - fail fast, not N empty batches
            except ImportError as e:
                # Missing SDK is a broken environment, not a transient API
                # hiccup: abort instead of degrading to an all-empty search
                # (128 empty batches per roll, zero candidates, zero cost -
                # looks like a run, finds nothing by construction).
                raise SystemExit(
                    f"{self.provider} proposer cannot import its SDK ({e}). "
                    f"Install it ('uv add google-genai' for gemini, "
                    f"'uv add anthropic' for anthropic) or switch "
                    f"discovery.llm.provider.") from e
            except Exception as e:
                last_err = e
        if items is None:
            logging.warning(f"{self.provider} proposer: unusable response "
                            f"twice ({last_err}); skipping this batch "
                            f"(family '{family}').")
            return []

        out = []
        for item in items[:n]:
            try:
                cand = Candidate.from_dict({**item, 'family': family,
                                            'name': ''})
                out.append(_named(family, cand.expression, cand.conditions,
                                  cand.rationale))
            except Exception:
                continue
        return out


def _parse_json_array(text: str) -> list:
    """Parse a JSON array out of an LLM response, tolerating markdown fences,
    surrounding prose, and TRUNCATION (max_tokens cutting the array
    mid-object): the complete prefix of a truncated array is salvaged rather
    than the whole batch being dropped. Raises ValueError when no array can
    be recovered."""
    text = text.strip()
    if text.startswith('```'):
        # strip ```json ... ``` fences
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('[')
        if start < 0:
            raise ValueError("no JSON array in response")
        end = text.rfind(']')
        obj = None
        if end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            obj = _salvage_array_prefix(text[start:])
            if obj is None:
                raise ValueError("no JSON array in response")
            logging.warning(f"truncated JSON array: salvaged {len(obj)} "
                            f"complete candidates")
    if isinstance(obj, dict):
        # some models wrap the array: {"candidates": [...]}
        for v in obj.values():
            if isinstance(v, list):
                return v
        raise ValueError("JSON object contains no array")
    if not isinstance(obj, list):
        raise ValueError(f"expected a JSON array, got {type(obj).__name__}")
    return obj


def _salvage_array_prefix(text: str) -> Optional[list]:
    """Complete top-level objects from the head of a truncated JSON array.

    text starts at '['. Scans brace depth (string- and escape-aware),
    remembers where each depth-1 object closes, and parses the longest
    prefix that forms a valid array. None when not even one object is
    complete."""
    depth = 0
    in_str = False
    esc = False
    last_close = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
            if ch == '}' and depth == 1:
                last_close = i
    if last_close < 0:
        return None
    try:
        return json.loads(text[:last_close + 1] + ']')
    except json.JSONDecodeError:
        return None


def load_api_key(name: str, path: Optional[Path] = None) -> str:
    """Read one API key from the gitignored .env file at the repo root
    (KEY=value lines - the same file the ETL reads its keys from). Raises
    with instructions when the file or the key is missing."""
    from config import load_env_key
    key = load_env_key(name, path)
    if not key:
        env_path = path or Path(__file__).resolve().parents[2] / '.env'
        raise RuntimeError(
            f"no {name} in '{env_path}'. Add a line (file is gitignored):\n"
            f"  {name}=...")
    return key


class AnthropicProposer(_ApiProposer):
    """Claude via the anthropic SDK; key from the generic .env variable."""

    provider = 'anthropic'

    def _complete(self, prompt: str) -> str:
        if self._client is None:
            import anthropic
            # Per-request timeout so a dropped connection raises instead of
            # hanging the whole run forever (retry-once then empty batch).
            self._client = anthropic.Anthropic(
                api_key=self._api_key(),
                timeout=float(self.llm_cfg.get('request_timeout_s', 120)),
                max_retries=0)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=int(self.llm_cfg['max_tokens']),
            system=_LLM_SYSTEM,
            messages=[{'role': 'user', 'content': prompt}],
        )
        u = getattr(resp, 'usage', None)
        if u is not None:
            self.usage['input_tokens'] += int(getattr(u, 'input_tokens', 0) or 0)
            self.usage['output_tokens'] += int(getattr(u, 'output_tokens', 0) or 0)
        return ''.join(b.text for b in resp.content
                       if getattr(b, 'type', '') == 'text')


class GeminiProposer(_ApiProposer):
    """Gemini via the google-genai SDK; key from the generic .env variable."""

    provider = 'gemini'

    def _complete(self, prompt: str) -> str:
        from google.genai import types as genai_types
        if self._client is None:
            from google import genai
            # HttpOptions.timeout is in MILLISECONDS: a dropped connection
            # raises instead of hanging the run (retry-once then empty batch).
            timeout_ms = int(float(self.llm_cfg.get('request_timeout_s', 120))
                             * 1000)
            self._client = genai.Client(
                api_key=self._api_key(),
                http_options=genai_types.HttpOptions(timeout=timeout_ms))
        # response_mime_type forces native JSON mode (no markdown fences, no
        # almost-JSON). A thinking budget of 0 stops 2.5-model thoughts from
        # eating the output-token budget and truncating the array mid-way.
        gen_cfg = dict(
            system_instruction=_LLM_SYSTEM,
            max_output_tokens=int(self.llm_cfg['max_tokens']),
            response_mime_type='application/json',
        )
        thinking_budget = self.llm_cfg.get('gemini_thinking_budget')
        if thinking_budget is not None:
            gen_cfg['thinking_config'] = genai_types.ThinkingConfig(
                thinking_budget=int(thinking_budget))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**gen_cfg),
        )
        um = getattr(resp, 'usage_metadata', None)
        if um is not None:
            self.usage['input_tokens'] += int(
                getattr(um, 'prompt_token_count', 0) or 0)
            self.usage['output_tokens'] += int(
                (getattr(um, 'candidates_token_count', 0) or 0)
                + (getattr(um, 'thoughts_token_count', 0) or 0))
        return resp.text or ''


# Backward-compatible alias (pre-provider-split name)
LLMProposer = AnthropicProposer

_API_PROPOSERS = {'anthropic': AnthropicProposer, 'gemini': GeminiProposer}


def make_proposer(kind: str, **kwargs) -> Proposer:
    """kind: 'random', an explicit provider ('anthropic'/'gemini'), or 'llm'
    which resolves the provider from config (discovery.llm.provider)."""
    if kind == 'random':
        return RandomProposer(**kwargs)
    if kind == 'llm':
        kind = get('discovery.llm.provider', 'anthropic')
    cls = _API_PROPOSERS.get(kind)
    if cls is None:
        raise ValueError(f"unknown proposer kind: {kind!r}")
    return cls(**kwargs)


def estimate_cost_usd(usage: dict, llm_cfg: dict, provider: str):
    """Dollar estimate from token counts and the config price table
    (discovery.llm.price_per_mtok). None when prices are not configured -
    prices change; the config is where the operator pins them."""
    prices = (llm_cfg.get('price_per_mtok') or {}).get(provider) or {}
    p_in, p_out = prices.get('input'), prices.get('output')
    if p_in is None or p_out is None:
        return None
    return (usage['input_tokens'] * float(p_in)
            + usage['output_tokens'] * float(p_out)) / 1e6
