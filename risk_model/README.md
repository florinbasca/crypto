# Risk Model Pipeline

Run scripts in this order:

```bash
python risk_model/pca_factors.py      # PCA factor loadings
python risk_model/factor_returns.py   # Risk factor time series
python risk_model/residual_returns.py # Factor-adjusted residuals
python risk_model/features.py         # All features
```

## Dependencies

| Script | Reads | Writes |
|--------|-------|--------|
| `pca_factors.py` | `prices` | `pca_loadings`, `pca_variance` |
| `factor_returns.py` | `prices` | `risk_factors` |
| `residual_returns.py` | `prices`, `risk_factors` | `residual_returns`, `factor_loadings` |
| `features.py` | `prices`, `residual_returns`, `factor_loadings`, `funding_rates`, `futures_metrics` | `features_{freq}` |

## Modules

- `features_futures.py` - Called by `features.py` to add futures-derived features (funding rates, OI, L/S ratios)
