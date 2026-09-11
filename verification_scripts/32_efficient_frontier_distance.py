# Distance from the efficient boundary at each point's own volatility.
import numpy as np, pandas as pd

TICKERS = ["AAPL", "IBM", "NFLX", "TSLA"]
price_series = {}
for t in TICKERS:
    df = pd.read_csv(f"./price_data/{t}.csv", index_col=0, parse_dates=True)
    s = df["Corrected_Adj_Close"].copy(); s.name = t
    price_series[t] = s
prices_df = pd.concat(price_series.values(), axis=1)
prices_df = prices_df.loc[(prices_df.index >= "2011-12-23") & (prices_df.index < "2022-10-21")].sort_index().dropna()
P = prices_df.iloc[-1].values.astype(float)
returns_df = prices_df.pct_change().dropna()
mu = returns_df.mean().values
Sigma = returns_df.cov().values
N = len(mu)

rng = np.random.default_rng(42)
n_samples = 200000  # dense enough for a stable local max
weights = rng.dirichlet(np.ones(N), size=n_samples)
cloud_ret = weights @ mu
cloud_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, Sigma, weights))

def max_return_at_volatility(target_vol, tol=0.0008):
    """best cloud return within a narrow volatility band around target_vol"""
    mask = np.abs(cloud_vol - target_vol) < tol
    if mask.sum() < 5:
        return None
    return cloud_ret[mask].max()

points = {
    "lambda=1":   (0.0165, 0.0013),
    "lambda=10":  (0.0165, 0.0012),
    "lambda=100": (0.0160, 0.0011),
    "classical":  (0.0285, 0.0021),
}
print(f"{'':12s}{'vol':>8s}{'ret':>10s}{'max_ret_at_this_vol':>22s}{'% of local frontier':>22s}")
for label, (v, r) in points.items():
    m = max_return_at_volatility(v)
    pct = 100*r/m if m else float('nan')
    print(f"{label:12s}{v:>8.4f}{r:>10.5f}{m:>22.5f}{pct:>21.1f}%")

# the classical optimum is the one point we know is exact, so use it as a check
b_opt_classical_vol, b_opt_classical_ret = 0.028465, 0.002093
m_at_classical = max_return_at_volatility(b_opt_classical_vol)
print(f"\nclassical optimum vs local frontier max: "
      f"{100*b_opt_classical_ret/m_at_classical:.1f}%  (should be near 100%)")
