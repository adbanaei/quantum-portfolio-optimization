# End-to-end run of the replacement code before it goes into the notebook.
import numpy as np
import pandas as pd

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
RANDOM_SEED = 42

# the replacement code, as it will go into the notebook
print("Sampling random fully-invested portfolios (paper's 'Random Allocations')...")
rng_frontier = np.random.default_rng(RANDOM_SEED)
N_RANDOM_PORTFOLIOS = 30000
random_weights = rng_frontier.dirichlet(np.ones(len(P)), size=N_RANDOM_PORTFOLIOS)
feasible_points = []
for w in random_weights:
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ Sigma @ w))
    feasible_points.append((vol, ret))

print(f"  Random portfolios sampled: {len(feasible_points)}")
vols_f = [x[0] for x in feasible_points]
rets_f = [x[1] for x in feasible_points]
print(f"  vol range: [{min(vols_f):.4f}, {max(vols_f):.4f}]")
print(f"  ret range: [{min(rets_f):.6f}, {max(rets_f):.6f}]")
print(f"  touches origin: {min(vols_f) < 1e-4}")
assert len(feasible_points) == N_RANDOM_PORTFOLIOS
assert min(vols_f) > 0.005, "should not come anywhere near the origin"
print("\nOK")
