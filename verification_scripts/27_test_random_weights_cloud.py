# Random fully-invested weights on the simplex, against exhaustive enumeration.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
BUDGET = 2000
N = len(mu)

def build_binary_encoding(P, B):
    Nn = len(P)
    n_max = [int(B / P[i]) for i in range(Nn)]
    d_list = [int(np.floor(np.log2(nm))) if nm > 0 else 0 for nm in n_max]
    C = np.zeros((Nn, sum(d + 1 for d in d_list))); col = 0
    for i in range(Nn):
        for j in range(d_list[i] + 1):
            C[i, col] = 2 ** j; col += 1
    return C
C = build_binary_encoding(P, BUDGET)
dim_b = C.shape[1]

def portfolio_return_risk_original(b_vec, C, mu, Sigma, P, BUDGET):
    n_units = C @ np.array(b_vec, dtype=float)
    x = n_units * P / BUDGET
    ret = float(mu @ x)
    vol = float(np.sqrt(x @ Sigma @ x))
    return ret, vol

import itertools
print("exhaustive enumeration, 2^15, no budget filter...")
current_cloud = []
for bits in itertools.product([0, 1], repeat=dim_b):
    ret, vol = portfolio_return_risk_original(bits, C, mu, Sigma, P, BUDGET)
    current_cloud.append((vol, ret))
current_cloud = np.array(current_cloud)
print(f"  {len(current_cloud)} points, vol range [{current_cloud[:,0].min():.4f}, {current_cloud[:,0].max():.4f}], "
      f"ret range [{current_cloud[:,1].min():.4f}, {current_cloud[:,1].max():.4f}]")
print(f"  touches origin (min vol < 0.001)? {current_cloud[:,0].min() < 0.001}")

print("\nrandom fully-invested weights on the simplex...")
rng = np.random.default_rng(0)
n_samples = 30000
weights = rng.dirichlet(np.ones(N), size=n_samples)  # rows sum to 1, long-only
hyp_ret = weights @ mu
hyp_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, Sigma, weights))
print(f"  {n_samples} points, vol range [{hyp_vol.min():.4f}, {hyp_vol.max():.4f}], "
      f"ret range [{hyp_ret.min():.4f}, {hyp_ret.max():.4f}]")
print(f"  touches origin (min vol < 0.001)? {hyp_vol.min() < 0.001}")

b_opt_classical = np.array([0,1,0,0,0,0,0,0,1,0,0,1,1,1,0])
r_classical, v_classical = portfolio_return_risk_original(b_opt_classical, C, mu, Sigma, P, BUDGET)
print(f"\nclassical: vol={v_classical:.4f}, ret={r_classical:.6f}")
print(f"  Is classical INSIDE the hypothesis cloud's range? "
      f"vol in [{hyp_vol.min():.4f},{hyp_vol.max():.4f}]: {hyp_vol.min()<=v_classical<=hyp_vol.max()}, "
      f"ret in [{hyp_ret.min():.6f},{hyp_ret.max():.6f}]: {hyp_ret.min()<=r_classical<=hyp_ret.max()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(current_cloud[:,0], current_cloud[:,1], s=2, alpha=0.15, color="lightblue")
axes[0].scatter([v_classical], [r_classical], s=150, color="black", marker="s", zorder=5)
axes[0].set_title("CURRENT: exhaustive 2^15 enumeration\n(touches origin, narrow band)")
axes[0].set_xlabel("Volatility"); axes[0].set_ylabel("Return")

axes[1].scatter(hyp_vol, hyp_ret, s=2, alpha=0.15, color="teal")
axes[1].scatter([v_classical], [r_classical], s=150, color="black", marker="s", zorder=5)
axes[1].set_title("HYPOTHESIS: random weights on simplex\n(fully invested, should NOT touch origin)")
axes[1].set_xlabel("Volatility"); axes[1].set_ylabel("Return")
fig.tight_layout()
fig.savefig("cloud_hypothesis_comparison.png", dpi=130)
print("\nsaved cloud_hypothesis_comparison.png")
