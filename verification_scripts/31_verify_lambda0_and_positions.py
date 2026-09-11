# Is the plotted lambda=0 point the true optimum, and where do the markers sit in the cloud?
import numpy as np, pandas as pd, itertools

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
BUDGET, RISK_Q = 2000, 0.5

def build_binary_encoding(P, B):
    N = len(P)
    n_max = [int(B / P[i]) for i in range(N)]
    d_list = [int(np.floor(np.log2(nm))) if nm > 0 else 0 for nm in n_max]
    C = np.zeros((N, sum(d + 1 for d in d_list))); col = 0
    for i in range(N):
        for j in range(d_list[i] + 1):
            C[i, col] = 2 ** j; col += 1
    return C
C = build_binary_encoding(P, BUDGET)
dim_b = C.shape[1]
P_prime = P / BUDGET
mu_pp = C.T @ (P_prime*mu); Sigma_pp = C.T @ (P_prime[:,None]*Sigma*P_prime[None,:]) @ C; P_pp = C.T @ P_prime

def evaluate_portfolio(b, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b, dtype=float)
    return mu_pp @ b - q*(b@Sigma_pp@b) - lam*(P_pp@b-1)**2

def portfolio_return_risk_original(b, C, mu, Sigma, P, BUDGET):
    n_units = C @ np.array(b, dtype=float)
    x = n_units * P / BUDGET
    ret = float(mu @ x)
    vol = float(np.sqrt(x @ Sigma @ x))
    return ret, vol

print("--- brute-force lambda=0 optimum: max mu''.b - q b'Sigma''b, no penalty ---")
best_val, best_b = -np.inf, None
for bits in itertools.product([0, 1], repeat=dim_b):
    val = evaluate_portfolio(bits, mu_pp, Sigma_pp, P_pp, RISK_Q, 0.0)
    if val > best_val:
        best_val, best_b = val, np.array(bits)
ret0, vol0 = portfolio_return_risk_original(best_b, C, mu, Sigma, P, BUDGET)
ppb0 = P_pp @ best_b
print(f"TRUE brute-force optimal b* for lambda=0: {best_b}")
print(f"  L(b*) = {best_val:.6f}")
print(f"  return={ret0:.6f}  volatility={vol0:.6f}  P''b={ppb0:.3f}")
print(f"\nPlotted lambda=0 point: return=0.006, volatility=0.075, P''b=3.425")
print(f"Match: {np.isclose(ret0, 0.006, atol=0.0005) and np.isclose(vol0, 0.075, atol=0.005)}")

print("\n--- each marker relative to the fully-invested cloud's own extent ---")
rng = np.random.default_rng(42)
n_samples = 30000
weights = rng.dirichlet(np.ones(len(P)), size=n_samples)
cloud_ret = weights @ mu
cloud_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, Sigma, weights))
print(f"Cloud extent: vol [{cloud_vol.min():.4f}, {cloud_vol.max():.4f}]  ret [{cloud_ret.min():.6f}, {cloud_ret.max():.6f}]")

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([cloud_vol/cloud_vol.max(), cloud_ret/cloud_ret.max()]))

def describe_position(label, vol, ret):
    in_range = cloud_vol.min() <= vol <= cloud_vol.max()
    vol_pct = 100*(vol - cloud_vol.min())/(cloud_vol.max()-cloud_vol.min())
    print(f"  {label:10s}: vol={vol:.4f} ret={ret:.6f}  "
          f"{'INSIDE' if in_range else 'OUTSIDE'} cloud vol-range  (vol at {vol_pct:.0f}th pct of cloud's own vol span)")

points = {
    "lambda=0": (0.075, 0.006),
    "lambda=1": (0.0165, 0.0013),
    "lambda=10": (0.0165, 0.0012),
    "classical": (0.0285, 0.0021),
}
for label, (v, r) in points.items():
    describe_position(label, v, r)
