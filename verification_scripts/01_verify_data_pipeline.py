# Rebuild the pipeline from the CSVs and check it against the notebook's own numbers.
import numpy as np
import pandas as pd

TICKERS = ["AAPL", "IBM", "NFLX", "TSLA"]
START_DATE = "2011-12-23"
END_DATE = "2022-10-21"
BUDGET = 2000
PRICE_DATA_DIR = "./price_data"

price_series = {}
for t in TICKERS:
    path = f"{PRICE_DATA_DIR}/{t}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    assert "Corrected_Adj_Close" in df.columns
    s = df["Corrected_Adj_Close"].copy()
    s.name = t
    price_series[t] = s

prices_df = pd.concat(price_series.values(), axis=1)
prices_df = prices_df.loc[(prices_df.index >= START_DATE) & (prices_df.index < END_DATE)]
prices_df = prices_df.sort_index().dropna()

print(f"Data shape: {prices_df.shape}")
print(f"Last date used (should be < {END_DATE}): {prices_df.index[-1].date()}")
print()

P = prices_df.iloc[-1].values.astype(float)
N = len(P)
returns_df = prices_df.pct_change().dropna()
mu = returns_df.mean().values
Sigma = returns_df.cov().values

print("Current prices P:")
for t, p in zip(TICKERS, P):
    print(f"  {t:6s}: {p:.6f}")
print("\nExpected returns mu:")
for t, m in zip(TICKERS, mu):
    print(f"  {t:6s}: {m:.6f}")
print("\nCovariance matrix Sigma:")
print(np.round(Sigma, 8))

expected_P = {"AAPL": 142.92, "IBM": 125.44, "NFLX": 268.16, "TSLA": 207.28}
print("\n--- vs notebook's printed output ---")
for t, p in zip(TICKERS, P):
    diff = abs(p - expected_P[t])
    print(f"  {t}: mine={p:.2f}  notebook printed={expected_P[t]}  diff={diff:.4f}  {'OK' if diff < 0.01 else 'MISMATCH'}")

# binary encoding, Eq 8-11
def build_binary_encoding(P, B):
    N = len(P)
    n_max = [int(B / P[i]) for i in range(N)]
    d_list = [int(np.floor(np.log2(nm))) if nm > 0 else 0 for nm in n_max]
    dim_b = sum(d + 1 for d in d_list)
    return n_max, d_list, dim_b

n_max, d_list, dim_b = build_binary_encoding(P, BUDGET)
print(f"\nn_max: {n_max}")
print(f"d_list: {d_list}")
print(f"dim_b (total qubits): {dim_b}  <-- paper reports 12")

# how far each price could move before d_i changes (i.e. before dim_b does)
print("\n--- price range giving the same d_i ---")
for i, t in enumerate(TICKERS):
    d = d_list[i]
    lo_n, hi_n = 2**d, 2**(d+1) - 1          # n_max range giving this d_i
    lo_p, hi_p = BUDGET / (hi_n + 1), BUDGET / lo_n   # n_max = int(B/P)
    print(f"  {t:6s}: current P={P[i]:.2f}, n_max={n_max[i]} -> d_i={d} stays fixed "
          f"for P in ({lo_p:.2f}, {hi_p:.2f}]")
