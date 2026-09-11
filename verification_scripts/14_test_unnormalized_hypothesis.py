# Does dropping the budget normalisation reproduce the paper's energy scale?
import numpy as np, pandas as pd, itertools
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

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
BUDGET, RISK_Q, LAMBDA = 2000, 0.5, 10.0

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

def build_qubo(mu_pp, Sigma_pp, P_pp, q, lam):
    n = len(mu_pp)
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"b{i}")
    linear = {f"b{i}": -mu_pp[i] + q*Sigma_pp[i,i] + lam*(P_pp[i]**2 - 2*P_pp[i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            v = 2*q*Sigma_pp[i,j] + 2*lam*P_pp[i]*P_pp[j]
            if v != 0: quadratic[(f"b{i}", f"b{j}")] = v
    qp.minimize(linear=linear, quadratic=quadratic, constant=lam)
    return qp

print("--- baseline: budget-normalized, as the notebook builds it, lambda=10 ---")
P_prime = P / BUDGET
mu_pp = C.T @ (P_prime * mu)
Sigma_pp = C.T @ (P_prime[:,None]*Sigma*P_prime[None,:]) @ C
P_pp = C.T @ P_prime
qp = build_qubo(mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
ising_op, offset = to_ising(QuadraticProgramToQubo().convert(qp))
coeffs = np.abs(np.array([complex(c).real for c in ising_op.coeffs]))
print(f"  offset={offset:.2f}  max|coeff|={coeffs.max():.2f}  sum|coeff|={coeffs.sum():.2f}")
print(f"  -> matches notebook's printed offset=20.843858: {abs(offset-20.843858)<1e-3}")

print("\n--- raw-dollar P in the C-transform, everything else identical ---")
mu_pp_raw = C.T @ (P * mu)
Sigma_pp_raw = C.T @ (P[:,None]*Sigma*P[None,:]) @ C
P_pp_raw = C.T @ P
qp_raw = build_qubo(mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA)
ising_op_raw, offset_raw = to_ising(QuadraticProgramToQubo().convert(qp_raw))
coeffs_raw = np.abs(np.array([complex(c).real for c in ising_op_raw.coeffs]))
print(f"  offset={offset_raw:.2f}  max|coeff|={coeffs_raw.max():.2f}  sum|coeff|={coeffs_raw.sum():.2f}")
print(f"  scale ratio vs baseline: offset {offset_raw/offset:.1f}x, max_coeff {coeffs_raw.max()/coeffs.max():.1f}x")

def evaluate_portfolio(b, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b, dtype=float)
    return mu_pp @ b - q*(b@Sigma_pp@b) - lam*(P_pp@b-1)**2

best_val = -np.inf
for bits in itertools.product([0,1], repeat=dim_b):
    v = evaluate_portfolio(bits, mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA)
    if v > best_val: best_val = v
ising_gs_raw = -best_val - offset_raw
print(f"  classical-optimum Ising energy under this hypothesis: {ising_gs_raw:.2e}")
print(f"  paper's Fig 2 E_min reaches approximately: -1e8")
print(f"  paper's Fig 3 E_min reaches approximately: -1e9")

# variant: raw dollars AND the constraint written in dollars, (P''b - B)^2
print("\n--- raw dollars, constraint (P''b - BUDGET)^2 ---")
def build_qubo_dollar_constraint(mu_pp, Sigma_pp, P_pp, q, lam, budget):
    n = len(mu_pp)
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"b{i}")
    linear = {f"b{i}": -mu_pp[i] + q*Sigma_pp[i,i] + lam*(P_pp[i]**2 - 2*budget*P_pp[i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            v = 2*q*Sigma_pp[i,j] + 2*lam*P_pp[i]*P_pp[j]
            if v != 0: quadratic[(f"b{i}", f"b{j}")] = v
    qp.minimize(linear=linear, quadratic=quadratic, constant=lam*budget**2)
    return qp
qp_raw2 = build_qubo_dollar_constraint(mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA, BUDGET)
ising_op_raw2, offset_raw2 = to_ising(QuadraticProgramToQubo().convert(qp_raw2))
coeffs_raw2 = np.abs(np.array([complex(c).real for c in ising_op_raw2.coeffs]))
print(f"  offset={offset_raw2:.4e}  max|coeff|={coeffs_raw2.max():.4e}")
