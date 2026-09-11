# The to_ising() checks again, on the real 15-variable problem.
import numpy as np
import pandas as pd
import itertools
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising
from qiskit.quantum_info import Statevector

TICKERS = ["AAPL", "IBM", "NFLX", "TSLA"]
START_DATE, END_DATE, BUDGET, RISK_Q, LAMBDA = "2011-12-23", "2022-10-21", 2000, 0.5, 10.0
PRICE_DATA_DIR = "./price_data"

price_series = {}
for t in TICKERS:
    df = pd.read_csv(f"{PRICE_DATA_DIR}/{t}.csv", index_col=0, parse_dates=True)
    s = df["Corrected_Adj_Close"].copy(); s.name = t
    price_series[t] = s
prices_df = pd.concat(price_series.values(), axis=1)
prices_df = prices_df.loc[(prices_df.index >= START_DATE) & (prices_df.index < END_DATE)].sort_index().dropna()

P = prices_df.iloc[-1].values.astype(float)
returns_df = prices_df.pct_change().dropna()
mu = returns_df.mean().values
Sigma = returns_df.cov().values

P_prime = P / BUDGET
mu_prime = P_prime * mu
Sigma_prime = (P_prime[:, None] * Sigma) * P_prime[None, :]

def build_binary_encoding(P, B):
    N = len(P)
    n_max = [int(B / P[i]) for i in range(N)]
    d_list = [int(np.floor(np.log2(nm))) if nm > 0 else 0 for nm in n_max]
    dim_b = sum(d + 1 for d in d_list)
    C = np.zeros((N, dim_b)); col = 0
    for i in range(N):
        for j in range(d_list[i] + 1):
            C[i, col] = 2 ** j; col += 1
    return C, n_max, d_list

C, n_max, d_list = build_binary_encoding(P, BUDGET)
dim_b = C.shape[1]
mu_pp = C.T @ mu_prime
Sigma_pp = C.T @ Sigma_prime @ C
P_pp = C.T @ P_prime
print(f"dim_b = {dim_b}  (expect 15, matching notebook)")

def build_qubo(mu_pp, Sigma_pp, P_pp, q, lam):
    n = len(mu_pp)
    qp = QuadraticProgram(name="Portfolio_QUBO")
    for i in range(n):
        qp.binary_var(name=f"b{i}")
    linear = {}
    for i in range(n):
        lin_i = -mu_pp[i] + q * Sigma_pp[i, i] + lam * (P_pp[i] ** 2 - 2 * P_pp[i])
        linear[f"b{i}"] = lin_i
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            q_ij = 2 * q * Sigma_pp[i, j] + 2 * lam * P_pp[i] * P_pp[j]
            if q_ij != 0.0:
                quadratic[(f"b{i}", f"b{j}")] = q_ij
    qp.minimize(linear=linear, quadratic=quadratic, constant=lam)
    return qp

def evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b_vec, dtype=float)
    return mu_pp @ b - q * (b @ Sigma_pp @ b) - lam * (P_pp @ b - 1) ** 2

qp = build_qubo(mu_pp, Sigma_pp, P_pp, q=RISK_Q, lam=LAMBDA)
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)
ising_op, offset = to_ising(qubo)
ising_op = ising_op.simplify(atol=1e-12)
print(f"offset = {offset:.6f}  (notebook printed: 20.843858)")
print(f"# Pauli terms = {len(ising_op)}  (notebook printed: 120)")
print("First 5 terms (should match notebook's printed values):")
for term in list(ising_op)[:5]:
    print(" ", term)

print("\n--- qp.objective(b) vs <b|H|b> + offset ---")
rng = np.random.default_rng(0)
test_bs = [np.zeros(dim_b), np.ones(dim_b)]
for _ in range(5):
    test_bs.append(rng.integers(0, 2, size=dim_b))

for b in test_bs:
    obj = qp.objective.evaluate(np.array(b, dtype=float))
    label = "".join(str(int(x)) for x in reversed(b))
    sv = Statevector.from_label(label)
    h_expect = sv.expectation_value(ising_op).real
    reconstructed = h_expect + offset
    print(f"  qp.obj={obj:12.6f}   <H>+offset={reconstructed:12.6f}   "
          f"match={abs(obj-reconstructed) < 1e-6}")

print("\n--- brute-force classical optimum (independent re-solve) ---")
best_val, best_b = -np.inf, None
for bits in itertools.product([0, 1], repeat=dim_b):
    val = evaluate_portfolio(bits, mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
    if val > best_val:
        best_val, best_b = val, np.array(bits)
print(f"Classical optimum L* = {best_val:.8f}  (notebook printed: 0.00162574)")
print(f"Classical optimum b* = {best_b}")
print(f"                       (notebook printed: [0 1 0 0 0 0 0 0 1 0 0 1 1 1 0])")

label = "".join(str(int(x)) for x in reversed(best_b))
sv = Statevector.from_label(label)
h_at_opt = sv.expectation_value(ising_op).real
print(f"\n<b*|H|b*> (raw, no offset) = {h_at_opt:.6f}")
print(f"<b*|H|b*> + offset = {h_at_opt + offset:.6f}  should equal qp.objective(b*) = -L* = {-best_val:.6f}")
print(f"notebook's reference line, -val_opt_classical - offset = {-best_val - offset:.6f}")
print(f"  equals <b*|H|b*>: {abs((-best_val - offset) - h_at_opt) < 1e-6}")

print("\n--- coefficient magnitude vs lambda ---")
for lam in [0, 1, 10, 100, 1000, 10000]:
    qp_lam = build_qubo(mu_pp, Sigma_pp, P_pp, q=RISK_Q, lam=lam)
    lin_dict = qp_lam.objective.linear.to_dict()
    quad_dict = qp_lam.objective.quadratic.to_dict()
    max_lin = max(abs(v) for v in lin_dict.values()) if lin_dict else 0
    max_quad = max(abs(v) for v in quad_dict.values()) if quad_dict else 0
    max_P = np.max(P_pp)
    predicted_quad_from_lambda = 2 * lam * max_P * max_P
    print(f"  lambda={lam:<6}  max|linear coef|={max_lin:>14.4f}  max|quadratic coef|={max_quad:>14.4f}  "
          f"predicted ~2*lam*max(P'')^2={predicted_quad_from_lambda:>14.4f}")
