# Where does the ~1e8 energy scale of the paper's Figs 2 and 3 come from?
import itertools

import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

TICKERS = ["AAPL", "IBM", "NFLX", "TSLA"]
START_DATE, END_DATE = "2011-12-23", "2022-10-21"
BUDGET, RISK_Q, LAMBDA = 2000.0, 0.5, 10.0
PAPER_E_MIN = -1.1e8  # off the 1e8 axis multiplier of Figs. 2 and 3

series = {}
for t in TICKERS:
    df = pd.read_csv(f"./price_data/{t}.csv", index_col=0, parse_dates=True)
    s = df["Corrected_Adj_Close"].copy()
    s.name = t
    series[t] = s
prices = pd.concat(series.values(), axis=1)
prices = prices.loc[(prices.index >= START_DATE) & (prices.index < END_DATE)].sort_index().dropna()

P = prices.iloc[-1].values.astype(float)
returns = prices.pct_change().dropna()
mu, Sigma = returns.mean().values, returns.cov().values

n_max = [int(BUDGET / p) for p in P]
d_list = [int(np.floor(np.log2(nm))) for nm in n_max]
C = np.zeros((len(P), sum(d + 1 for d in d_list)))
col = 0
for i in range(len(P)):
    for j in range(d_list[i] + 1):
        C[i, col] = 2 ** j
        col += 1


def ground_energy(mu_v, Sigma_v, P_v, target):
    """Ising ground energy of  max_b  mu.n - q n'Sig n - lam (P'n - target)^2,  n = C b."""
    mu_b, Sig_b, P_b = C.T @ mu_v, C.T @ Sigma_v @ C, C.T @ P_v
    nb = len(mu_b)
    qp = QuadraticProgram()
    for i in range(nb):
        qp.binary_var(name=f"b{i}")
    qp.minimize(
        linear={f"b{i}": -mu_b[i] + RISK_Q * Sig_b[i, i]
                         + LAMBDA * (P_b[i] ** 2 - 2 * P_b[i] * target) for i in range(nb)},
        quadratic={(f"b{i}", f"b{j}"): 2 * RISK_Q * Sig_b[i, j] + 2 * LAMBDA * P_b[i] * P_b[j]
                   for i in range(nb) for j in range(i + 1, nb)},
        constant=LAMBDA * target ** 2,
    )
    op, offset = to_ising(QuadraticProgramToQubo().convert(qp))
    allb = np.array(list(itertools.product([0, 1], repeat=nb)), dtype=float)
    L = (allb @ mu_b) - RISK_Q * np.einsum("ij,jk,ik->i", allb, Sig_b, allb) \
        - LAMBDA * (allb @ P_b - target) ** 2
    return -L.max() - offset, offset, np.abs(op.coeffs).max()


Pn = P / BUDGET
E_norm, off_norm, cmax_norm = ground_energy(Pn * mu, (Pn[:, None] * Sigma) * Pn[None, :], Pn, 1.0)
E_raw, off_raw, cmax_raw = ground_energy(mu, Sigma, P, BUDGET)

print(f"{'formulation':<46}{'E_ground':>15}{'offset':>15}{'max|coeff|':>14}")
print("-" * 90)
print(f"{'normalised (paper Eq. 7, this notebook)':<46}{E_norm:>15.3f}{off_norm:>15.3f}{cmax_norm:>14.3f}")
print(f"{'raw dollars, constraint P^T n = B':<46}{E_raw:>15.4e}{off_raw:>15.4e}{cmax_raw:>14.4e}")

print(f"\nratio of ground energies : {E_raw / E_norm:,.0f}")
print(f"B^2                      : {BUDGET ** 2:,.0f}")
print(f"predicted == observed    : {abs(E_raw / E_norm - BUDGET ** 2) / BUDGET ** 2 < 1e-4}")

print(f"\ngap to the paper's axis (E_min ~ {PAPER_E_MIN:.1e}):")
print(f"  normalised formulation : {abs(PAPER_E_MIN / E_norm):,.0f}x  -- unexplainable by any data difference")
print(f"  raw-dollar formulation : {abs(PAPER_E_MIN / E_raw):.2f}x  -- same order of magnitude")
