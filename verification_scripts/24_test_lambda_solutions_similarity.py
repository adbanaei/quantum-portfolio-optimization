# Does VQE land on different bitstrings for different lambda, or much the same one?
import time, numpy as np, pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising
from qiskit import transpile

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

def build_qubo(mu_pp, Sigma_pp, P_pp, q, lam):
    n = len(mu_pp)
    qp = QuadraticProgram()
    for i in range(n): qp.binary_var(name=f"b{i}")
    linear = {f"b{i}": -mu_pp[i] + q*Sigma_pp[i,i] + lam*(P_pp[i]**2 - 2*P_pp[i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            v = 2*q*Sigma_pp[i,j] + 2*lam*P_pp[i]*P_pp[j]
            if v != 0: quadratic[(f"b{i}", f"b{j}")] = v
    qp.minimize(linear=linear, quadratic=quadratic, constant=lam)
    return qp

def evaluate_portfolio(b, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b, dtype=float)
    return mu_pp @ b - q*(b@Sigma_pp@b) - lam*(P_pp@b-1)**2

sampler = AerSampler(seed=42, options={"backend_options": {"method": "matrix_product_state"}})
sampler.options.default_shots = 2000

# lambda=10's VQE solution, from the notebook's own run
b_lambda10 = np.array([0,1,1,0,1,0,1,0,0,0,0,1,0,0,0])
b_classical = np.array([0,1,0,0,0,0,0,0,1,0,0,1,1,1,0])

decoded = {"lambda=10 (from notebook transcript)": b_lambda10}

for lam in [1, 100]:
    qp = build_qubo(mu_pp, Sigma_pp, P_pp, RISK_Q, lam)
    ising_op, offset = to_ising(QuadraticProgramToQubo().convert(qp))
    ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry","rz"], entanglement_blocks="cz",
                       entanglement="linear", reps=3)
    ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
    rng = np.random.default_rng(12345)  # same seed as the earlier lambda=10 runs
    init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
    optimizer = COBYLA(maxiter=250)
    t0=time.time()
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init)
    result = vqe.compute_minimum_eigenvalue(ising_op)
    bound = ansatz_d.assign_parameters(list(result.optimal_parameters.values())
                                        if isinstance(result.optimal_parameters, dict)
                                        else list(result.optimal_parameters))
    bound.measure_all()
    transpiled = transpile(bound, optimization_level=1)
    job = sampler.run([transpiled], shots=2000)
    counts = job.result()[0].data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=int)
    decoded[f"lambda={lam}"] = b_vec
    print(f"lambda={lam}: {time.time()-t0:.1f}s, b_vec={b_vec}, "
          f"L(b)={evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, RISK_Q, lam):.6f}")

print("\n--- pairwise Hamming distances between decoded solutions, out of 15 ---")
names = list(decoded.keys()) + ["classical b*"]
vecs = list(decoded.values()) + [b_classical]
print(f"{'':38s}" + "".join(f"{n[:14]:>16s}" for n in names))
for i, ni in enumerate(names):
    row = f"{ni:38s}"
    for j, nj in enumerate(names):
        h = int(np.sum(vecs[i] != vecs[j]))
        row += f"{h:>16d}"
    print(row)
