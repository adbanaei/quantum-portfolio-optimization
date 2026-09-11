# Run the actual VQE against the raw-dollar Hamiltonian, not just its ground state.
import time, pickle, numpy as np, pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NFT
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
mu_pp_raw = C.T @ (P * mu)
Sigma_pp_raw = C.T @ (P[:, None] * Sigma * P[None, :]) @ C
P_pp_raw = C.T @ P

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

qp_raw = build_qubo(mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA)
ising_op_raw, offset_raw = to_ising(QuadraticProgramToQubo().convert(qp_raw))
ising_op_raw = ising_op_raw.simplify(atol=1e-6)
print(f"offset={offset_raw:.4e}")

sampler = AerSampler(seed=42)
sampler.options.default_shots = 2000
sampler.options.backend_options = {"method": "matrix_product_state",
                                    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0}

results = {}
for OptClass, name, maxiter in [(COBYLA, "COBYLA", 200), (SPSA, "SPSA", 200), (NFT, "NFT", 200)]:
    ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                       entanglement="linear", reps=3)
    ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
    history = []
    def cb(count, params, value, meta, _h=history):
        _h.append((count, float(value)))
    rng = np.random.default_rng(12345)
    init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
    optimizer = OptClass(maxiter=maxiter)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
    t0 = time.time()
    result = vqe.compute_minimum_eigenvalue(ising_op_raw)
    elapsed = time.time() - t0
    results[name] = history
    print(f"{name}: {len(history)} evals, {elapsed:.1f}s, final energy={float(np.real(result.eigenvalue)):.4e}, "
          f"min energy reached={min(v for _,v in history):.4e}")

with open("raw_dollar_vqe_results.pkl", "wb") as f:
    pickle.dump({"results": results, "offset": offset_raw}, f)
