# Does Cairo noise on the raw-dollar Hamiltonian account for Fig 3's scale?
import time, pickle, numpy as np, pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising
from qiskit import transpile

# same raw-dollar Hamiltonian as script 15
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

try:
    from qiskit_ibm_runtime.fake_provider import FakeCairoV2
    cairo_nm = NoiseModel.from_backend(FakeCairoV2())
    print("Using FakeCairoV2 (matches notebook's actual code path)")
except Exception as e:
    print(f"FakeCairoV2 unavailable ({e}) -- aborting rather than quietly "
          f"substituting another noise model")
    raise

noisy_sampler = AerSampler(seed=42)
noisy_sampler.options.default_shots = 2000
noisy_sampler.options.backend_options = {
    "noise_model": cairo_nm, "method": "statevector",
    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0,
}

ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                   entanglement="linear", reps=3)
ansatz_d = transpile(ansatz.decompose(reps=10), backend=noisy_sampler._backend, optimization_level=1)
print(f"num_parameters: {ansatz_d.num_parameters}")

t0 = time.time()
history = []
def cb(count, params, value, meta):
    history.append((count, float(value)))
rng = np.random.default_rng(12345)
init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
optimizer = COBYLA(maxiter=200)
vqe = SamplingVQE(sampler=noisy_sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
result = vqe.compute_minimum_eigenvalue(ising_op_raw)
elapsed = time.time() - t0
print(f"COBYLA/noisy-Cairo/raw-dollar: {len(history)} evals, {elapsed:.1f}s")
print(f"final energy: {float(np.real(result.eigenvalue)):.4e}")
print(f"energy range visited: {min(v for _,v in history):.4e} to {max(v for _,v in history):.4e}")
print(f"(for comparison) NOISELESS raw-dollar COBYLA final energy was: -1.9706e+08")
print(f"(for comparison) paper's Fig 2 (noiseless) scale: ~1e8; Fig 3 (noisy) scale: ~1e9")

with open("noisy_raw_dollar_result.pkl", "wb") as f:
    pickle.dump({"history": history, "elapsed": elapsed}, f)
