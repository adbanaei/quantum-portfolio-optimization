# Rerun both Hamiltonians with the noise model attached the way that works.
import time, pickle, numpy as np, pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
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

mu_pp_raw = C.T @ (P * mu); Sigma_pp_raw = C.T @ (P[:,None]*Sigma*P[None,:]) @ C; P_pp_raw = C.T @ P
ising_raw, offset_raw = to_ising(QuadraticProgramToQubo().convert(
    build_qubo(mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA)))
ising_raw = ising_raw.simplify(atol=1e-6)

P_prime = P / BUDGET
mu_pp = C.T @ (P_prime * mu); Sigma_pp = C.T @ (P_prime[:,None]*Sigma*P_prime[None,:]) @ C; P_pp = C.T @ P_prime
ising_real, offset_real = to_ising(QuadraticProgramToQubo().convert(
    build_qubo(mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)))
ising_real = ising_real.simplify(atol=1e-6)

cairo_nm = NoiseModel.from_backend(FakeCairoV2())

CORRECT_NOISY_OPTIONS = {"backend_options": {"noise_model": cairo_nm, "method": "statevector",
                                              "max_parallel_threads": 0, "max_parallel_experiments": 0,
                                              "max_parallel_shots": 0}}

def run_vqe(ising_op, sampler, maxiter=200, seed=12345, reps=3):
    ansatz = TwoLocal(num_qubits=ising_op.num_qubits, rotation_blocks=["ry", "rz"],
                       entanglement_blocks="cz", entanglement="linear", reps=reps)
    ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
    history = []
    def cb(count, params, value, meta):
        history.append((count, float(value)))
    rng = np.random.default_rng(seed)
    init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
    optimizer = COBYLA(maxiter=maxiter)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
    t0 = time.time()
    result = vqe.compute_minimum_eigenvalue(ising_op)
    return history, float(np.real(result.eigenvalue)), time.time() - t0

print("--- (A) raw-dollar Hamiltonian, genuinely noisy ---")
noisy_sampler_correct = AerSampler(seed=42, options=CORRECT_NOISY_OPTIONS)
noisy_sampler_correct.options.default_shots = 2000
print("verify noise attached:", noisy_sampler_correct._backend.options.noise_model is not None)
hist_a, final_a, t_a = run_vqe(ising_raw, noisy_sampler_correct)
print(f"genuinely-noisy raw-dollar COBYLA: {len(hist_a)} evals, {t_a:.1f}s, final={final_a:.4e}")
print(f"(compare) noiseless raw-dollar COBYLA final was: -1.9706e+08")
print(f"(compare) paper Fig 2 (noiseless) scale ~1e8, Fig 3 (noisy) scale ~1e9")

print("\n--- (B) real budget-normalized lambda=10 Hamiltonian, genuinely noisy ---")
noisy_sampler_correct2 = AerSampler(seed=42, options=CORRECT_NOISY_OPTIONS)
noisy_sampler_correct2.options.default_shots = 2000
hist_b, final_b, t_b = run_vqe(ising_real, noisy_sampler_correct2)
print(f"genuinely-noisy REAL-Hamiltonian COBYLA: {len(hist_b)} evals, {t_b:.1f}s, final={final_b:.4f}")
print(f"(compare) noiseless REAL-Hamiltonian COBYLA final (maxiter=200) was: -18.5572")
print(f"(compare) classical optimum (Ising energy): -20.845484")

with open("genuinely_noisy_results.pkl", "wb") as f:
    pickle.dump({"raw_dollar": hist_a, "real": hist_b}, f)
