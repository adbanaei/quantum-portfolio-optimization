# <H> at a few fixed parameter points, noisy against noiseless.
import time, numpy as np, pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit_algorithms import SamplingVQE
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
    for i in range(n): qp.binary_var(name=f"b{i}")
    linear = {f"b{i}": -mu_pp[i] + q*Sigma_pp[i,i] + lam*(P_pp[i]**2 - 2*P_pp[i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            v = 2*q*Sigma_pp[i,j] + 2*lam*P_pp[i]*P_pp[j]
            if v != 0: quadratic[(f"b{i}", f"b{j}")] = v
    qp.minimize(linear=linear, quadratic=quadratic, constant=lam)
    return qp

P_prime = P / BUDGET
mu_pp = C.T @ (P_prime*mu); Sigma_pp = C.T @ (P_prime[:,None]*Sigma*P_prime[None,:]) @ C; P_pp = C.T @ P_prime
ising_real, offset_real = to_ising(QuadraticProgramToQubo().convert(build_qubo(mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)))
ising_real = ising_real.simplify(atol=1e-6)

mu_pp_raw = C.T @ (P*mu); Sigma_pp_raw = C.T @ (P[:,None]*Sigma*P[None,:]) @ C; P_pp_raw = C.T @ P
ising_raw, offset_raw = to_ising(QuadraticProgramToQubo().convert(build_qubo(mu_pp_raw, Sigma_pp_raw, P_pp_raw, RISK_Q, LAMBDA)))
ising_raw = ising_raw.simplify(atol=1e-6)

cairo_nm = NoiseModel.from_backend(FakeCairoV2())
SHOTS = 2000

ideal_sampler = AerSampler(seed=42, options={"backend_options": {"method": "matrix_product_state"}})
ideal_sampler.options.default_shots = SHOTS
noisy_sampler = AerSampler(seed=42, options={"backend_options": {"noise_model": cairo_nm, "method": "statevector"}})
noisy_sampler.options.default_shots = SHOTS
print("noise genuinely attached:", noisy_sampler._backend.options.noise_model is not None)

ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry","rz"], entanglement_blocks="cz", entanglement="linear", reps=3)
rng = np.random.default_rng(12345)
test_points = [rng.uniform(-np.pi, np.pi, size=ansatz.decompose(reps=10).num_parameters) for _ in range(3)]

for ham_name, ham, offset in [("REAL (budget-normalized)", ising_real, offset_real),
                                ("RAW-DOLLAR", ising_raw, offset_raw)]:
    print(f"\n{'='*72}\n{ham_name} Hamiltonian (offset={offset:.4e})\n{'='*72}")
    # energy off SamplingVQE's first callback value, optimizer clamped as low as it goes
    from qiskit_algorithms.optimizers import COBYLA
    for sname, sampler in [("ideal", ideal_sampler), ("noisy(Cairo)", noisy_sampler)]:
        ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
        first_vals = []
        t0=time.time()
        for i, pt in enumerate(test_points[:1]):
            hist=[]
            def cb(c,p,v,m,_h=hist): _h.append(v)
            vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d,
                               optimizer=COBYLA(maxiter=1), initial_point=pt, callback=cb)
            vqe.compute_minimum_eigenvalue(ham)
            first_vals.append(hist[0])
        print(f"  {sname:15s}: <H> at random init point = {first_vals[0]:.4e}  ({time.time()-t0:.1f}s)")
