# At the same budget, does a smaller ansatz land closer to the optimum?
import time, pickle, numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit import transpile

mu_pp = np.load("mu_pp.npy"); Sigma_pp = np.load("Sigma_pp.npy"); P_pp = np.load("P_pp.npy")
with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)
dim_b = len(mu_pp)
RISK_Q, LAMBDA = 0.5, 10.0

def evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b_vec, dtype=float)
    return mu_pp @ b - q * (b @ Sigma_pp @ b) - lam * (P_pp @ b - 1) ** 2

b_opt_classical = np.array([0,1,0,0,0,0,0,0,1,0,0,1,1,1,0])
val_opt_classical = evaluate_portfolio(b_opt_classical, mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
ising_energy_opt = -val_opt_classical - offset

sampler = AerSampler(seed=42)
sampler.options.default_shots = 2000
sampler.options.backend_options = {"method": "matrix_product_state",
                                    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0}

print(f"Classical optimum in Ising-energy space: {ising_energy_opt:.4f}\n")
print(f"{'reps':>5}{'#params':>10}{'maxiter':>9}{'final_energy':>14}{'gap_%':>9}{'match':>8}{'hamming':>9}{'time_s':>8}")

for reps in [1, 2, 3]:
    ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                       entanglement="linear", reps=reps)
    ansatz_d = ansatz.decompose(reps=10)
    ansatz_d = transpile(ansatz_d, backend=sampler._backend, optimization_level=1)

    rng = np.random.default_rng(12345)  # same seed across reps
    init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)

    optimizer = COBYLA(maxiter=200)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init)
    t0 = time.time()
    result = vqe.compute_minimum_eigenvalue(ising_op)
    elapsed = time.time() - t0

    bound = ansatz_d.assign_parameters(list(result.optimal_parameters.values())
                                        if isinstance(result.optimal_parameters, dict)
                                        else list(result.optimal_parameters))
    bound.measure_all()
    transpiled = transpile(bound, optimization_level=1)
    job = sampler.run([transpiled], shots=2000)
    counts = job.result()[0].data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)
    match = np.allclose(b_vec, b_opt_classical)
    hamming = int(np.sum(b_vec != b_opt_classical))

    energy = float(np.real(result.eigenvalue))
    gap_pct = 100 * (energy - ising_energy_opt) / abs(ising_energy_opt)
    print(f"{reps:>5}{ansatz_d.num_parameters:>10}{200:>9}{energy:>14.4f}{gap_pct:>8.2f}%{str(match):>8}{hamming:>9}{elapsed:>8.1f}")
