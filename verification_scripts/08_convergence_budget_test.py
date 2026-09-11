# Is COBYLA maxiter=200-250 enough for a 120-parameter ansatz? One long run, full history.
import time, pickle, numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit import transpile

mu_pp = np.load("mu_pp.npy"); Sigma_pp = np.load("Sigma_pp.npy")
P_pp = np.load("P_pp.npy"); C = np.load("C.npy")
with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)

RISK_Q, LAMBDA = 0.5, 10.0
dim_b = len(mu_pp)

def evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b_vec, dtype=float)
    return mu_pp @ b - q * (b @ Sigma_pp @ b) - lam * (P_pp @ b - 1) ** 2

import itertools
best_val, best_b = -np.inf, None
for bits in itertools.product([0, 1], repeat=dim_b):
    val = evaluate_portfolio(bits, mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
    if val > best_val:
        best_val, best_b = val, np.array(bits)
print(f"classical optimum L*={best_val:.8f}  b*={best_b}")

sampler = AerSampler(seed=42)
sampler.options.default_shots = 2000
sampler.options.backend_options = {"method": "matrix_product_state",
                                    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0}

ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                   entanglement="linear", reps=3)
ansatz_d = ansatz.decompose(reps=10)
ansatz_d = transpile(ansatz_d, backend=sampler._backend, optimization_level=1)
print("num_parameters:", ansatz_d.num_parameters)

MAXITER = 1800
history = []
def callback(eval_count, params, value, meta):
    history.append((eval_count, float(value)))
    if eval_count % 100 == 0:
        print(f"  eval={eval_count:5d}  energy={value:10.4f}  elapsed={time.time()-t0:7.1f}s")

rng = np.random.default_rng(12345)
init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)

optimizer = COBYLA(maxiter=MAXITER)
vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer,
                   initial_point=init, callback=callback)

t0 = time.time()
result = vqe.compute_minimum_eigenvalue(ising_op)
elapsed = time.time() - t0
print(f"\nTotal time for {MAXITER} maxiter: {elapsed:.1f}s ({len(history)} evals recorded)")
print(f"Final eigenvalue: {result.eigenvalue}")

with open("long_run_result.pkl", "wb") as f:
    pickle.dump({"history": history, "result_eigenvalue": complex(result.eigenvalue),
                 "optimal_parameters": list(result.optimal_parameters.values())
                             if isinstance(result.optimal_parameters, dict)
                             else list(result.optimal_parameters),
                 "elapsed": elapsed}, f)

print("\nSaved to long_run_result.pkl")
