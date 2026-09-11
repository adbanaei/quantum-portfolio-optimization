# Decode the long COBYLA run and tabulate energy against eval count.
import pickle, numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit import transpile

mu_pp = np.load("mu_pp.npy"); Sigma_pp = np.load("Sigma_pp.npy")
P_pp = np.load("P_pp.npy"); C = np.load("C.npy")
with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)
with open("long_run_result.pkl", "rb") as f:
    saved = pickle.load(f)

RISK_Q, LAMBDA = 0.5, 10.0
dim_b = len(mu_pp)

def evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, q, lam):
    b = np.array(b_vec, dtype=float)
    return mu_pp @ b - q * (b @ Sigma_pp @ b) - lam * (P_pp @ b - 1) ** 2

b_opt_classical = np.array([0,1,0,0,0,0,0,0,1,0,0,1,1,1,0])
val_opt_classical = evaluate_portfolio(b_opt_classical, mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)

sampler = AerSampler(seed=42)
sampler.options.default_shots = 2000
sampler.options.backend_options = {"method": "matrix_product_state",
                                    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0}

ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                   entanglement="linear", reps=3)
ansatz_d = ansatz.decompose(reps=10)
ansatz_d = transpile(ansatz_d, backend=sampler._backend, optimization_level=1)

bound = ansatz_d.assign_parameters(saved["optimal_parameters"])
bound.measure_all()
transpiled = transpile(bound, optimization_level=1)
job = sampler.run([transpiled], shots=2000)
counts = job.result()[0].data.meas.get_counts()
best_bitstring = max(counts, key=counts.get)
b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)
top_prob = counts[best_bitstring] / sum(counts.values())

obj = evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
match = np.allclose(b_vec, b_opt_classical)
hamming = int(np.sum(b_vec != b_opt_classical))

print(f"long run: COBYLA, {len(saved['history'])} evals, {saved['elapsed']:.1f}s")
print(f"Decoded b_vec       : {b_vec.astype(int)}")
print(f"Classical b*        : {b_opt_classical}")
print(f"Hamming distance    : {hamming} / {dim_b} bits differ")
print(f"Mode-bitstring prob : {top_prob:.3f} (out of 2000 shots)")
print(f"Objective (VQE)     : {obj:.8f}")
print(f"Objective (classical): {val_opt_classical:.8f}")
print(f"Match               : {match}")

# the notebook's own maxiter=200 run: hamming 6/15, objective -0.236465

print("\n--- gap to the classical optimum at various eval counts ---")
ising_energy_opt = -val_opt_classical - offset
print(f"Classical optimum in Ising-energy space: {ising_energy_opt:.4f}")
print(f"{'eval#':>8} {'energy':>12} {'gap_to_opt':>12} {'gap_%':>8}")
checkpoints = [50, 100, 150, 200, 250, 300, 500, 700, 900, 1100]
hist = dict(saved["history"])
hist_sorted = sorted(saved["history"])
for cp in checkpoints:
    vals = [v for e, v in hist_sorted if e <= cp]
    if not vals:
        continue
    e_at_cp = vals[-1]
    gap = e_at_cp - ising_energy_opt
    pct = 100 * gap / abs(ising_energy_opt)
    print(f"{cp:>8} {e_at_cp:>12.4f} {gap:>12.4f} {pct:>7.2f}%")
