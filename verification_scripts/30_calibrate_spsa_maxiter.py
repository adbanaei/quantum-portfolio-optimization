# Find the SPSA maxiter that lands near 100 evaluations.
import numpy as np, pickle
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA, NFT
from qiskit import transpile

with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)
dim_b = ising_op.num_qubits
sampler = AerSampler(seed=42, options={"backend_options": {"method": "matrix_product_state"}})
sampler.options.default_shots = 2000
ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                   entanglement="linear", reps=3)
ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
n_params = ansatz_d.num_parameters

for OptClass, name, maxiter in [(SPSA, "SPSA", 33), (SPSA, "SPSA", 20), (NFT, "NFT", 48)]:
    eval_count = [0]
    def cb(count, params, value, meta):
        eval_count[0] = count
    optimizer = OptClass(maxiter=maxiter)
    rng = np.random.default_rng(7)
    init = rng.uniform(-np.pi, np.pi, size=n_params)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
    vqe.compute_minimum_eigenvalue(ising_op)
    predicted = 3*maxiter+1 if name=="SPSA" else 2*maxiter+3
    print(f"{name:6s} maxiter={maxiter:4d} -> actual={eval_count[0]:4d}   (formula predicts {predicted})")

# 3x+1 doesn't hold at smaller maxiter, so narrow in directly
for maxiter in [25, 28]:
    eval_count = [0]
    def cb(count, params, value, meta):
        eval_count[0] = count
    optimizer = SPSA(maxiter=maxiter)
    rng = np.random.default_rng(7)
    init = rng.uniform(-np.pi, np.pi, size=n_params)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
    vqe.compute_minimum_eigenvalue(ising_op)
    print(f"SPSA   maxiter={maxiter:4d} -> actual={eval_count[0]:4d}")
