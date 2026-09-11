# What do maxiter=100 (Cobyla) and 50 (SPSA/NFT) actually cost at this problem size?
import time, numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NFT
from qiskit import transpile
import pickle

with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)   # real budget-normalized lambda=10, 15 qubits

dim_b = ising_op.num_qubits
sampler = AerSampler(seed=42, options={"backend_options": {"method": "matrix_product_state"}})
sampler.options.default_shots = 2000

print(f"dim_b={dim_b}")
for reps in [3]:
    ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                       entanglement="linear", reps=reps)
    ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
    n_params = ansatz_d.num_parameters
    print(f"reps={reps}: {n_params} parameters, COBYLA's own floor = num_vars+2 = {n_params+2}")

    for OptClass, name, requested_maxiter in [(COBYLA, "COBYLA", 100), (SPSA, "SPSA", 50), (NFT, "NFT", 50)]:
        eval_count = [0]
        def cb(count, params, value, meta):
            eval_count[0] = count
        optimizer = OptClass(maxiter=requested_maxiter)
        rng = np.random.default_rng(7)
        init = rng.uniform(-np.pi, np.pi, size=n_params)
        vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
        t0 = time.time()
        result = vqe.compute_minimum_eigenvalue(ising_op)
        elapsed = time.time() - t0
        print(f"  {name:10s}: requested maxiter={requested_maxiter:4d}  ->  actual eval_count={eval_count[0]:4d}"
              f"   ({elapsed:.1f}s)")
