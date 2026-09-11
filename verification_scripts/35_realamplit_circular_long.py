import time, numpy as np, pickle
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit import transpile

with open("ising_op.pkl", "rb") as f:
    ising_op, offset = pickle.load(f)
dim_b = ising_op.num_qubits
sampler = AerSampler(seed=42, options={"backend_options": {"method": "matrix_product_state"}})
sampler.options.default_shots = 2000

ansatz = TwoLocal(num_qubits=dim_b, rotation_blocks=["ry"], entanglement_blocks="cx", entanglement="circular", reps=3)
ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
rng = np.random.default_rng(2024)
init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)

t0 = time.time()
def cb(count, params, value, meta):
    if count % 40 == 0:
        print(f"  ...eval {count}: energy={value:.3f}  elapsed={time.time()-t0:.0f}s", flush=True)

vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=COBYLA(maxiter=350), initial_point=init, callback=cb)
result = vqe.compute_minimum_eigenvalue(ising_op)
print(f"RealAmplit circular (60 params) maxiter=350: final = {float(np.real(result.eigenvalue)):.3f}  ({time.time()-t0:.1f}s)")
