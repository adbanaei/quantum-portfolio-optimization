# Does the spread between ansatzes shrink with a bigger budget?
import time, pickle, numpy as np
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

ansatz_configs = {
    "PauliTwo":            dict(rotation_blocks=["ry"], entanglement_blocks="cz", entanglement="linear", reps=1),
    "TwoLocal full":       dict(rotation_blocks=["ry","rz"], entanglement_blocks="cz", entanglement="full", reps=3),
    "RealAmplit circular": dict(rotation_blocks=["ry"], entanglement_blocks="cx", entanglement="circular", reps=3),
}

for maxiter_target in [120, 800]:
    print(f"\n{'='*70}\nmaxiter={maxiter_target}\n{'='*70}")
    finals = {}
    for name, cfg in ansatz_configs.items():
        ansatz = TwoLocal(num_qubits=dim_b, **cfg)
        ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
        rng = np.random.default_rng(2024)
        init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
        optimizer = COBYLA(maxiter=maxiter_target)
        t0 = time.time()
        vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init)
        result = vqe.compute_minimum_eigenvalue(ising_op)
        e = float(np.real(result.eigenvalue))
        finals[name] = e
        print(f"  {name:22s} ({ansatz_d.num_parameters:3d} params): final energy = {e:8.3f}   ({time.time()-t0:.1f}s)")
    spread = max(finals.values()) - min(finals.values())
    print(f"  spread across the three: {spread:.3f}")

print(f"\nclassical optimum, Ising energy: -20.845")
