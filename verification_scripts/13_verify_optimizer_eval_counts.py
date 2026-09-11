# How many evaluations does each optimizer really spend per iteration?
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NFT
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile

n_qubits = 4
ising_op = SparsePauliOp.from_list([("ZIII", 1.0), ("IZII", -0.5), ("IIZI", 0.7),
                                     ("IIIZ", -0.3), ("ZZII", 0.4)])

sampler = AerSampler(seed=1)
sampler.options.default_shots = 200

MAXITER = 50
for OptClass, name in [(COBYLA, "COBYLA"), (SPSA, "SPSA"), (NFT, "NFT")]:
    ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                       entanglement="linear", reps=1)
    ansatz_d = transpile(ansatz.decompose(reps=10), backend=sampler._backend, optimization_level=1)
    eval_count = [0]
    def cb(count, params, value, meta):
        eval_count[0] = count
    optimizer = OptClass(maxiter=MAXITER)
    rng = np.random.default_rng(0)
    init = rng.uniform(-np.pi, np.pi, size=ansatz_d.num_parameters)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_d, optimizer=optimizer, initial_point=init, callback=cb)
    result = vqe.compute_minimum_eigenvalue(ising_op)
    print(f"{name:10s}: configured maxiter={MAXITER:4d}  ->  actual callback eval_count reached = {eval_count[0]}"
          f"   (ratio = {eval_count[0]/MAXITER:.2f}x)")
