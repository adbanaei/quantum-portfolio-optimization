# Routing overhead on the real Cairo coupling map, which the noise model doesn't charge.
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import PauliTwoDesign, RealAmplitudes, TwoLocal
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

N_QUBITS, REPS, SEED = 15, 3, 42
backend = FakeCairoV2()
basis = NoiseModel.from_backend(backend).basis_gates
coupling = backend.coupling_map
print(f"Cairo: {backend.num_qubits} qubits, {len(list(coupling))} directed edges")
print(f"noise-model basis gates: {sorted(basis)}\n")

ansatzes = {
    "PauliTwo": PauliTwoDesign(num_qubits=N_QUBITS, reps=REPS, seed=SEED),
    **{f"TwoLocal {e}": TwoLocal(num_qubits=N_QUBITS, rotation_blocks=["ry", "rz"],
                                 entanglement_blocks="cz", entanglement=e, reps=REPS)
       for e in ("linear", "full", "circular", "pairwise")},
    **{f"RealAmplit. {e}": RealAmplitudes(num_qubits=N_QUBITS, entanglement=e, reps=REPS)
       for e in ("linear", "full", "circular", "pairwise")},
}

TWO_QUBIT = ("cx", "cz", "ecr", "swap")


def cost(ansatz, coupling_map):
    tqc = transpile(ansatz.decompose(reps=10), basis_gates=basis, coupling_map=coupling_map,
                    optimization_level=1, seed_transpiler=11)
    ops = tqc.count_ops()
    return sum(v for k, v in ops.items() if k in TWO_QUBIT), tqc.depth()


print(f"{'ansatz':<24}{'2q sim':>8}{'2q dev':>8}{'factor':>9}{'depth sim':>11}{'depth dev':>11}")
print("-" * 71)
overheads = {}
for name, ansatz in ansatzes.items():
    sim_2q, sim_d = cost(ansatz, None)
    dev_2q, dev_d = cost(ansatz, coupling)
    overheads[name] = dev_2q / sim_2q
    print(f"{name:<24}{sim_2q:>8}{dev_2q:>8}{overheads[name]:>8.1f}x{sim_d:>11}{dev_d:>11}")

print("\nby entanglement pattern:")
for pattern in ("linear", "pairwise", "circular", "full"):
    vals = [v for k, v in overheads.items() if k.endswith(pattern)]
    print(f"  {pattern:<10} mean two-qubit overhead {np.mean(vals):.1f}x")
print(f"  {'PauliTwo':<10} mean two-qubit overhead {overheads['PauliTwo']:.1f}x")
