# Does decode put qubit k's measured value at position k? Probe one qubit at a time.
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
import numpy as np

n_qubits = 6  # the mechanism doesn't depend on qubit count

sampler = AerSampler(seed=42)
sampler.options.default_shots = 200

print("X on qubit k only -> b_vec[k]=1, rest 0?\n")
all_ok = True
for k in range(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.x(k)
    qc.measure_all()
    transpiled = transpile(qc, optimization_level=1)
    job = sampler.run([transpiled], shots=200)
    counts = job.result()[0].data.meas.get_counts()

    # decode_vqe_result()'s logic, verbatim
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)

    expected = np.zeros(n_qubits); expected[k] = 1.0
    ok = np.array_equal(b_vec, expected)
    all_ok &= ok
    print(f"  X on qubit {k}:  raw_bitstring={best_bitstring}  "
          f"decoded b_vec={b_vec.astype(int)}  expected={expected.astype(int)}  {'OK' if ok else 'MISMATCH'}")

print(f"\nALL QUBIT POSITIONS CORRECTLY DECODED: {all_ok}")

print("\n--- is variable b_i the same qubit i that to_ising() used? ---")
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

for i_target in range(n_qubits):
    qp = QuadraticProgram()
    for i in range(n_qubits):
        qp.binary_var(name=f"b{i}")
    # only b_{i_target} gets a nonzero coefficient
    linear = {f"b{i}": (100.0 if i == i_target else 0.0) for i in range(n_qubits)}
    qp.minimize(linear=linear, quadratic={}, constant=0)
    converter = QuadraticProgramToQubo()
    ising_op, offset = to_ising(converter.convert(qp))
    ising_op = ising_op.simplify(atol=1e-9)
    labels = ising_op.paulis.to_labels()
    coeffs = [complex(c).real for c in ising_op.coeffs]
    nz = [(lab, c) for lab, c in zip(labels, coeffs) if abs(c) > 1e-6]
    assert len(nz) == 1, f"expected exactly one nonzero term, got {nz}"
    lab = nz[0][0]
    qubit_with_z = n_qubits - 1 - lab.index("Z")
    print(f"  variable b{i_target} (only nonzero coef) -> Pauli term {lab} -> Z is on qubit {qubit_with_z}  "
          f"{'OK (i==qubit)' if qubit_with_z == i_target else 'MISMATCH'}")
