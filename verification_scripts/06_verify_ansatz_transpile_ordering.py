# Can decompose()+transpile() permute the qubits before measurement?
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.circuit.library import TwoLocal
import numpy as np

sampler = AerSampler(seed=42)
sampler.options.default_shots = 200
print("Does AerSampler (SamplerV2) have `_backend`?", hasattr(sampler, "_backend"))
print("Does AerSampler (SamplerV2) have `backend`?  ", hasattr(sampler, "backend"))
print("-> _prepare_ansatz_for_sampler will take the 'backend is None' branch:",
      not hasattr(sampler, "_backend") and not hasattr(sampler, "backend"))

print("\n--- X on qubit k, through the notebook's decompose()+transpile() ---")
n_qubits = 6
all_ok = True
for k in range(n_qubits):
    qc = QuantumCircuit(n_qubits, name="probe")
    qc.x(k)

    # _prepare_ansatz_for_sampler, backend=None branch
    prepared = qc.decompose(reps=10)
    prepared = transpile(prepared, optimization_level=1)

    # then decode_vqe_result's steps
    bound = prepared
    bound.measure_all()
    transpiled = transpile(bound, optimization_level=1)

    job = sampler.run([transpiled], shots=200)
    counts = job.result()[0].data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)

    expected = np.zeros(n_qubits); expected[k] = 1.0
    ok = np.array_equal(b_vec, expected)
    all_ok &= ok
    print(f"  X on qubit {k} -> decode()+transpile() -> b_vec={b_vec.astype(int)}  "
          f"expected={expected.astype(int)}  {'OK' if ok else 'MISMATCH'}")

print(f"\nall ok: {all_ok}")

print("\n--- same, with a real TwoLocal ansatz: RY(pi) on qubit k, rest at 0 ---")
all_ok2 = True
for k in range(n_qubits):
    ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks=["ry"], entanglement_blocks="cz",
                       entanglement="linear", reps=0)  # reps=0: just one rotation layer, no entangler
    params = np.zeros(ansatz.num_parameters)
    params[k] = np.pi  # reps=0 TwoLocal has one RY per qubit, in qubit order
    bound = ansatz.assign_parameters(params)

    prepared = bound.decompose(reps=10)
    prepared = transpile(prepared, optimization_level=1)
    prepared.measure_all()
    transpiled = transpile(prepared, optimization_level=1)

    job = sampler.run([transpiled], shots=200)
    counts = job.result()[0].data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)

    expected = np.zeros(n_qubits); expected[k] = 1.0
    ok = np.array_equal(b_vec, expected)
    all_ok2 &= ok
    print(f"  RY(pi) on param/qubit {k} -> b_vec={b_vec.astype(int)}  expected={expected.astype(int)}  "
          f"{'OK' if ok else 'MISMATCH'}")

print(f"\nall ok: {all_ok2}")
