# Script 06 again, but through the notebook's own _prepare_ansatz_for_sampler.
import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.circuit.library import TwoLocal

sampler = AerSampler(seed=42)
sampler.options.default_shots = 200
print("type(sampler._backend):", type(sampler._backend))
print("sampler._backend:", sampler._backend)
print("sampler._backend.configuration().coupling_map:",
      getattr(sampler._backend.configuration(), "coupling_map", "N/A")
      if hasattr(sampler._backend, "configuration") else "no .configuration()")

# copied verbatim from the notebook
def _prepare_ansatz_for_sampler(ansatz, sampler):
    prepared = ansatz.decompose(reps=10)
    backend = None
    if hasattr(sampler, "_backend"):
        backend = sampler._backend
    elif hasattr(sampler, "backend"):
        backend = sampler.backend
    if backend is not None:
        try:
            prepared = transpile(prepared, backend=backend, optimization_level=1)
        except Exception as exc:
            print(f"    [transpile-with-backend FAILED: {type(exc).__name__}: {exc}; falling back]")
            prepared = transpile(prepared, optimization_level=1)
    else:
        prepared = transpile(prepared, optimization_level=1)
    return prepared


def decode_like_notebook(bound_circuit, sampler, shots=200):
    transpiled = transpile(bound_circuit, optimization_level=1)
    job = sampler.run([transpiled], shots=shots)
    pub_result = job.result()[0]
    counts = pub_result.data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    b_vec = np.array([int(c) for c in reversed(best_bitstring)], dtype=float)
    return b_vec


print("\n--- RY(pi) on qubit k, through _prepare_ansatz_for_sampler + decode ---")
n_qubits = 6
all_ok = True
for k in range(n_qubits):
    ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks=["ry"], entanglement_blocks="cz",
                       entanglement="linear", reps=0)
    ansatz_prepared = _prepare_ansatz_for_sampler(ansatz, sampler)

    params = np.zeros(ansatz_prepared.num_parameters)
    params[k] = np.pi
    bound = ansatz_prepared.assign_parameters(params)
    bound.measure_all()

    b_vec = decode_like_notebook(bound, sampler)
    expected = np.zeros(n_qubits); expected[k] = 1.0
    ok = np.array_equal(b_vec, expected)
    all_ok &= ok
    print(f"  RY(pi) intended for qubit {k} -> decoded b_vec={b_vec.astype(int)}  "
          f"expected={expected.astype(int)}  {'OK' if ok else '*** MISMATCH ***'}")

print(f"\nall ok: {all_ok}")

if not all_ok:
    print("\nmismatch -- dumping parameter order and layout:")
    ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks=["ry"], entanglement_blocks="cz",
                       entanglement="linear", reps=0)
    print("Original ansatz.parameters order:", list(ansatz.parameters))
    ansatz_prepared = _prepare_ansatz_for_sampler(ansatz, sampler)
    print("Prepared (decomposed+transpiled) circuit qubits:", ansatz_prepared.qubits)
    print("Prepared circuit .parameters order:", list(ansatz_prepared.parameters))
    print("Prepared circuit layout:", getattr(ansatz_prepared, "layout", None))
