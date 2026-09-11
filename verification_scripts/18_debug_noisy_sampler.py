# Is the noisy sampler applying any noise at all? GHZ probe.
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

cairo_nm = NoiseModel.from_backend(FakeCairoV2())
print("noise model basis gates:", cairo_nm.basis_gates)

n = 6
qc = QuantumCircuit(n)
qc.h(0)
for i in range(n - 1):
    qc.cx(i, i + 1)
qc.measure_all()

ideal_sampler = AerSampler(seed=1)
ideal_sampler.options.default_shots = 4000

noisy_sampler = AerSampler(seed=1)
noisy_sampler.options.default_shots = 4000
noisy_sampler.options.backend_options = {
    "noise_model": cairo_nm, "method": "statevector",
    "max_parallel_threads": 0, "max_parallel_experiments": 0, "max_parallel_shots": 0,
}

for label, sampler in [("IDEAL", ideal_sampler), ("NOISY (Cairo)", noisy_sampler)]:
    tqc = transpile(qc, backend=sampler._backend, optimization_level=1)
    job = sampler.run([tqc], shots=4000)
    counts = job.result()[0].data.meas.get_counts()
    n_distinct = len(counts)
    top5 = sorted(counts.items(), key=lambda x: -x[1])[:5]
    print(f"\n{label}: {n_distinct} distinct bitstrings (an ideal GHZ gives 2)")
    for bs, c in top5:
        print(f"    {bs}: {c}")

# is noisy_sampler._backend itself noisy, independent of options.backend_options?
print("\nideal_sampler._backend:", ideal_sampler._backend)
print("noisy_sampler._backend:", noisy_sampler._backend)
print("are they the same object/type?", type(ideal_sampler._backend) == type(noisy_sampler._backend))
