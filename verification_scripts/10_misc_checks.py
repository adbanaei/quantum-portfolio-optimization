# Two small things flagged while reading the notebook.
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import TwoLocal

print("--- is QuadraticProgramToQubo() a no-op on an already-binary, "
      "unconstrained QP, as the notebook's comment claims? ---")
qp = QuadraticProgram()
for i in range(4):
    qp.binary_var(name=f"b{i}")
lin = {"b0": 1.1, "b1": -2.2, "b2": 3.3, "b3": -0.5}
quad = {("b0","b1"): 0.7, ("b2","b3"): -1.4}
qp.minimize(linear=lin, quadratic=quad, constant=9.9)

converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)

same_linear = qp.objective.linear.to_dict(use_name=True) == qubo.objective.linear.to_dict(use_name=True)
same_quad = qp.objective.quadratic.to_dict(use_name=True) == qubo.objective.quadratic.to_dict(use_name=True)
same_const = qp.objective.constant == qubo.objective.constant
same_sense = qp.objective.sense == qubo.objective.sense
print(f"linear coefficients identical  : {same_linear}")
print(f"quadratic coefficients identical: {same_quad}")
print(f"constant identical              : {same_const}  ({qp.objective.constant} vs {qubo.objective.constant})")
print(f"sense identical                 : {same_sense}")
print("pass-through:", all([same_linear, same_quad, same_const, same_sense]))

print("\n--- do two separately built, identical TwoLocal instances give the same "
      ".parameters order? (section 14 rebuilds the ansatz and reuses "
      "result.optimal_parameters from a different object) ---")
a1 = TwoLocal(num_qubits=15, rotation_blocks=["ry","rz"], entanglement_blocks="cz",
              entanglement="linear", reps=3)
a2 = TwoLocal(num_qubits=15, rotation_blocks=["ry","rz"], entanglement_blocks="cz",
              entanglement="linear", reps=3)
p1 = [str(p) for p in a1.parameters]
p2 = [str(p) for p in a2.parameters]
print(f"a1 has {len(p1)} params, a2 has {len(p2)} params")
print(f"Parameter NAME sequences identical: {p1 == p2}")
print(f"First 6 param names (a1): {p1[:6]}")
print(f"First 6 param names (a2): {p2[:6]}")

# and again after decompose+transpile, the path actually used
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
sampler = AerSampler(seed=42)
a1d = transpile(a1.decompose(reps=10), backend=sampler._backend, optimization_level=1)
a2d = transpile(a2.decompose(reps=10), backend=sampler._backend, optimization_level=1)
p1d = [str(p) for p in a1d.parameters]
p2d = [str(p) for p in a2d.parameters]
print(f"\nAfter decompose()+transpile(): identical order: {p1d == p2d}")
print(f"Same number of params after decompose+transpile: {len(p1d)} vs {len(p2d)}")
