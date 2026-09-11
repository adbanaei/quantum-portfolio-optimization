# Which spin polarity does to_ising() use? Sweep both, on asymmetric coefficients.
import numpy as np
import itertools
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising
from qiskit.quantum_info import Statevector

n = 3
# Deliberately asymmetric so no permutation/sign choice looks accidentally right
lin = {"b0": 1.0, "b1": 2.0, "b2": 5.0}
quad = {("b0", "b1"): 0.3, ("b0", "b2"): 0.7, ("b1", "b2"): 1.1}
const = -1.5

qp = QuadraticProgram(name="toy2")
for i in range(n):
    qp.binary_var(name=f"b{i}")
qp.minimize(linear=lin, quadratic=quad, constant=const)

converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)
ising_op, offset = to_ising(qubo)
ising_op = ising_op.simplify(atol=1e-12)
print("QP objective:", qp.objective.linear.to_dict(use_name=True),
      qp.objective.quadratic.to_dict(use_name=True), qp.objective.constant)
print("Ising offset:", offset)
for pauli, coeff in zip(ising_op.paulis.to_labels(), ising_op.coeffs):
    print(f"  {pauli}  {complex(coeff).real:.6f}")

def qp_obj(b_tuple):
    return qp.objective.evaluate(np.array(b_tuple, dtype=float))

def h_expect(b_tuple):
    """<b|H|b> using Qiskit's own bit ordering: b_tuple[0] is variable b0.
    Statevector.from_label expects leftmost char = highest qubit index, so
    to place variable b_i's VALUE onto qubit i we must reverse b_tuple."""
    label = "".join(str(int(x)) for x in reversed(b_tuple))
    sv = Statevector.from_label(label)
    return sv.expectation_value(ising_op).real

print("\n--- <bitstring|H|bitstring> vs qp.objective(b), all 8 b ---")
print(f"{'b(var order)':<14}{'qp.obj':>10}{'<H> (var_i->qubit_i)':>24}")
for bits in itertools.product([0, 1], repeat=n):
    o = qp_obj(bits)
    h = h_expect(bits)
    print(f"{str(bits):<14}{o:>10.4f}{h:>24.4f}   diff={o-h:+.4f}")

print("\n--- same, against the bit-flipped vector (1-b) ---")
for bits in itertools.product([0, 1], repeat=n):
    flipped = tuple(1 - x for x in bits)
    o_flipped = qp_obj(flipped)
    h = h_expect(bits)
    print(f"{str(bits):<14} qp.obj(1-b)={o_flipped:>8.4f}   <H>={h:>8.4f}   diff={o_flipped-h:+.4f}")
