# Check to_ising()'s sign and offset convention against a symbolic derivation.
import numpy as np
import sympy as sp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

n = 3
rng = np.random.default_rng(7)
lin = {f"b{i}": round(float(rng.uniform(-3, 3)), 4) for i in range(n)}
quad = {}
for i in range(n):
    for j in range(i + 1, n):
        quad[(f"b{i}", f"b{j}")] = round(float(rng.uniform(-2, 2)), 4)
const = round(float(rng.uniform(-5, 5)), 4)

print("QP built with:")
print("linear:", lin)
print("quadratic:", quad)
print("constant:", const)

qp = QuadraticProgram(name="toy")
for i in range(n):
    qp.binary_var(name=f"b{i}")
qp.minimize(linear=lin, quadratic=quad, constant=const)
print("\nQP objective (as built):")
print(qp.prettyprint())

s = sp.symbols(f"s0:{n}")
b_of_s = [(1 + si) / 2 for si in s]

expr = const
for i in range(n):
    expr += lin[f"b{i}"] * b_of_s[i]
for i in range(n):
    for j in range(i + 1, n):
        expr += quad[(f"b{i}", f"b{j}")] * b_of_s[i] * b_of_s[j]

expr = sp.expand(expr)
print("\nsymbolic objective in s_i:", expr)

poly = sp.Poly(expr, *s)
offset_hand = float(poly.coeff_monomial(1))
h_hand = {}
for i in range(n):
    mono = [0] * n
    mono[i] = 1
    h_hand[i] = float(poly.coeff_monomial(sp.prod(si**e for si, e in zip(s, mono))))
J_hand = {}
for i in range(n):
    for j in range(i + 1, n):
        mono = [0] * n
        mono[i] = 1
        mono[j] = 1
        J_hand[(i, j)] = float(poly.coeff_monomial(sp.prod(si**e for si, e in zip(s, mono))))

print(f"\nHand-derived offset : {offset_hand:.6f}")
print(f"Hand-derived h_i    : {h_hand}")
print(f"Hand-derived J_ij   : {J_hand}")

converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)
ising_op, offset_qiskit = to_ising(qubo)
ising_op = ising_op.simplify(atol=1e-10)

print(f"\nQiskit offset: {offset_qiskit:.6f}")
print("Qiskit Pauli terms:")
h_qiskit = {}
J_qiskit = {}
for pauli, coeff in zip(ising_op.paulis.to_labels(), ising_op.coeffs):
    coeff = complex(coeff).real
    z_positions = [n - 1 - k for k, ch in enumerate(pauli) if ch == "Z"]
    print(f"  {pauli}  coeff={coeff:.6f}   (Z on qubit(s) {z_positions})")
    if len(z_positions) == 1:
        h_qiskit[z_positions[0]] = coeff
    elif len(z_positions) == 2:
        J_qiskit[tuple(sorted(z_positions))] = coeff

print(f"\n{'term':<10}{'hand':>12}{'qiskit':>12}{'match?':>10}")
ok = True
row = ("offset", offset_hand, offset_qiskit)
match = abs(row[1] - row[2]) < 1e-6
ok &= match
print(f"{row[0]:<10}{row[1]:>12.6f}{row[2]:>12.6f}{str(match):>10}")
for i in range(n):
    hv = h_hand[i]
    qv = h_qiskit.get(i, 0.0)
    match = abs(hv - qv) < 1e-6
    ok &= match
    print(f"{'h_'+str(i):<10}{hv:>12.6f}{qv:>12.6f}{str(match):>10}")
for i in range(n):
    for j in range(i + 1, n):
        hv = J_hand[(i, j)]
        qv = J_qiskit.get((i, j), 0.0)
        match = abs(hv - qv) < 1e-6
        ok &= match
        print(f"{'J_'+str(i)+str(j):<10}{hv:>12.6f}{qv:>12.6f}{str(match):>10}")

print("term-by-term match:", ok)

print("\n--- <b|H|b> vs qp.objective(b), all 2^n bitstrings ---")
import itertools
from qiskit.quantum_info import Statevector

all_ok = True
for bits in itertools.product([0, 1], repeat=n):
    b_arr = np.array(bits)
    qp_val = qp.objective.evaluate(b_arr)
    bitstring = "".join(str(bit) for bit in reversed(bits))
    sv = Statevector.from_label(bitstring)
    h_expect = sv.expectation_value(ising_op).real

    match = abs(qp_val - h_expect) < 1e-6
    all_ok &= match
    print(f"  b={bits}  qp.objective={qp_val:9.5f}  <H>={h_expect:9.5f}  match={match}")

print("\nall bitstrings match:", all_ok)
