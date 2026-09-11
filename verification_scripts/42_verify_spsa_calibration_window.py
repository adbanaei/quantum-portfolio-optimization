# How many SPSA evaluations happen before the first parameter update?
import numpy as np
from qiskit_algorithms.optimizers import COBYLA, NFT, SPSA

calls = {"n": 0}
history = []


def loss(x):
    calls["n"] += 1
    v = float(np.sum(x ** 2))
    history.append(v)
    return v


def run(opt, dim):
    calls["n"] = 0
    history.clear()
    opt.minimize(loss, np.full(dim, 0.3))
    return calls["n"], list(history)


print("--- evaluation counts vs. nominal maxiter ---")
for maxiter in (50, 100, 200):
    n_default, _ = run(SPSA(maxiter=maxiter), 8)
    n_fixed, _ = run(SPSA(maxiter=maxiter, learning_rate=0.01, perturbation=0.01), 8)
    print(f"  SPSA(maxiter={maxiter:>3}): default {n_default:>4} evals | "
          f"calibration disabled {n_fixed:>4} evals | difference {n_default - n_fixed:>3}")

# Distance from x0 is the clean signal; the loss can overshoot right after calibration.
print("\n--- where the first parameter update happens ---")
x0 = np.full(8, 0.3)
distances = []


def loss_tracked(x):
    distances.append(float(np.linalg.norm(np.asarray(x) - x0)))
    return float(np.sum(np.asarray(x) ** 2))


SPSA(maxiter=100).minimize(loss_tracked, x0)
d = np.asarray(distances)
print(f"  evaluations 1-50   : ||x - x0|| in [{d[:50].min():.4f}, {d[:50].max():.4f}]  "
      f"(spread {np.ptp(d[:50]):.4f})")
print(f"  evaluations 51-100 : ||x - x0|| in [{d[50:100].min():.4f}, {d[50:100].max():.4f}]  "
      f"(spread {np.ptp(d[50:100]):.4f})")
print(f"  calibration spread is exactly zero: {np.ptp(d[:50]) == 0.0}")

print("\n--- the other two optimizers, for contrast ---")
for label, opt, dim in [("COBYLA, 120 params", COBYLA(maxiter=100), 120),
                        ("COBYLA,  60 params", COBYLA(maxiter=100), 60),
                        ("NFT,      8 params", NFT(maxiter=50), 8)]:
    n_evals, _ = run(opt, dim)
    print(f"  {label}: {n_evals} evaluations (no calibration phase)")
