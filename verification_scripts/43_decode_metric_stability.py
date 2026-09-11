# Three decode rules, 27 configurations, five shot seeds each.
import glob
import itertools

import numpy as np
import pandas as pd
import yaml
from qiskit import transpile
from qiskit.circuit.library import PauliTwoDesign, RealAmplitudes, TwoLocal
from qiskit_aer.primitives import SamplerV2 as AerSampler

BUDGET, RISK_Q, LAMBDA, REPS, SEED = 2000.0, 0.5, 10.0, 3, 42
SHOT_SEEDS = [42, 1, 7, 123, 2024]

series = {}
for t in ["AAPL", "IBM", "NFLX", "TSLA"]:
    df = pd.read_csv(f"./price_data/{t}.csv", index_col=0, parse_dates=True)
    s = df["Corrected_Adj_Close"].copy()
    s.name = t
    series[t] = s
prices = pd.concat(series.values(), axis=1)
prices = prices.loc[(prices.index >= "2011-12-23") & (prices.index < "2022-10-21")].sort_index().dropna()
P = prices.iloc[-1].values.astype(float)
rets = prices.pct_change().dropna()
mu, Sigma = rets.mean().values, rets.cov().values

Pn = P / BUDGET
mu_p, Sig_p = Pn * mu, (Pn[:, None] * Sigma) * Pn[None, :]
n_max = [int(BUDGET / p) for p in P]
d_list = [int(np.floor(np.log2(nm))) for nm in n_max]
C = np.zeros((len(P), sum(d + 1 for d in d_list)))
col = 0
for i in range(len(P)):
    for j in range(d_list[i] + 1):
        C[i, col] = 2 ** j
        col += 1
mu_pp, Sig_pp, P_pp = C.T @ mu_p, C.T @ Sig_p @ C, C.T @ Pn


def objective(b):
    b = np.asarray(b, dtype=float)
    return mu_pp @ b - RISK_Q * (b @ Sig_pp @ b) - LAMBDA * (P_pp @ b - 1) ** 2


allb = np.array(list(itertools.product([0, 1], repeat=C.shape[1])), dtype=float)
vals = (allb @ mu_pp) - RISK_Q * np.einsum("ij,jk,ik->i", allb, Sig_pp, allb) \
    - LAMBDA * (allb @ P_pp - 1) ** 2
L_star = vals.max()


def build(name):
    if name == "PauliTwo":
        return PauliTwoDesign(num_qubits=C.shape[1], reps=REPS, seed=SEED)
    family, ent = name.split()
    if family == "TwoLocal":
        return TwoLocal(num_qubits=C.shape[1], rotation_blocks=["ry", "rz"],
                        entanglement_blocks="cz", entanglement=ent, reps=REPS)
    return RealAmplitudes(num_qubits=C.shape[1], entanglement=ent, reps=REPS)


state_path = sorted(glob.glob("./run_states/*.yaml"))[-1]
state = yaml.safe_load(open(state_path))
results = state["ansatz_optimizer_sweep_noiseless"]["results"]
print(f"run state: {state_path}\n")

rows = []
for opt in results:
    for name, entry in results[opt].items():
        params = list(entry["optimal_parameters"].values())
        circuit = build(name).decompose(reps=10).assign_parameters(params)
        circuit.measure_all()
        tqc = transpile(circuit, optimization_level=1)
        per_seed = {"top": [], "best": [], "vqe": []}
        for shot_seed in SHOT_SEEDS:
            sampler = AerSampler(seed=shot_seed, options={"backend_options": {"method": "statevector"}})
            sampler.options.default_shots = 2000
            counts = sampler.run([tqc], shots=2000).result()[0].data.meas.get_counts()
            scored = {bs: objective([int(c) for c in reversed(bs)]) for bs in counts}
            per_seed["top"].append(scored[max(counts, key=counts.get)])
            per_seed["best"].append(max(scored.values()))
            per_seed["vqe"].append(entry["readout"]["obj_vqe"])
        rows.append((f"{name} / {opt}", {k: (max(v) - min(v)) for k, v in per_seed.items()}))
        print(f"  {rows[-1][0]:<34} spread over shot seeds: "
              + "  ".join(f"{k}={v:7.3f}" for k, v in rows[-1][1].items()))

print(f"\n{'read-out rule':<10}{'median spread':>15}{'worst spread':>15}")
print("-" * 40)
for rule in ("top", "best", "vqe"):
    spreads = sorted(r[1][rule] for r in rows)
    print(f"{rule:<10}{spreads[len(spreads)//2]:>15.3f}{spreads[-1]:>15.3f}")

print(f"\nclassical optimum L* = {L_star:.6f}")
