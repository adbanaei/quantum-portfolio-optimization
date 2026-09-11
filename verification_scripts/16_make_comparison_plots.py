import pickle, numpy as np
import matplotlib.pyplot as plt

with open("raw_dollar_vqe_results.pkl", "rb") as f:
    saved = pickle.load(f)
results, offset_raw = saved["results"], saved["offset"]
classical_gs_raw = -2.24e8  # script 14's brute-force result

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"COBYLA": "#1f77b4", "SPSA": "#ff7f0e", "NFT": "#2ca02c"}
for name, hist in results.items():
    evals, vals = zip(*hist)
    ax.plot(evals, vals, label=name, color=colors[name], linewidth=1.3)
ax.axhline(classical_gs_raw, color="black", linestyle=":", linewidth=2,
           label="Classical optimum (Ising energy, raw-dollar hypothesis)")
ax.set_xlabel("Evaluations (\"Epochs\")")
ax.set_ylabel("Energy (Hamiltonian eigenvalue)")
ax.set_title("VQE convergence under the RAW-DOLLAR (non-budget-normalized) Hamiltonian\n"
             "TwoLocal linear, reps=3, 2000 shots, λ=10 — for scale comparison to paper's Fig. 2 only")
ax.legend()
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
fig.tight_layout()
fig.savefig("raw_dollar_convergence_comparison.png", dpi=140)
print("saved raw_dollar_convergence_comparison.png")
print(f"y-axis range: {min(v for h in results.values() for _,v in h):.2e} to "
      f"{max(v for h in results.values() for _,v in h):.2e}")
print(f"paper Fig 2 y-axis range: roughly -1e8 to +6e8")

# Fig 4: does annualizing put our numbers on the paper's axes?
# (return, volatility) pairs as the notebook itself reported them, daily units
print("\n--- annualization check ---")
data = {
    "lambda=0":    (0.005679, 0.071966),
    "lambda=1":    (0.001531, 0.022409),
    "lambda=10":   (0.001116, 0.016607),
    "lambda=100":  (0.001779, 0.025233),
    "lambda=1000": (0.001286, 0.019360),
    "lambda=10000":(0.001611, 0.021863),
    "classical":   (0.002093, 0.028465),
}
T_TRADING, T_CALENDAR = 252, 365
print(f"{'label':<14}{'daily ret':>10}{'daily vol':>10}{'ann.ret(252d)':>15}{'ann.vol(sqrt252)':>18}")
for label, (r, v) in data.items():
    print(f"{label:<14}{r:>10.6f}{v:>10.6f}{r*T_TRADING:>15.4f}{v*np.sqrt(T_TRADING):>18.4f}")
print(f"\npaper's Fig 4 axis ranges: Expected Return 0.2-1.2, Expected Volatility 0.25-0.55")

# comparison plot: annualized reproduction points, styled like lambda_frontier.png
fig2, ax2 = plt.subplots(figsize=(7, 6))
markers = {"lambda=0": ("*", "brown", 300), "lambda=1": ("p", "limegreen", 220),
           "lambda=10": ("o", "darkgreen", 220), "lambda=100": ("^", "orange", 220),
           "lambda=1000": ("D", "deeppink", 200), "lambda=10000": ("h", "magenta", 220),
           "classical": ("s", "black", 260)}
for label, (r, v) in data.items():
    m, c, s = markers[label]
    ax2.scatter(v*np.sqrt(T_TRADING), r*T_TRADING, marker=m, color=c, s=s,
                edgecolor="black", label=label, zorder=3)
ax2.set_xlabel("Annualized Expected Volatility (σ√252)")
ax2.set_ylabel("Annualized Expected Return (μ×252)")
ax2.set_title("This notebook's own λ-sweep results, annualized —\nfor scale comparison to paper's Fig. 4 only (not a re-optimization)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("annualized_lambda_frontier_comparison.png", dpi=140)
print("\nsaved annualized_lambda_frontier_comparison.png")
