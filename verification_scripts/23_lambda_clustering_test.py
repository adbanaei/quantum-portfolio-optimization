# Can annualization explain the lambda ordering? It provably can't -- confirm, then look further.
import numpy as np

# (return, volatility) pairs in daily units, as the notebook reported them
data = {
    "lambda=0":    (0.005679, 0.071966),
    "lambda=1":    (0.001531, 0.022409),
    "lambda=10":   (0.001116, 0.016607),
    "lambda=100":  (0.001779, 0.025233),
    "lambda=1000": (0.001286, 0.019360),
    "lambda=10000":(0.001611, 0.021863),
    "classical":   (0.002093, 0.028465),
}

labels = list(data.keys())
rets_daily = np.array([data[l][0] for l in labels])
vols_daily = np.array([data[l][1] for l in labels])
order_daily_ret = np.argsort(rets_daily)
order_ann_ret = np.argsort(rets_daily * 252)
order_daily_vol = np.argsort(vols_daily)
order_ann_vol = np.argsort(vols_daily * np.sqrt(252))
print("Return ranking, daily :", [labels[i] for i in order_daily_ret])
print("Return ranking, annualized (x252):", [labels[i] for i in order_ann_ret])
print("-> identical ranking:", [labels[i] for i in order_daily_ret] == [labels[i] for i in order_ann_ret])
print()
print("Volatility ranking, daily:", [labels[i] for i in order_daily_vol])
print("Volatility ranking, annualized (x sqrt252):", [labels[i] for i in order_ann_vol])
print("-> identical ranking:", [labels[i] for i in order_daily_vol] == [labels[i] for i in order_ann_vol])

print("\n--- each lambda relative to classical, in our own data ---")
r_c, v_c = data["classical"]
for l in labels:
    r, v = data[l]
    print(f"  {l:12s}: return={r:.6f} ({100*r/r_c:.0f}% of classical)   "
          f"volatility={v:.6f} ({100*v/v_c:.0f}% of classical)")
