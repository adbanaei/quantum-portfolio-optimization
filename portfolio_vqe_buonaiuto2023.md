<div dir="rtl" class="farsi-text">

# بهینه‌سازی پورتفولیو با الگوریتم VQE

پیاده‌سازی مقاله:

<div dir="ltr">

> **Buonaiuto, G., Gargiulo, F., De Pietro, G., Esposito, M., & Pota, M. (2023).**<br>
> *Best practices for portfolio optimization by quantum computing, experimented on real quantum devices.*<br>
> Scientific Reports, 13, 19434. https://doi.org/10.1038/s41598-023-45392-w

</div>
</div>

<div dir="rtl" class="farsi-text">

## ۱. تنظیمات اولیه

این بخش تمام پارامترهای قابل‌تنظیم نوت‌بوک را در یک‌جا جمع می‌کند. چند نکته درباره‌ی مقادیر پیش‌فرض:

- `VQE_SHOTS=2000` و `DEFAULT_REPS=3` مستقیما از بخش «تنظیمات تجربی» مقاله گرفته شده‌اند.
- دو بودجه‌ی تکرار جداگانه وجود دارد: `DEFAULT_MAXITER` (برای اجرای مقدماتی بخش ۹) و `ANSATZ_MAXITER` (برای مقایسه‌ی سیستماتیک ۹×۳ بخش‌های ۱۰ و ۱۱) — این دو عمدا متفاوت‌اند، دلیل در بخش‌های مربوطه.
- `END_DATE` مطابق قرارداد خودِ `yfinance` **بازه‌ی باز** است؛ یعنی سری قیمت در آخرین روز معاملاتی *پیش از* این تاریخ تمام می‌شود و تاریخ پایانِ اعلام‌شده‌ی مقاله خودش جزو داده نیست.
- `BUDGET_TOL` و `SOFT_MATCH_TOL` آستانه‌های گزارش‌اند (برآورده‌شدن قید بودجه، و نزدیکی به جواب کلاسیک در فضای بازده/نوسان) و در یک‌جا نگه داشته شده‌اند تا جدول‌ها و گزارش یک عدد واحد را نقل کنند.
- فلگ‌های `GET_PRICE_DATA`/`SHOW_PLOTS`/`SAVE_PLOTS`/`INCLUDE_FULL_HISTORIES` رفتار I/O نوت‌بوک را کنترل می‌کنند، و `RUN_TOPOLOGY_DIAGNOSTIC` ارزیابی توپولوژی بخش ۱۲.۱ را روشن/خاموش می‌کند.

</div>


```python
# --- Configuration ---
GET_PRICE_DATA = False
SHOW_PLOTS = False
SAVE_PLOTS = True
INCLUDE_FULL_HISTORIES = True   # -> True to embed full convergence traces + optimal_parameters
RUN_TOPOLOGY_DIAGNOSTIC = True  # transpile-only audit of the SWAP cost the noise model omits
USE_PYPI_MIRROR = False         # local runs only: route pip through a regional mirror
PYPI_MIRROR = "https://package-mirror.liara.ir/repository/pypi/simple"

VQE_SHOTS = 2000
DEFAULT_REPS = 3
DEFAULT_ENTANGLEMENT = "linear"
DEFAULT_MAXITER = 200
ANSATZ_MAXITER = 100
DEFAULT_PROGRESS_EVERY = 10
ANSATZ_PROGRESS_EVERY = 25
N_RANDOM_PORTFOLIOS = 30000

# Read-out tolerances, kept here so the tables and the report quote one number.
BUDGET_TOL = 0.15      # |P''^T b - 1| still counted as "budget constraint satisfied"
SOFT_MATCH_TOL = 0.10  # relative (return, volatility) distance still counted as "reaches classical"

# ---- Parameters from the paper (Section 2.1) ----
TICKERS   = ["AAPL", "IBM", "NFLX", "TSLA"]
START_DATE = "2011-12-23"
END_DATE   = "2022-10-21"   # exclusive, per the note above

DIV_CUTOFF_DATE = "2023-02-18"  # paper's "Received" date -- proxy for authors' download date
MANUAL_RESIDUAL_CORRECTION = {
    "AAPL": 1.0,
    "IBM":  1.0,
    "NFLX": 1.0,
    "TSLA": 1.0,
}

BUDGET     = 2000          # B = 2000 (Section: Experimental settings)
RISK_Q     = 0.5           # risk aversion q = 0.5
LAMBDA     = 10.0          # penalty coefficient λ (best value from paper, Figure 4)
RANDOM_SEED = 42

OPTIMIZER_ORDER = ["Cobyla", "SPSA", "NFT"]

OPTIMIZER_ORDER_NOISY = [
    "Cobyla",
    "SPSA",
    "NFT"
]


try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    HOME_DIR = "/content/drive/MyDrive/quantum-portfolio-optimization"
else:
    HOME_DIR = "."

PRICE_DATA_DIR = f"{HOME_DIR}/price_data"
CACHE_DIR = f"{HOME_DIR}/.cache_dir"
PLOT_DIR = f"{HOME_DIR}/plots"
STATE_DIR = f"{HOME_DIR}/run_states"
```

    Mounted at /content/drive


<div dir="rtl" class="farsi-text">

## ۲. ایمپورت کتابخانه‌ها

کتابخانه‌های Qiskit (نسخه‌ی ≥ ۱.۰)، Qiskit Aer (شبیه‌ساز)، Qiskit Optimization (برای `QuadraticProgram`)، و ابزارهای استاندارد علمی پایتون (numpy/pandas/matplotlib) ایمپورت می‌شوند. تشخیص خودکار محیط Colab (`IN_COLAB`) در ادامه‌ی نوت‌بوک برای انتخاب بین GPU (در Colab) و CPU (اجرای محلی) در ساخت samplerها استفاده می‌شود.

چند نکته درباره‌ی نصب و پیکربندی این سلول:

- **نسخه‌های پین‌شده:** نسخه‌ها دقیقا همان‌هایی هستند که نتایج cache‌شده‌ی `run_states/` زیر آن‌ها تولید شده‌اند. افزون بر این، `TwoLocal`، `RealAmplitudes` و `PauliTwoDesign` در Qiskit 2.0 حذف شده‌اند، پس نصب بدون پین، ایمپورت‌های همین سلول را مشکل‌دار می‌کند.
- **نصب Aer:** بسته‌ی `qiskit-aer-gpu` روی نسخه‌ی ۰.۱۵.۱ متوقف شده (برای CPython بالاتر از ۳.۱۳ فایل wheel ندارد) و `qiskit-aer-gpu-cu11` نسخه‌ی نگه‌داری‌شده‌ی GPU است. به همین دلیل کاندیداها به ترتیب امتحان می‌شوند و در صورت به مشکل خوردن یکی، به بعدی عقب‌نشینی می‌شود، به‌جای آنکه کل نصب متوقف شود.
- **هشدارها:** فیلتر `PendingDeprecationWarning` عمدا محدود به همان سه خانواده‌ی ansatz است (که در هر بار ساخت هشدار می‌دهند) تا هشدار نامرتبط دیگری پنهان نشود.
- **دستگاه شبیه‌سازی:** `resolve_aer_backend` دستگاه و روش واقعا در دسترس را تعیین می‌کند (درخواست GPU از Aer در نبودِ آن، چند سلول بعد و درون اولین `sampler.run()` به مشکل می‌خورد). روی CPU و زیر مدل نویز کامل، `matrix_product_state` حدود یک مرتبه‌ی بزرگی سریع‌تر از statevector متراکم است.
- **`CACHE_REFERENCE_BACKEND`:** گزینه‌های sampler بخشی از کلید cache هستند (بخش ۹.۲)، پس تغییر دستگاه یا روش شبیه‌سازی یعنی محاسبه‌ی دوباره‌ی همه‌ی پیکربندی‌های cache‌شده؛ این ثابت همان دستگاهی را نگه می‌دارد که نتایج موجود با آن تولید شده‌اند تا این وضعیت هشدار داده شود.

</div>


```python
# Version pins, for the reasons given above.
import subprocess
import sys

CORE_PINS = [
    "qiskit==1.4.6",
    "qiskit-algorithms==0.4.0",
    "qiskit-optimization==0.7.0",
    "qiskit-ibm-runtime==0.29.0",
]
SUPPORT_PINS = ["yfinance==0.2.66", "joblib==1.5.3", "pyyaml==6.0.3",
                "numpy", "matplotlib"]
# Aer builds tried in order, falling back rather than aborting the resolution.
AER_CANDIDATES = (["qiskit-aer-gpu==0.15.1", "qiskit-aer-gpu",
                   "qiskit-aer-gpu-cu11", "qiskit-aer"] if IN_COLAB
                  else ["qiskit-aer==0.15.1", "qiskit-aer"])


def pip_install(packages):
    """Install via the running interpreter. Returns the CompletedProcess: pip
    reports failures on stdout, so callers must check returncode, not stderr."""
    extra = ["--no-cache-dir"] if IN_COLAB else []
    if not IN_COLAB and USE_PYPI_MIRROR:
        extra += ["-i", PYPI_MIRROR]
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *extra, *packages],
        capture_output=True, text=True,
    )


def _report(result):
    tail = "\n".join(filter(None, [result.stdout[-3000:], result.stderr[-3000:]]))
    print(tail or "(pip produced no output)")


print(f"Python {sys.version.split()[0]}  |  Colab: {IN_COLAB}")

_result = pip_install(CORE_PINS + SUPPORT_PINS)
if _result.returncode != 0:
    _report(_result)
    raise RuntimeError("pip install of the pinned core packages failed -- see output above.")

for _candidate in AER_CANDIDATES:
    _result = pip_install([_candidate])
    if _result.returncode == 0:
        AER_INSTALLED = _candidate
        break
else:
    _report(_result)
    raise RuntimeError("could not install any qiskit-aer build -- see output above.")

if AER_INSTALLED != AER_CANDIDATES[0]:
    print(f"[warn] {AER_CANDIDATES[0]} has no wheel for this interpreter; "
          f"installed '{AER_INSTALLED}' instead.")
    print("       Cached VQE results stay valid (the Aer version is not part of the "
          "cache key),\n       but any configuration recomputed now will differ in its "
          "shot RNG stream.")

# --- Standard libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from joblib import Memory
import yaml
import json
import hashlib
import warnings
import itertools
import time
import datetime
import platform
import math
import glob
import os
from importlib import metadata as importlib_metadata


warnings.filterwarnings("ignore", category=DeprecationWarning)
# Targeted, so an unrelated pending deprecation still surfaces.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        message=r".*(TwoLocal|RealAmplitudes|PauliTwoDesign|NLocal).*")
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.size'] = 11
if SAVE_PLOTS:
    os.makedirs(PLOT_DIR, exist_ok=True)


# --- Qiskit core (>=1.0, <2.0) ---
from qiskit import transpile
from qiskit.circuit.library import TwoLocal, RealAmplitudes, PauliTwoDesign

# --- Qiskit Aer (noise simulation) ---
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel

# --- Qiskit Algorithms (community package) ---
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NFT
from qiskit_algorithms.utils import algorithm_globals

# --- Qiskit Optimization ---
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising


def resolve_aer_backend():
    """Device and simulation method actually available. Asking Aer for a GPU it
    does not have fails inside the first sampler.run(), several cells later."""
    from qiskit_aer import AerSimulator
    try:
        devices = tuple(AerSimulator().available_devices())
    except Exception:
        devices = ("CPU",)
    if "GPU" in devices:
        return "GPU", "statevector", devices
    # Under a full noise model on CPU, dense statevector is roughly an order of magnitude slower.
    return "CPU", "matrix_product_state", devices


AER_DEVICE, AER_NOISY_METHOD, AER_AVAILABLE_DEVICES = resolve_aer_backend()

# The backend the cached results were produced on; see the note above.
CACHE_REFERENCE_BACKEND = ("GPU", "statevector")


def _installed(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not installed"


print(f"\nAer devices available: {AER_AVAILABLE_DEVICES}")
print(f"Using device={AER_DEVICE}, noisy-simulation method={AER_NOISY_METHOD}")
if (AER_DEVICE, AER_NOISY_METHOD) != CACHE_REFERENCE_BACKEND:
    print(f"[warn] cached results used device={CACHE_REFERENCE_BACKEND[0]}, "
          f"method={CACHE_REFERENCE_BACKEND[1]}; every configuration will recompute.")

print("\nAll imports successful. Resolved versions:")
for _pkg in ["qiskit", "qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11", "qiskit-algorithms",
             "qiskit-optimization", "qiskit-ibm-runtime", "numpy", "pandas", "joblib"]:
    _v = _installed(_pkg)
    if _v != "not installed":
        print(f"  {_pkg:<22} {_v}")
```

    Python 3.13.15  |  Colab: True
    [warn] qiskit-aer-gpu==0.15.1 has no wheel for this interpreter; installed 'qiskit-aer-gpu-cu11' instead.
           Cached VQE results stay valid (the Aer version is not part of the cache key),
           but any configuration recomputed now will differ in its shot RNG stream.
    
    Aer devices available: ('CPU', 'GPU')
    Using device=GPU, noisy-simulation method=statevector
    
    All imports successful. Resolved versions:
      qiskit                 1.4.6
      qiskit-aer-gpu-cu11    0.17.2
      qiskit-algorithms      0.4.0
      qiskit-optimization    0.7.0
      qiskit-ibm-runtime     0.29.0
      numpy                  2.1.3
      pandas                 2.2.3
      joblib                 1.5.3


<div dir="rtl" class="farsi-text">

## ۳. دریافت داده‌های مالی

مطابق بخش ۲.۱ مقاله: Apple (AAPL)، IBM، Netflix (NFLX)، Tesla (TSLA)  
بازه زمانی: 2011-12-23 تا 2022-10-21 (همان بازه مقاله)

</div>

<div dir="rtl" class="farsi-text">

### ۳.۱. داده‌ی مرجع مقاله (جدول ۱)

مقاله تنها بخشی از جدول ۱ خود را به‌صورت متنی گزارش کرده — همین دوازده تاریخ (۲۳ دسامبر ۲۰۱۶ تا ۱۱ ژانویه‌ی ۲۰۱۷) برای هر چهار دارایی — که در ادامه برای راستی‌آزمایی داده‌ی تصحیح‌شده به کار می‌رود.

</div>


```python
# Table 1 as published in Buonaiuto et al. (2023).
PAPER_TABLE1 = {
    "AAPL": {
        "2016-12-23": 27.219765, "2016-12-27": 27.392632,
        "2016-12-28": 27.275822, "2016-12-29": 27.268820,
        "2016-12-30": 27.056236, "2017-01-03": 27.133329,
        "2017-01-04": 27.102961, "2017-01-05": 27.240784,
        "2017-01-06": 27.544472, "2017-01-09": 27.796768,
        "2017-01-10": 27.824797, "2017-01-11": 27.974308,
    },
    "IBM": {
        "2016-12-23": 119.262428, "2016-12-27": 119.570061,
        "2016-12-28": 118.890434, "2016-12-29": 119.183701,
        "2016-12-30": 118.747368, "2017-01-03": 119.605820,
        "2017-01-04": 121.086693, "2017-01-05": 120.686058,
        "2017-01-06": 121.279861, "2017-01-09": 119.934898,
        "2017-01-10": 118.411110, "2017-01-11": 120.006439,
    },
    "NFLX": {
        "2016-12-23": 125.589996, "2016-12-27": 128.350006,
        "2016-12-28": 125.889999, "2016-12-29": 125.330002,
        "2016-12-30": 123.800003, "2017-01-03": 127.489998,
        "2017-01-04": 129.410004, "2017-01-05": 131.809998,
        "2017-01-06": 131.070007, "2017-01-09": 130.949997,
        "2017-01-10": 129.889999, "2017-01-11": 130.500000,
    },
    "TSLA": {
        "2016-12-23": 14.222667, "2016-12-27": 14.635333,
        "2016-12-28": 14.649333, "2016-12-29": 14.312000,
        "2016-12-30": 14.246000, "2017-01-03": 14.466000,
        "2017-01-04": 15.132667, "2017-01-05": 15.116667,
        "2017-01-06": 15.267333, "2017-01-09": 15.418667,
        "2017-01-10": 15.324667, "2017-01-11": 15.315333,
    },
}
```

<div dir="rtl" class="farsi-text">

### ۳.۲. دریافت و تصحیح قیمت‌ها

قیمت‌ها از `yfinance` دریافت و در سه مرحله تصحیح می‌شوند:

1. واگرد تقسیم‌های سهام رخ‌داده پس از تاریخ پایان بازه‌ی مقاله.
2. واگرد سودهای سهامی پرداخت‌شده بین آن تاریخ و *لحظه‌ی دریافت داده*.
3. اصلاح معیار برش مرحله‌ی قبل: چون `yfinance` نسبت به لحظه‌ی درخواست تعدیل می‌کند نه به تاریخ پایان بازه، و مقاله ماه‌ها بعد از پایان بازه (۱۸ فوریه‌ی ۲۰۲۳) دریافت شده، معیار برش باید تاریخ دریافت مقاله باشد، نه تاریخ پایان بازه.

جزئیات کامل و اعداد راستی‌آزمایی در گزارش (بخش «داده‌های مالی متفاوت») آمده است.

</div>


```python
def get_split_calendar(ticker):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    splits = tk.splits
    return [{"date": str(idx.date()), "ratio": float(val)} for idx, val in splits.items()]


def compute_split_correction(split_calendar, end_date_str):
    end_date = pd.Timestamp(end_date_str)
    post_end = [s for s in split_calendar if pd.Timestamp(s["date"]) > end_date]
    within = [s for s in split_calendar if pd.Timestamp(s["date"]) <= end_date]
    factor = 1.0
    for s in post_end:
        factor *= s["ratio"]
    return factor, post_end, within


def download_raw(ticker):
    import yfinance as yf
    # end= is exclusive: the series stops on the last trading day before END_DATE.
    hist = yf.download(ticker, start=START_DATE, end=END_DATE,
                        auto_adjust=False, progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


def build_dividend_correction_series(ticker, price_index, end_date_str):
    """
    Reconstruct, for every date in price_index, the multiplicative factor
    that undoes ONLY the excess dividends (those dated after end_date_str)
    that today's yfinance auto_adjust bakes in but the paper's authors
    never saw. Verified exact (matches yfinance's own Adj Close/Close
    ratio to 7 significant figures) against real post-End_Date price data.
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)

    divs = tk.dividends
    if divs.index.tz is not None:
        divs.index = divs.index.tz_localize(None)

    # No end date: today's auto_adjust has folded in every dividend up to *now*.
    raw_unadj = yf.download(ticker, start=START_DATE,
                             auto_adjust=False, progress=False)
    if isinstance(raw_unadj.columns, pd.MultiIndex):
        raw_unadj.columns = raw_unadj.columns.get_level_values(0)
    raw_unadj.index = raw_unadj.index.tz_localize(None) if raw_unadj.index.tz else raw_unadj.index

    end_date = pd.Timestamp(end_date_str)
    excess_divs = divs[divs.index > end_date]

    excess_factor = 1.0
    factor_log = []
    for ex_date, div_amt in excess_divs.items():
        prior_dates = raw_unadj.index[raw_unadj.index < ex_date]
        if len(prior_dates) == 0:
            continue
        prior_close = float(raw_unadj.loc[prior_dates.max(), "Close"])
        step = 1.0 - (div_amt / prior_close)
        excess_factor *= step
        factor_log.append({"ex_date": str(ex_date.date()), "div": float(div_amt),
                            "prior_close": prior_close, "step_factor": step})

    correction = 1.0 / excess_factor if excess_factor != 0 else 1.0
    correction_series = pd.Series(correction, index=price_index)

    return correction_series, excess_factor, factor_log


def fetch_and_fix_prices():
    os.makedirs(PRICE_DATA_DIR, exist_ok=True)
    summary = {"generated_at": datetime.datetime.now().isoformat(),
               "paper_end_date": END_DATE, "div_cutoff_date": DIV_CUTOFF_DATE,
               "tickers": {}}

    for t in TICKERS:
        print(f"\n{'='*70}\n{t}\n{'='*70}")

        split_calendar = get_split_calendar(t)
        split_factor, post_end_splits, within_splits = compute_split_correction(
            split_calendar, DIV_CUTOFF_DATE)
        print(f"Split correction factor: x{split_factor}  "
              f"(post-cutoff splits: {post_end_splits or 'none'})")

        print("Downloading raw (auto_adjust=False) history...")
        raw = download_raw(t)

        print("Computing dividend-adjustment excess correction...")
        div_correction_series, excess_factor, factor_log = build_dividend_correction_series(
            t, raw.index, DIV_CUTOFF_DATE)
        print(f"Dividend excess factor (today's over-adjustment): {excess_factor:.8f}")
        print(f"Dividend correction (undo factor): {1/excess_factor:.8f}"
              if excess_factor != 0 else "N/A")
        if factor_log:
            print(f"  {len(factor_log)} post-cutoff dividends folded in by "
                  f"today's auto_adjust, now being undone:")
            for f in factor_log[:3]:
                print(f"    {f['ex_date']}: div={f['div']}, "
                      f"prior_close={f['prior_close']:.2f}, step={f['step_factor']:.6f}")
            if len(factor_log) > 3:
                print(f"    ... and {len(factor_log)-3} more")

        manual_factor = MANUAL_RESIDUAL_CORRECTION.get(t, 1.0)
        print(f"Manual residual correction factor (last-resort, flat-gap fix): "
              f"x{manual_factor:.8f}")

        # raw Close -> split correction -> excess-dividend undo -> manual residual.
        fixed = raw.copy()
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"]
                      if c in fixed.columns]
        fixed[price_cols] = fixed[price_cols] * split_factor

        adj_col = "Adj Close" if "Adj Close" in fixed.columns else None
        if adj_col:
            fixed["Corrected_Adj_Close"] = (
                fixed[adj_col]
                * div_correction_series.reindex(fixed.index)
                * manual_factor
            )

        fixed_path = os.path.join(PRICE_DATA_DIR, f"{t}.csv")
        fixed.to_csv(fixed_path)
        print(f"Saved -> {fixed_path}")

        # Verify against Table 1
        print("\nVerifying against paper's Table 1 (using Corrected_Adj_Close):")
        checks = []
        for d, expected in PAPER_TABLE1[t].items():
            ts = pd.Timestamp(d)
            if ts in fixed.index and "Corrected_Adj_Close" in fixed.columns:
                actual = float(fixed.loc[ts, "Corrected_Adj_Close"])
                pct_diff = abs(actual - expected) / expected * 100
                status = "PASS" if pct_diff < 0.5 else "FAIL"
                print(f"  {d}: actual={actual:.6f}  paper={expected}  "
                      f"diff={pct_diff:.4f}%  [{status}]")
                checks.append({"date": d, "actual": actual, "expected": expected,
                                "pct_diff": pct_diff, "passed": pct_diff < 0.5})

        summary["tickers"][t] = {
            "fixed_csv": fixed_path,
            "split_correction_factor": split_factor,
            "post_end_date_splits": post_end_splits,
            "dividend_excess_factor": excess_factor,
            "dividend_undo_factor": 1 / excess_factor if excess_factor != 0 else None,
            "post_end_date_dividends_count": len(factor_log),
            "manual_residual_correction_factor": manual_factor,
            "table1_verification": checks,
            "all_checks_passed": all(c["passed"] for c in checks) if checks else None,
        }

    summary_path = os.path.join(PRICE_DATA_DIR, "correction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
```

<div dir="rtl" class="farsi-text">

### ۳.۳. بارگذاری داده‌ی تصحیح‌شده

فایل‌های CSV تصحیح‌شده (خروجی تابع بالا، در `PRICE_DATA_DIR`) خوانده و به بازه‌ی زمانی مقاله (`START_DATE` تا `END_DATE`، با همان قرارداد end-exclusive خودِ `yfinance`) محدود می‌شوند.

</div>


```python
if GET_PRICE_DATA:
    fetch_and_fix_prices()

algorithm_globals.random_seed = RANDOM_SEED
np.random.seed(RANDOM_SEED)

print(f"Loading offline price data for: {TICKERS}")
price_series = {}
for t in TICKERS:
    path = f"{PRICE_DATA_DIR}/{t}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Corrected_Adj_Close" not in df.columns:
        raise ValueError(f"{path} is missing a 'Corrected_Adj_Close' column")
    s = df["Corrected_Adj_Close"].copy()
    s.name = t
    price_series[t] = s

prices_df = pd.concat(price_series.values(), axis=1)

# Same window as the original yf.download(start=, end=), end= exclusive.
prices_df = prices_df.loc[
    (prices_df.index >= START_DATE) & (prices_df.index < END_DATE)
]
prices_df = prices_df.sort_index().dropna()

print(f"\nData shape: {prices_df.shape}  ({len(prices_df)} trading days)")
print(f"Window: {prices_df.index[0].date()} .. {prices_df.index[-1].date()}  "
      f"(END_DATE={END_DATE} exclusive)")
print("Price column: 'Corrected_Adj_Close' -- source of P, mu, Sigma and the qubit count")
print("\nLast 5 rows:")
prices_df.tail()
```

    Loading offline price data for: ['AAPL', 'IBM', 'NFLX', 'TSLA']
    
    Data shape: (2724, 4)  (2724 trading days)
    Window: 2011-12-23 .. 2022-10-20  (END_DATE=2022-10-21 exclusive)
    Price column: 'Corrected_Adj_Close' -- source of P, mu, Sigma and the qubit count
    
    Last 5 rows:






  <div id="df-d8b9659e-9df4-4a33-ba27-693a8bbc6e38" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>IBM</th>
      <th>NFLX</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-10-14</th>
      <td>137.930729</td>
      <td>117.362241</td>
      <td>230.000000</td>
      <td>204.990005</td>
    </tr>
    <tr>
      <th>2022-10-17</th>
      <td>141.947608</td>
      <td>118.809217</td>
      <td>245.100002</td>
      <td>219.350006</td>
    </tr>
    <tr>
      <th>2022-10-18</th>
      <td>143.283247</td>
      <td>120.197553</td>
      <td>240.860004</td>
      <td>220.190002</td>
    </tr>
    <tr>
      <th>2022-10-19</th>
      <td>143.392907</td>
      <td>119.777161</td>
      <td>272.380009</td>
      <td>222.039993</td>
    </tr>
    <tr>
      <th>2022-10-20</th>
      <td>142.924413</td>
      <td>125.437990</td>
      <td>268.160000</td>
      <td>207.279999</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d8b9659e-9df4-4a33-ba27-693a8bbc6e38')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d8b9659e-9df4-4a33-ba27-693a8bbc6e38 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d8b9659e-9df4-4a33-ba27-693a8bbc6e38');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




<div dir="rtl" class="farsi-text">

### ۳.۴. آمار مالی (معادلات ۱ تا ۴ مقاله)

از سری قیمت تصحیح‌شده، این کمیت‌ها محاسبه می‌شوند:

- بردار قیمت فعلی $P$ (معادله‌ی ۱)
- بازده‌ی روزانه (معادله‌ی ۲)
- میانگین بازده $\mu$ (معادله‌ی ۳)
- ماتریس کوواریانس $\Sigma$ (معادله‌ی ۴)

</div>


```python
# ---- Compute financial statistics (Equations 1-4 of the paper) ----

# Eq. 1: current prices P_i = p^T_i, read from the loaded series (last close).
P = prices_df.iloc[-1].values.astype(float)   # shape: (N,)
N = len(P)

# Eq. 2: Daily returns r^t_i = (p^t_i - p^{t-1}_i) / p^{t-1}_i
returns_df = prices_df.pct_change().dropna()

# Eq. 3: Expected (mean) return mu_i
mu = returns_df.mean().values        # shape: (N,)

# Eq. 4: Covariance matrix Sigma
Sigma = returns_df.cov().values      # shape: (N, N)

print("=" * 50)
print(f"Number of assets (N): {N}")
print(f"\nCurrent prices P (USD), as of {prices_df.index[-1].date()}:")
for t, p in zip(TICKERS, P):
    print(f"  {t:6s}: {p:.2f}")

print(f"\nExpected daily returns mu:")
for t, m in zip(TICKERS, mu):
    print(f"  {t:6s}: {m:.6f}")

print(f"\nCovariance matrix Sigma (diagonal = variance):")
print(np.round(Sigma, 8))
```

    ==================================================
    Number of assets (N): 4
    
    Current prices P (USD), as of 2022-10-20:
      AAPL  : 142.92
      IBM   : 125.44
      NFLX  : 268.16
      TSLA  : 207.28
    
    Expected daily returns mu:
      AAPL  : 0.001067
      IBM   : 0.000133
      NFLX  : 0.001687
      TSLA  : 0.002362
    
    Covariance matrix Sigma (diagonal = variance):
    [[3.29200e-04 1.00510e-04 1.64380e-04 2.30330e-04]
     [1.00510e-04 2.08690e-04 6.66200e-05 1.05310e-04]
     [1.64380e-04 6.66200e-05 9.88760e-04 3.31450e-04]
     [2.30330e-04 1.05310e-04 3.31450e-04 1.26766e-03]]


<div dir="rtl" class="farsi-text">

### ۳.۵. نمودار تاریخچه‌ی قیمت

صرفا یک بررسی بصری سریع از سری قیمت تصحیح‌شده‌ی هر چهار دارایی، پیش از ادامه‌ی محاسبات.

</div>


```python
# ---- Plot price history ----
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for ax, ticker, color in zip(axes.flatten(), TICKERS, colors):
    ax.plot(prices_df.index, prices_df[ticker], color=color, linewidth=0.8)
    ax.set_title(ticker, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

plt.suptitle("Asset Closing Prices (2011–2022)", fontsize=14, y=1.01)
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/price_history.png", bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)
```

<div dir="rtl" class="farsi-text">

## ۴. فرمول‌بندی مسئله Integer Portfolio Optimization

مطابق معادله (۷) مقاله:

<div dir="ltr">

$$\max_{n} \mathcal{L}(n) = \mu'^{T} n - q \cdot n^{T} \Sigma' n \quad \text{s.t.} \; P'^{T} n = 1$$

</div>

که در آن:

<div dir="ltr">

- $P' = P/B$ (قیمت‌های نرمالیزه)
- $\mu' = P' \circ \mu$ (بازده نرمالیزه)
- $\Sigma' = (P' \circ \Sigma)^T \circ P'$ (کوواریانس نرمالیزه)

</div>
</div>


```python
# ---- Integer formulation transformations (Equation 7) ----
P_prime  = P / BUDGET                              # P' = P/B
mu_prime = P_prime * mu                            # mu' = P' ∘ mu  (element-wise)
Sigma_prime = (P_prime[:, None] * Sigma) * P_prime[None, :]  # Sigma'_ij = P'_i * Sigma_ij * P'_j

print("Normalized prices P':")
for t, p in zip(TICKERS, P_prime):
    print(f"  {t:6s}: {p:.6f}")

print(f"\nNormalized returns mu':")
for t, m in zip(TICKERS, mu_prime):
    print(f"  {t:6s}: {m:.10f}")
```

    Normalized prices P':
      AAPL  : 0.071462
      IBM   : 0.062719
      NFLX  : 0.134080
      TSLA  : 0.103640
    
    Normalized returns mu':
      AAPL  : 0.0000762660
      IBM   : 0.0000083268
      NFLX  : 0.0002261675
      TSLA  : 0.0002448409


<div dir="rtl" class="farsi-text">

## ۵. کدگذاری باینری متغیرهای صحیح

مطابق معادلات (۸-۱۱) مقاله — این بخش تفاوت اصلی با پیاده‌سازی استاندارد Qiskit است:

<div dir="ltr">

$$n^{max}_i = \lfloor B/P_i \rfloor \quad (8)$$
$$d_i = \lfloor \log_2 n^{max}_i \rfloor \quad (9)$$
$$n_i = \sum_{j=0}^{d_i} 2^j b_{i,j} \quad (10)$$

</div>

ماتریس کدگذاری $C$ (معادله ۱۱) تعداد qubit های مورد نیاز را کاهش می‌دهد.

</div>


```python
# ---- Binary encoding of integer variables (Equations 8-11) ----

def build_binary_encoding(P, B):
    """
    Build binary encoding matrix C for integer portfolio variables.

    For each asset i:
      n_max_i = floor(B / P_i)          (Eq. 8)
      d_i     = floor(log2(n_max_i))    (Eq. 9)
      n_i     = sum_{j=0}^{d_i} 2^j * b_{i,j}  (Eq. 10)

    Returns:
      C      : (N, dim_b) encoding matrix
      n_max  : list of max units per asset
      d_list : list of d_i values
    """
    N = len(P)
    n_max  = [int(B / P[i]) for i in range(N)]
    d_list = [int(np.floor(np.log2(nm))) if nm > 0 else 0 for nm in n_max]
    dim_b  = sum(d + 1 for d in d_list)

    C = np.zeros((N, dim_b))
    col = 0
    for i in range(N):
        for j in range(d_list[i] + 1):
            C[i, col] = 2 ** j   # Eq. 11
            col += 1

    return C, n_max, d_list


def total_qubits_for_budget(P, B):
    """Qubit count Eqs. (8)-(9) would produce for a given budget, or None if the
    budget cannot buy a single unit of some asset."""
    units = [int(B / p) for p in P]
    if min(units) < 1:
        return None
    return sum(int(np.floor(np.log2(u))) + 1 for u in units)


C, n_max, d_list = build_binary_encoding(P, BUDGET)
dim_b = C.shape[1]  # total number of binary variables = number of qubits

print("=" * 55)
print(f"{'Asset':<8} {'Price':>9} {'n_max':>7} {'d_i':>5} {'bits':>6}")
print("-" * 55)
for i, t in enumerate(TICKERS):
    print(f"{t:<8} {P[i]:>9.2f} {n_max[i]:>7d} {d_list[i]:>5d} {d_list[i]+1:>6d}")
print("-" * 55)
print(f"{'Total qubits needed:':<35} {dim_b:>6d}")
print(f"\nEncoding matrix C  (shape {C.shape}):")
print(C)

# The paper reports 12 qubits for these same four assets at B = 2000.
PAPER_QUBIT_COUNT = 12
budgets_matching_paper = [B for B in range(1, 20001)
                          if total_qubits_for_budget(P, B) == PAPER_QUBIT_COUNT]
print(f"\nPaper reports {PAPER_QUBIT_COUNT} qubits for the same four assets at B={BUDGET}; "
      f"these prices give {dim_b}.")
if budgets_matching_paper:
    lo, hi = budgets_matching_paper[0], budgets_matching_paper[-1]
    contiguous = (hi - lo + 1) == len(budgets_matching_paper)
    print(f"  Budgets reproducing {PAPER_QUBIT_COUNT} qubits: B in [{lo}, {hi}]"
          f"{'' if contiguous else ' (non-contiguous)'}, i.e. about "
          f"{BUDGET / ((lo + hi) / 2):.1f}x smaller than the paper's stated budget.")
else:
    print(f"  No integer budget up to 20000 reproduces {PAPER_QUBIT_COUNT} qubits "
          f"with these four assets.")
```

    =======================================================
    Asset        Price   n_max   d_i   bits
    -------------------------------------------------------
    AAPL        142.92      13     3      4
    IBM         125.44      15     3      4
    NFLX        268.16       7     2      3
    TSLA        207.28       9     3      4
    -------------------------------------------------------
    Total qubits needed:                    15
    
    Encoding matrix C  (shape (4, 15)):
    [[1. 2. 4. 8. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 2. 4. 8. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 4. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 4. 8.]]
    
    Paper reports 12 qubits for the same four assets at B=2000; these prices give 15.
      Budgets reproducing 12 qubits: B in [1004, 1072], i.e. about 1.9x smaller than the paper's stated budget.


<div dir="rtl" class="farsi-text">

با اعمال ماتریس کدگذاری $C$ (تعریف‌شده در بخش قبل) روی $\mu'$، $\Sigma'$، و $P'$، ضرایب تبدیل‌شده به فضای باینری ($\mu''$، $\Sigma''$، $P''$ — معادله‌ی ۱۱ مقاله) به دست می‌آیند؛ این‌ها همان کمیت‌هایی هستند که فرمول‌بندی QUBO بخش بعد روی آن‌ها بنا می‌شود:

<div dir="ltr">

$$\mu'' = C^T \mu' \qquad \Sigma'' = C^T \Sigma' C \qquad P'' = C^T P'$$

</div>
</div>


```python
# ---- Apply encoding to get transformed coefficients (after Eq. 11) ----
mu_pp    = C.T @ mu_prime          # shape: (dim_b,)
Sigma_pp = C.T @ Sigma_prime @ C   # shape: (dim_b, dim_b)
P_pp     = C.T @ P_prime           # shape: (dim_b,)

print(f"mu''   shape: {mu_pp.shape}")
print(f"Sigma'' shape: {Sigma_pp.shape}")
print(f"P''    shape: {P_pp.shape}")
print(f"\nP'' (budget constraint vector):")
print(np.round(P_pp, 6))
```

    mu''   shape: (15,)
    Sigma'' shape: (15, 15)
    P''    shape: (15,)
    
    P'' (budget constraint vector):
    [0.071462 0.142924 0.285849 0.571698 0.062719 0.125438 0.250876 0.501752
     0.13408  0.26816  0.53632  0.10364  0.20728  0.41456  0.82912 ]


<div dir="rtl" class="farsi-text">

## ۶. فرمول‌بندی QUBO

مطابق معادله (۱۳) مقاله، با اضافه کردن جریمه به تابع هدف:

<div dir="ltr">

$$\max_b \mathcal{L}(b) = \mu''^T b - q \cdot b^T \Sigma'' b - \lambda (P''^T b - 1)^2$$

</div>

این را با `QuadraticProgram` از Qiskit پیاده می‌کنیم. چون `QuadraticProgram` مسئله را به‌صورت **کمینه‌سازی** می‌گیرد، تابع هدف بالا با علامت منفی نوشته و جمله‌ی جریمه بسط داده می‌شود. با استفاده از $b_i^2 = b_i$ (که برای متغیر باینری برقرار است)، بسط کامل جمله‌ی جریمه چنین است:

<div dir="ltr">

$$\lambda\,(P''^T b - 1)^2 = \lambda \sum_i P''^2_i b_i \;+\; 2\lambda \sum_{i<j} P''_i P''_j\, b_i b_j \;-\; 2\lambda \sum_i P''_i b_i \;+\; \lambda$$

</div>

با جمع‌کردن این جملات با تابع هدف، ضرایب نهایی مسئله‌ی کمینه‌سازی به دست می‌آید:

<div dir="ltr">

$$\text{linear}_i = -\mu''_i + q\,\Sigma''_{ii} + \lambda\,(P''^2_i - 2P''_i)$$
$$\text{quadratic}_{ij} = 2q\,\Sigma''_{ij} + 2\lambda\,P''_i P''_j \qquad (i<j)$$
$$\text{constant} = \lambda$$

</div>

جمله‌ی قطری $\Sigma''_{ii}$ به‌خاطر همان اتحاد $b_i^2 = b_i$ به بخش خطی منتقل می‌شود، و ثابت $\lambda$ روی جواب بهینه اثری ندارد ولی مقیاس انرژی را جابه‌جا می‌کند.

</div>


```python
# ---- Build QUBO (Equation 13) ----

def build_qubo(mu_pp, Sigma_pp, P_pp, q, lam):
    """
    Build QuadraticProgram for the QUBO formulation (Equation 13):

        max   mu''^T b  -  q * b^T Sigma'' b  -  lambda * (P''^T b - 1)^2
        s.t.  b_i in {0,1}

    The constraint is embedded as a penalty term, so the QP is truly unconstrained.
    Converted to minimization: min -L(b)
    """
    n = len(mu_pp)
    qp = QuadraticProgram(name="Portfolio_QUBO")

    # Add binary variables b_0, b_1, ..., b_{n-1}
    for i in range(n):
        qp.binary_var(name=f"b{i}")

    # ---- Linear coefficients (minimization form; see the derivation above) ----
    linear = {}
    for i in range(n):
        lin_i  = -mu_pp[i]
        lin_i += q * Sigma_pp[i, i]          # diagonal quadratic becomes linear
        lin_i += lam * (P_pp[i]**2 - 2*P_pp[i])
        linear[f"b{i}"] = lin_i

    # ---- Build quadratic coefficients ----
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            q_ij  = 2 * q * Sigma_pp[i, j]
            q_ij += 2 * lam * P_pp[i] * P_pp[j]
            if q_ij != 0.0:
                quadratic[(f"b{i}", f"b{j}")] = q_ij

    # Constant from penalty: lambda * 1 (independent of b, does not affect optimizer but shifts energy)
    constant = lam

    qp.minimize(linear=linear, quadratic=quadratic, constant=constant)
    return qp


qp = build_qubo(mu_pp, Sigma_pp, P_pp, q=RISK_Q, lam=LAMBDA)

print("QUBO formulation built successfully.")
print(f"  Number of binary variables: {qp.get_num_vars()}")
print(f"  Number of qubits required : {qp.get_num_vars()}")
```

    QUBO formulation built successfully.
      Number of binary variables: 15
      Number of qubits required : 15


<div dir="rtl" class="farsi-text">

## ۷. تبدیل به هامیلتونین پائولی (Ising Hamiltonian)

مطابق معادله (۱۶) مقاله با تبدیل
<div dir="ltr">

$$b_i \to \frac{1+s_i}{2}$$

</div>

این پیاده‌سازی از متد `to_ising` کتابخانه‌ی Qiskit استفاده می‌کند که قرارداد درونی

<div dir="ltr">

$$b_i\to(1-Z_i)/2$$

</div>

دارد — علامت مخالف با متن مقاله برای جمله‌های خطی، ولی معادل از نظر حالت پایه. جزئیات و راستی‌آزمایی نمادین در گزارش.

پیش از `to_ising`، مسئله از `QuadraticProgramToQubo` عبور داده می‌شود. این مبدل وظیفه‌اش تبدیل هر ساختار باقی‌مانده (قید یا متغیر غیرباینری) به فرم QUBO است؛ اما چون مسئله‌ی ما از پیش بدون قید است (جریمه در تابع هدف جاسازی شده)، عملا بدون تغییر از آن عبور می‌کند و تنها به‌عنوان یک لایه‌ی اطمینان در مسیر باقی مانده است.

</div>


```python
# ---- Convert QUBO to Ising Hamiltonian (Equation 16) ----
converter = QuadraticProgramToQubo()
qubo_converted = converter.convert(qp)

ising_op, offset = to_ising(qubo_converted)
ising_op = ising_op.simplify(atol=1e-12)

print(f"Ising Hamiltonian:")
print(f"  Number of qubits  : {ising_op.num_qubits}")
print(f"  Number of Pauli terms: {len(ising_op)}")
print(f"  Energy offset     : {offset:.6f}")
print(f"\nFirst 5 Pauli terms:")
for term in list(ising_op)[:5]:
    print(f"  {term}")
print(f"\nNote: VQE minimises Ising energy. Classical optimum in Ising space ≈ -L* - offset (computed after brute-force).")

```

    Ising Hamiltonian:
      Number of qubits  : 15
      Number of Pauli terms: 120
      Energy offset     : 20.843858
    
    First 5 Pauli terms:
      SparsePauliOp(['IIIIIIIIIIIIIIZ'],
                  coeffs=[-0.89535688+0.j])
      SparsePauliOp(['IIIIIIIIIIIIIZI'],
                  coeffs=[-1.79071375+0.j])
      SparsePauliOp(['IIIIIIIIIIIIZII'],
                  coeffs=[-3.58142751+0.j])
      SparsePauliOp(['IIIIIIIIIIIZIII'],
                  coeffs=[-7.16285502+0.j])
      SparsePauliOp(['IIIIIIIIIIZIIII'],
                  coeffs=[-0.78583491+0.j])
    
    Note: VQE minimises Ising energy. Classical optimum in Ising space ≈ -L* - offset (computed after brute-force).


<div dir="rtl" class="farsi-text">

## ۸. حل کلاسیک با NumPy (معیار مقایسه)

جستجوی brute-force روی تمام

<div dir="ltr">

$$2^{\text{dim}_b}$$

</div>

حالت برای مسائل کوچک — معادل branch-and-bound مقاله.

جواب کلاسیک، افزون بر فضای پورتفولیو، در فضای انرژی آیزینگ هم لازم است: `QuadraticProgram` مقدار $-\mathcal{L}$ را کمینه می‌کند و تابع هدف آن برابر $\langle H \rangle + \text{offset}$ است، پس تصویر همان بهینه در فضای انرژی برابر $E^* = -\mathcal{L}^* - \text{offset}$ است. این کمیت هم خط‌چین مرجع نمودارهای همگرایی است و هم مخرج نسبت کیفیت $E_{vqe}/E^*$ — همان معیاری که مقاله در شکل ۶ خود گزارش می‌کند.

</div>


```python
# ---- Classical brute-force solution (exact benchmark, Section 2.2 of paper) ----

def evaluate_portfolio(b_vec, mu_pp, Sigma_pp, P_pp, q, lam):
    """Evaluate the QUBO objective (Eq. 13) for a binary vector b."""
    b = np.array(b_vec, dtype=float)
    L = mu_pp @ b - q * (b @ Sigma_pp @ b) - lam * (P_pp @ b - 1)**2
    return L


def brute_force_optimal(mu_pp, Sigma_pp, P_pp, q, lam):
    """Enumerate all 2^dim_b binary vectors and find maximum."""
    n = len(mu_pp)
    best_val = -np.inf
    best_b   = None

    for bits in itertools.product([0, 1], repeat=n):
        val = evaluate_portfolio(bits, mu_pp, Sigma_pp, P_pp, q, lam)
        if val > best_val:
            best_val = val
            best_b   = np.array(bits, dtype=int)

    return best_b, best_val


def portfolio_return_risk_original(b_vec, C, mu, Sigma, P, BUDGET):
    """
    Compute portfolio expected return and volatility in original space.
    b_vec : binary allocation vector (dim_b,)
    Returns: (expected_return, volatility) — both dimensionless (fraction of budget)
    """
    n_units = C @ np.array(b_vec, dtype=float)   # integer units per asset
    x = n_units * P / BUDGET                     # investment fractions
    ret  = float(mu @ x)                         # portfolio expected return
    risk = float(x @ Sigma @ x)                  # portfolio variance
    return ret, np.sqrt(risk)                    # (return, volatility)


print(f"Searching over {2**dim_b} binary configurations (dim_b = {dim_b})...")
t0 = time.time()
b_opt_classical, val_opt_classical = brute_force_optimal(mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA)
t_classical = time.time() - t0

# Decode to number of units
n_opt = C @ b_opt_classical

print(f"\n{'='*50}")
print(f"CLASSICAL OPTIMAL SOLUTION  (time: {t_classical:.3f}s)")
print(f"{'='*50}")
print(f"Optimal binary vector b*  : {b_opt_classical}")
print(f"Optimal units per asset n*: {n_opt}")
for t, ni, pi in zip(TICKERS, n_opt, P):
    print(f"  {t:6s}: {int(ni)} units  x  ${pi:.2f}  =  ${int(ni)*pi:.2f}")
total_spend = sum(int(ni)*pi for ni,pi in zip(n_opt, P))
print(f"  Total investment: ${total_spend:.2f}  (Budget: ${BUDGET})")
print(f"\nObjective L(b*): {val_opt_classical:.8f}")

# Ising-space image of the same optimum: E* = -L* - offset (see the note above).
E_classical_ising = -val_opt_classical - offset
print(f"Classical optimum in Ising energy space E*: {E_classical_ising:.6f}")
```

    Searching over 32768 binary configurations (dim_b = 15)...
    
    ==================================================
    CLASSICAL OPTIMAL SOLUTION  (time: 0.452s)
    ==================================================
    Optimal binary vector b*  : [0 1 0 0 0 0 0 0 1 0 0 1 1 1 0]
    Optimal units per asset n*: [2. 0. 1. 7.]
      AAPL  : 2 units  x  $142.92  =  $285.85
      IBM   : 0 units  x  $125.44  =  $0.00
      NFLX  : 1 units  x  $268.16  =  $268.16
      TSLA  : 7 units  x  $207.28  =  $1450.96
      Total investment: $2004.97  (Budget: $2000)
    
    Objective L(b*): 0.00162574
    Classical optimum in Ising energy space E*: -20.845484


<div dir="rtl" class="farsi-text">

## ۹. حل با SamplingVQE روی شبیه‌ساز بدون نویز

پیاده‌سازی الگوریتم VQE مطابق بخش ۲.۴ مقاله با SamplingVQE از `qiskit_algorithms`

</div>

<div dir="rtl" class="farsi-text">

### ۹.۱. توابع پایه‌ی اجرا و رمزگشایی VQE

- `run_vqe`: یک اجرای VQE را با ansatz، بهینه‌ساز، و sampler داده‌شده انجام می‌دهد و تاریخچه‌ی همگرایی را برمی‌گرداند. این تاریخچه بر حسب *تعداد ارزیابی تابع هدف* است، نه تعداد تکرار بهینه‌ساز؛ و برای SPSA، اولین ۵۰ ارزیابی مرحله‌ی کالیبراسیون (`SPSA.calibrate`) است، نه گام‌های نزولی.
- `decode_vqe_result`: مدار بهینه‌شده را با پارامترهای نهایی نمونه‌برداری می‌کند و جواب را با **سه قاعده‌ی مختلف** بیرون می‌کشد، چون این سه با هم یکسان نیستند:
  - `top` — پرتکرارترین بیت‌استرینگ. شهودی است اما در این مقیاس قابل‌اتکا نیست: با ۲۰۰۰ شات روی ۱۵ کیوبیت، پرتکرارترین بیت‌استرینگ معمولا کمتر از ۲٪ شات‌ها را دارد و تنها با چند شمارش از نفر دوم جلوتر است، پس صرفا با تغییر seed شات عوض می‌شود.
  - `best` — بهترین بیت‌استرینگ در میان همین شات‌ها.
  - `vqe` — مقدار `SamplingVQEResult.best_measurement`، یعنی بهترین بیت‌استرینگ دیده‌شده در کل مسیر بهینه‌سازی. این همان چیزی است که `MinimumEigenOptimizer` خودِ Qiskit برمی‌گرداند و **جوابی است که باید به‌عنوان «جواب VQE» در نظر گرفته شود**.
- `obj_mean` میانگین وزن‌دار روی کل توزیع شات‌هاست و تصویرِ فضای‌پورتفولیویِ همان انرژی‌ای است که بهینه‌ساز کمینه می‌کند؛ بنابراین برای *رتبه‌بندی* پیکربندی‌ها همین کمیت درست است، نه `obj_top`.

</div>


```python
# ---- SamplingVQE: running a configuration, and reading a portfolio back out ----
SPSA_CALIBRATION_EVALS = 2 * 25   # 25 probe steps, 2 evaluations each (section 9.1)


def optimizer_calibration_evals(optimizer):
    """Leading objective evaluations that are calibration probes, not optimisation steps."""
    if isinstance(optimizer, SPSA):
        settings = optimizer.settings
        if settings.get("learning_rate") is None and settings.get("perturbation") is None:
            return SPSA_CALIBRATION_EVALS
    return 0


def _sampler_backend(sampler):
    backend = getattr(sampler, "_backend", None)
    return backend if backend is not None else getattr(sampler, "backend", None)


def _prepare_ansatz_for_sampler(ansatz, sampler):
    """Decompose and transpile an ansatz into the gate set its sampler's backend
    runs. This is what makes the noisy runs noisy: Aer attaches error channels by
    gate name, so an untranspiled circuit skips the single-qubit ones."""
    prepared = ansatz.decompose(reps=10)
    backend = _sampler_backend(sampler)
    if backend is not None:
        try:
            return transpile(prepared, backend=backend, optimization_level=1)
        except Exception:
            pass
    return transpile(prepared, optimization_level=1)


def run_vqe(
    hamiltonian,
    ansatz,
    optimizer,
    sampler,
    seed=42,
    label="VQE",
    progress_every=25,
    initial_point=None,
):
    """
    Run SamplingVQE and return result + convergence history.

    Returns:
        result      : SamplingVQEResult
        energy_hist : list of energy values per objective evaluation
        elapsed_s   : wall time in seconds
    """
    algorithm_globals.random_seed = seed
    energy_hist = []
    t0 = time.time()

    ansatz_prepared = _prepare_ansatz_for_sampler(ansatz, sampler)

    # Required, not drawn internally; see make_initial_point in section 9.2.
    assert initial_point is not None, (
        "initial_point must be precomputed before calling run_vqe "
        "(use make_initial_point() or go through run_vqe_cached())."
    )

    def callback(eval_count, params, value, meta):
        energy_hist.append(float(value))
        if progress_every and eval_count % progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"[{label}] eval={eval_count:4d} "
                f"energy={float(value): .6f} elapsed={elapsed:6.1f}s"
            )

    vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz_prepared,
        optimizer=optimizer,
        callback=callback,
        initial_point=initial_point,
    )

    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    elapsed = time.time() - t0

    # History counts objective evaluations, not optimizer iterations (section 9.1).
    n_calibration = optimizer_calibration_evals(optimizer)
    print(f"[{label}]  eigenvalue: {float(np.real(result.eigenvalue)):.6f}  "
          f"evals: {len(energy_hist)}"
          f"{f' (first {n_calibration} are calibration)' if n_calibration else ''}"
          f"  time: {elapsed:.1f}s")
    return result, energy_hist, elapsed


def _bits_from_bitstring(bitstring):
    """Qiskit bitstrings are little-endian: leftmost character is the highest qubit."""
    return np.array([int(c) for c in reversed(bitstring)], dtype=float)


def make_problem(mu_pp, Sigma_pp, P_pp, q, lam, C, b_reference, mu, Sigma, P, budget):
    """Bundle everything a decoded bitstring needs to be scored, so read-out call
    sites stay one line long even where lambda varies."""
    return {"mu_pp": mu_pp, "Sigma_pp": Sigma_pp, "P_pp": P_pp, "q": q, "lam": lam,
            "C": C, "b_reference": b_reference,
            "mu": mu, "Sigma": Sigma, "P": P, "budget": budget}


def decode_vqe_result(result, ansatz, sampler, problem, shots=VQE_SHOTS,
                      reference_lam=None):
    """Sample the optimised circuit and read a portfolio out of it under three
    rules: ``top`` (modal bitstring), ``best`` (best of this shot set) and ``vqe``
    (``SamplingVQEResult.best_measurement``). Section 9.1 above explains why they
    disagree and which to quote.

    ``obj_mean`` is the shot-weighted objective, i.e. the portfolio-space image of
    the energy being minimised. ``reference_lam`` additionally scores every
    read-out under a common penalty, so runs at different lambda stay comparable.
    """
    ansatz_decomposed = ansatz.decompose(reps=10)
    params = result.optimal_parameters
    param_values = list(params.values()) if isinstance(params, dict) else list(params)

    bound = ansatz_decomposed.assign_parameters(param_values)
    # Before transpiling, so the transpiler carries the mapping: clbit i still holds qubit i.
    bound.measure_all()
    backend = _sampler_backend(sampler)
    circuit = (transpile(bound, backend=backend, optimization_level=1)
               if backend is not None else transpile(bound, optimization_level=1))

    counts = sampler.run([circuit], shots=shots).result()[0].data.meas.get_counts()
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])

    bits = {bs: _bits_from_bitstring(bs) for bs in counts}
    objs = {bs: evaluate_portfolio(v, problem["mu_pp"], problem["Sigma_pp"],
                                   problem["P_pp"], problem["q"], problem["lam"])
            for bs, v in bits.items()}

    out = {"shots": total,
           "n_distinct": len(counts),
           "top_count": ordered[0][1],
           "runner_up_count": ordered[1][1] if len(ordered) > 1 else 0}

    selections = {"top": ordered[0][0], "best": max(objs, key=objs.get)}
    best_measurement = getattr(result, "best_measurement", None)
    selections["vqe"] = (best_measurement["bitstring"] if best_measurement is not None
                         else selections["best"])

    for tag, bitstring in selections.items():
        # best_measurement need not reappear in the shot set drawn just now.
        b_vec = bits.get(bitstring)
        if b_vec is None:
            b_vec = _bits_from_bitstring(bitstring)
        ret, vol = portfolio_return_risk_original(
            b_vec, problem["C"], problem["mu"], problem["Sigma"],
            problem["P"], problem["budget"])
        out[f"b_{tag}"] = b_vec
        out[f"n_{tag}"] = problem["C"] @ b_vec
        out[f"obj_{tag}"] = evaluate_portfolio(b_vec, problem["mu_pp"], problem["Sigma_pp"],
                                               problem["P_pp"], problem["q"], problem["lam"])
        out[f"budget_{tag}"] = float(problem["P_pp"] @ b_vec)
        out[f"ret_{tag}"], out[f"vol_{tag}"] = ret, vol
        out[f"match_{tag}"] = bool(np.allclose(b_vec, problem["b_reference"]))
        if reference_lam is not None:
            out[f"obj_ref_{tag}"] = evaluate_portfolio(
                b_vec, problem["mu_pp"], problem["Sigma_pp"], problem["P_pp"],
                problem["q"], reference_lam)

    # Shot-weighted averages over the full output distribution.
    weights = {bs: n / total for bs, n in counts.items()}
    rv = {bs: portfolio_return_risk_original(v, problem["C"], problem["mu"],
                                             problem["Sigma"], problem["P"], problem["budget"])
          for bs, v in bits.items()}
    out["obj_mean"] = sum(objs[bs] * w for bs, w in weights.items())
    out["budget_mean"] = sum(float(problem["P_pp"] @ bits[bs]) * w for bs, w in weights.items())
    out["ret_mean"] = sum(rv[bs][0] * w for bs, w in weights.items())
    out["vol_mean"] = sum(rv[bs][1] * w for bs, w in weights.items())
    if reference_lam is not None:
        out["obj_ref_mean"] = sum(
            evaluate_portfolio(bits[bs], problem["mu_pp"], problem["Sigma_pp"],
                               problem["P_pp"], problem["q"], reference_lam) * w
            for bs, w in weights.items())
    return out


def shade_calibration(ax, n_calibration, label="calibration"):
    """Mark the leading evaluations of an SPSA trace that are calibration probes."""
    if not n_calibration:
        return
    ax.axvspan(0, n_calibration, color="0.88", zorder=0)
    ax.text(n_calibration / 2, 0.015, label, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="0.35")


def print_run_table(rows, header="Configuration"):
    """Summarise a sweep. `rows` is (label, decode, energy, n_evals); decode=None
    marks a configuration that was not computed. E/E* is the paper's Fig. 6 ratio."""
    width = 112
    print("\n" + "=" * width)
    print(f"{header:<34} {'E_vqe':>9} {'E/E*':>6} {'obj(vqe)':>10} {'obj(mean)':>11} "
          f"{'obj(top)':>10} {'match':>6} {'top%':>6} {'evals':>6}")
    print("-" * width)
    for label, dec, energy, n_evals in rows:
        if dec is None:
            print(f"{label:<34} {'--':>9} {'--':>6} {'--':>10} {'--':>11} "
                  f"{'--':>10} {'--':>6} {'--':>6} {'--':>6}")
            continue
        match = "vqe" if dec["match_vqe"] else ("best" if dec["match_best"] else "-")
        print(f"{label:<34} {energy:>9.3f} {energy / E_classical_ising:>6.3f} "
              f"{dec['obj_vqe']:>10.6f} {dec['obj_mean']:>11.4f} {dec['obj_top']:>10.4f} "
              f"{match:>6} {100 * dec['top_count'] / dec['shots']:>5.1f}% {n_evals:>6d}")
    print("-" * width)
    print(f"{'Classical optimum':<34} {E_classical_ising:>9.3f} {1.0:>6.3f} "
          f"{val_opt_classical:>10.6f}")


PROBLEM = make_problem(mu_pp, Sigma_pp, P_pp, RISK_Q, LAMBDA, C,
                       b_opt_classical, mu, Sigma, P, BUDGET)

print("SamplingVQE helper functions defined.")
```

    SamplingVQE helper functions defined.


<div dir="rtl" class="farsi-text">

### ۹.۲. مکانیزم کش کردن نتایج

`run_vqe` روی نمونه‌های تکراری (Qiskit sampler/ansatz/optimizer) کار می‌کند که یا pickle نمی‌شوند یا pickle آن‌ها نشانه‌ی معناداری از «آیا فیزیک مسئله عوض شده» نیست. به همین دلیل، به‌جای cache کردن خودِ این اشیا، یک "recipe" ساده و قابل‌خواندن (رشته‌ها/اعداد صریح) از همان مقادیری که پیش از ساخت این اشیا در دست است ساخته و hash می‌شود — این recipe همان چیزی است که واقعا خروجی `run_vqe` را تعیین می‌کند.

</div>


```python
# ---- Caching wrapper for run_vqe().

memory = Memory(CACHE_DIR, verbose=1)


def _hamiltonian_fingerprint(hamiltonian):
    """Stable fingerprint of a SparsePauliOp: sorted (pauli, coeff) pairs."""
    labels = hamiltonian.paulis.to_labels()
    coeffs = [complex(c) for c in hamiltonian.coeffs]
    pairs = sorted(zip(labels, (repr(c) for c in coeffs)))
    blob = json.dumps(pairs, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _ansatz_fingerprint(ansatz):
    """
    Fingerprint an ansatz by its structural config, not its object identity.
    Works for TwoLocal / RealAmplitudes / PauliTwoDesign, which all expose
    these attributes. Falls back to class name + num_qubits + reps if some
    attribute is missing -- which is what happens for an already-transpiled
    circuit, see make_initial_point.
    """
    parts = {
        "class": type(ansatz).__name__,
        "num_qubits": ansatz.num_qubits,
        "reps": getattr(ansatz, "reps", None),
    }
    # These attributes aren't present on every ansatz type.
    for attr in ("entanglement", "rotation_blocks", "entanglement_blocks"):
        val = getattr(ansatz, attr, None)
        if val is not None:
            # Gate .name is a session counter, not stable across runs; use the class name.
            if isinstance(val, (list, tuple)):
                val = [
                    type(v).__name__ if hasattr(v, "name") else str(v)
                    for v in val
                ]
            elif hasattr(val, "name") and attr != "entanglement":
                val = type(val).__name__
            else:
                val = str(val)
            parts[attr] = val
    return parts


def _optimizer_fingerprint(optimizer):
    """class + maxiter, plus any setting that differs from the class default, so a
    tuned optimizer cannot collide with the cache entry of an untuned one. With
    nothing tuned this is exactly {class, maxiter} -- what the cache was built with.
    """
    settings = dict(getattr(optimizer, "settings", None) or {})
    maxiter = getattr(optimizer, "maxiter", None)
    if maxiter is None:
        maxiter = settings.get("maxiter")

    fingerprint = {"class": type(optimizer).__name__, "maxiter": maxiter}
    try:
        defaults = dict(type(optimizer)().settings)
    except Exception:
        defaults = {}
    # repr() rather than != so numpy arrays and callables compare without raising.
    tuned = {k: repr(v) for k, v in settings.items()
             if k != "maxiter" and repr(v) != repr(defaults.get(k))}
    if tuned:
        fingerprint["tuned_settings"] = tuned
    return fingerprint


def _canonical_noise_fingerprint(nm):
    """
    NoiseModel content is unstable to fingerprint two different ways:
    1. to_dict() embeds a random UUID per QuantumError, regenerated at
       every construction (verified: differs even within one process).
    2. str(NoiseModel) and to_dict()'s internal list ordering follow
       Python dict/set iteration order, which depends on PYTHONHASHSEED --
       randomized fresh per process. Verified empirically: str(nm) for
       the exact same FakeCairoV2-derived noise model differs across
       process restarts even though content is identical (confirmed via
       diff of two separate `python3 -c` runs).
    Strip the random ids, then sort the 'errors' list into a
    canonical order before hashing, so hash-seed-driven reordering can't
    change the resulting fingerprint.
    """
    def strip_ids(o):
        if isinstance(o, dict):
            return {k: strip_ids(v) for k, v in o.items() if k != "id"}
        if isinstance(o, list):
            return [strip_ids(v) for v in o]
        return o

    d = strip_ids(nm.to_dict(serializable=True))
    if "errors" in d and isinstance(d["errors"], list):
        d["errors"] = sorted(d["errors"], key=lambda e: json.dumps(e, sort_keys=True, default=str))
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _sampler_fingerprint(sampler):
    opts = getattr(sampler, "options", None)
    shots = getattr(opts, "default_shots", None) if opts else None
    backend_options = dict(getattr(opts, "backend_options", {}) or {}) if opts else {}
    backend_options.pop("max_parallel_threads", None)
    backend_options.pop("max_parallel_experiments", None)
    backend_options.pop("max_parallel_shots", None)

    noise_model = backend_options.get("noise_model", None)
    if noise_model is not None:
        backend_options["noise_model"] = _canonical_noise_fingerprint(noise_model)

    sampler_seed = getattr(sampler, "seed", None)
    return {"shots": shots, "backend_options": backend_options, "sampler_seed": sampler_seed}


def _initial_point_seed(hamiltonian, ansatz_prepared, seed):
    """
    Deterministic seed for the initial_point draw, derived only from
    stable, pure inputs -- never from global RNG state or call order.
    """
    blob = json.dumps({
        "ham": _hamiltonian_fingerprint(hamiltonian),
        "ansatz": _ansatz_fingerprint(ansatz_prepared),
        "seed": seed,
    }, sort_keys=True, default=str)
    # Fold the hash into a valid numpy seed (0 to 2**32-1)
    return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)


def make_initial_point(hamiltonian, ansatz_prepared, seed):
    """
    Deterministic initial_point, never dependent on kernel call history.

    `ansatz_prepared` is already transpiled, so its fingerprint carries only
    {class, num_qubits}: the draw depends on the Hamiltonian, the seed and the
    parameter count, not the ansatz family. That is deliberate -- ansatzes with
    equal parameter counts start from the same point, keeping sections 10 and 13
    like-for-like. It is also part of the cache key: changing it invalidates
    every cached configuration.
    """
    n_params = ansatz_prepared.num_parameters
    local_seed = _initial_point_seed(hamiltonian, ansatz_prepared, seed)
    rng = np.random.default_rng(local_seed)
    return rng.uniform(-np.pi, np.pi, size=n_params)


def _make_cache_key(hamiltonian, ansatz, optimizer, sampler, seed, initial_point):
    key = {
        "hamiltonian": _hamiltonian_fingerprint(hamiltonian),
        "ansatz": _ansatz_fingerprint(ansatz),
        "optimizer": _optimizer_fingerprint(optimizer),
        "sampler": _sampler_fingerprint(sampler),
        "seed": seed,
        "initial_point": list(initial_point) if initial_point is not None else None,
    }
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
```

<div dir="rtl" class="farsi-text">

دو تابع زیر خودِ `run_vqe` را cache می‌کنند:

- `_run_vqe_cached`: تابع خام cache‌شده است (با `ignore` صریح روی تمام اشیای Qiskit).
- `run_vqe_cached`: رابط کاربردی است که `initial_point` را پیش از ساخت کلید cache محاسبه می‌کند — دقیقا همان نکته‌ای که در بالا توضیح داده شد.

</div>


```python
@memory.cache(ignore=["hamiltonian", "ansatz", "optimizer", "sampler",
                       "label", "progress_every", "initial_point"])
def _run_vqe_cached(cache_key, hamiltonian, ansatz, optimizer, sampler,
                     seed, label, progress_every, initial_point):
    """
    Actual cached call.

    CRITICAL: `ignore=[...]` above tells joblib to hash ONLY `cache_key`
    (and `seed`, left un-ignored as a redundant sanity check) -- NOT the
    live Qiskit objects. `cache_key` is the only thing that should
    decide cache hit/miss; `ignore` makes that actually true.
    """
    return run_vqe(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        seed=seed,
        label=label,
        progress_every=progress_every,
        initial_point=initial_point,
    )
_run_vqe_cached.func_id = "portfolio_vqe_stable/_run_vqe_cached"

def run_vqe_cached(hamiltonian, ansatz, optimizer, sampler,
                    seed=42, label="VQE", progress_every=25, initial_point=None):
    """
    Drop-in cached replacement for run_vqe(). Same signature, same
    return value (result, energy_hist, elapsed_s) -- except elapsed_s
    will read ~0 on a cache hit, which is the point.

    initial_point is now precomputed HERE, deterministically, before
    the cache key is built -- so identical (hamiltonian, ansatz, seed)
    always yields the identical initial_point, the identical cache_key,
    and therefore a guaranteed cache hit if nothing changed.
    """
    if initial_point is None:
        ansatz_prepared = _prepare_ansatz_for_sampler(ansatz, sampler)
        initial_point = make_initial_point(hamiltonian, ansatz_prepared, seed)

    cache_key = _make_cache_key(hamiltonian, ansatz, optimizer, sampler, seed, initial_point)
    return _run_vqe_cached(
        cache_key, hamiltonian, ansatz, optimizer, sampler,
        seed, label, progress_every, initial_point,
    )
```

<div dir="rtl" class="farsi-text">

### ۹.۳. یک اجرای مقدماتی VQE

پیش از مقایسه‌ی سیستماتیک بخش‌های بعد، یک اجرای منفرد VQE — با تنظیمات پیش‌فرض (`TwoLocal`، entanglement خطی، `reps=3`، COBYLA با `maxiter=200`) و sampler بدون نویز — به‌عنوان نمونه‌ای گویا اجرا می‌شود.

</div>


```python
# ---- Noiseless sampler (IBM QASM-style, ideal simulation) ----
ideal_sampler = AerSampler(
    seed=RANDOM_SEED,
    options={
        "backend_options": {
            "method": "statevector",
            "max_parallel_threads": 0,
            "max_parallel_experiments": 0,
            "max_parallel_shots": 0,
            "device": AER_DEVICE,
        }
    },
)
ideal_sampler.options.default_shots = VQE_SHOTS

n_qubits = ising_op.num_qubits
reps_default = DEFAULT_REPS
entanglement_default = DEFAULT_ENTANGLEMENT

ansatz_default = TwoLocal(
    num_qubits=n_qubits,
    rotation_blocks=["ry", "rz"],
    entanglement_blocks="cz",
    entanglement=entanglement_default,
    reps=reps_default,
)

cobyla_default = COBYLA(maxiter=DEFAULT_MAXITER)

print(f"Ansatz: TwoLocal ({entanglement_default} entanglement, reps={reps_default})")
print(f"Number of parameters: {ansatz_default.num_parameters}")
print(f"Number of qubits: {n_qubits}")
print(f"Sampler shots: {VQE_SHOTS}")
print(f"Optimizer: COBYLA (maxiter={DEFAULT_MAXITER})")
print("\nRunning VQE...")

result_ideal, hist_ideal, t_ideal = run_vqe_cached(
    hamiltonian=ising_op,
    ansatz=ansatz_default,
    optimizer=cobyla_default,
    sampler=ideal_sampler,
    label="TwoLocal+COBYLA (ideal)",
    progress_every=DEFAULT_PROGRESS_EVERY,
)
```

    Ansatz: TwoLocal (linear entanglement, reps=3)
    Number of parameters: 120
    Number of qubits: 15
    Sampler shots: 2000
    Optimizer: COBYLA (maxiter=200)
    
    Running VQE...


<div dir="rtl" class="farsi-text">

نتیجه‌ی این اجرای مقدماتی رمزگشایی و با جواب کلاسیک بخش ۸ مقایسه می‌شود.

</div>


```python
# ---- Read the portfolio back out of the optimised circuit ----
decode_ideal = decode_vqe_result(result_ideal, ansatz_default, ideal_sampler, PROBLEM)

print(f"Most frequent bitstring : {decode_ideal['b_top'].astype(int)}")
print(f"  held {decode_ideal['top_count']}/{decode_ideal['shots']} shots "
      f"({100 * decode_ideal['top_count'] / decode_ideal['shots']:.1f}%), "
      f"runner-up {decode_ideal['runner_up_count']}, "
      f"{decode_ideal['n_distinct']} distinct bitstrings measured")
print(f"VQE solution (best_measurement): {decode_ideal['b_vqe'].astype(int)}")
for t, ni in zip(TICKERS, decode_ideal["n_vqe"]):
    print(f"  {t}: {int(ni)} units")

print()
for _row, _val in [
    ("most frequent bitstring", decode_ideal["obj_top"]),
    (f"best of the {decode_ideal['shots']} shots", decode_ideal["obj_best"]),
    ("VQE best_measurement", decode_ideal["obj_vqe"]),
    ("shot-weighted mean", decode_ideal["obj_mean"]),
    ("classical optimum", val_opt_classical),
]:
    print(f"  objective, {_row:<28}: {_val: .8f}")
print(f"\nMatches classical optimum:  top={decode_ideal['match_top']}  "
      f"best={decode_ideal['match_best']}  vqe={decode_ideal['match_vqe']}")

# Cross-check: the shot-weighted objective is the portfolio-space image of E = -L - offset.
energy_from_objective = -decode_ideal["obj_mean"] - offset
print(f"\nConsistency check: eigenvalue {float(np.real(result_ideal.eigenvalue)):.4f}  vs  "
      f"-obj_mean - offset = {energy_from_objective:.4f}")
```

    Most frequent bitstring : [0 1 1 0 1 0 1 0 0 0 0 1 0 0 0]
      held 118/2000 shots (5.9%), runner-up 49, 635 distinct bitstrings measured
    VQE solution (best_measurement): [0 1 0 0 0 0 0 0 1 0 0 1 1 1 0]
      AAPL: 2 units
      IBM: 0 units
      NFLX: 1 units
      TSLA: 7 units
    
      objective, most frequent bitstring     : -0.23646513
      objective, best of the 2000 shots      :  0.00147273
      objective, VQE best_measurement        :  0.00162574
      objective, shot-weighted mean          : -1.39574963
      objective, classical optimum           :  0.00162574
    
    Matches classical optimum:  top=False  best=False  vqe=True
    
    Consistency check: eigenvalue -19.4809  vs  -obj_mean - offset = -19.4481


<div dir="rtl" class="farsi-text">

## ۱۰. مقایسه ansatz های مختلف

مطابق بخش ۳.۲ مقاله (Figure 2) — **۹ کانفیگ** آزمایش می‌شود:
- **TwoLocal** با entanglement های full / linear / circular / pairwise
- **RealAmplitudes** با entanglement های full / linear / circular / pairwise
- **PauliTwoDesign**

مقاله نشان می‌دهد TwoLocal یا RealAmplitudes با **linear یا pairwise** entanglement بهترین تعادل بین expressivity و trainability را دارند.

</div>


```python
# ---- Compare ansatzes x optimizers (Section 3.2, Figure 2 of paper) ----

ansatz_configs = {
    "PauliTwo": PauliTwoDesign(
        num_qubits=n_qubits, reps=DEFAULT_REPS, seed=RANDOM_SEED
    ),
    "TwoLocal linear": TwoLocal(
        num_qubits=n_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="linear", reps=DEFAULT_REPS
    ),
    "TwoLocal full": TwoLocal(
        num_qubits=n_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="full", reps=DEFAULT_REPS
    ),
    "TwoLocal circular": TwoLocal(
        num_qubits=n_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="circular", reps=DEFAULT_REPS
    ),
    "TwoLocal pairwise": TwoLocal(
        num_qubits=n_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="pairwise", reps=DEFAULT_REPS
    ),
    "RealAmplit. linear": RealAmplitudes(
        num_qubits=n_qubits, entanglement="linear", reps=DEFAULT_REPS
    ),
    "RealAmplit. full": RealAmplitudes(
        num_qubits=n_qubits, entanglement="full", reps=DEFAULT_REPS
    ),
    "RealAmplit. circular": RealAmplitudes(
        num_qubits=n_qubits, entanglement="circular", reps=DEFAULT_REPS
    ),
    "RealAmplit. pairwise": RealAmplitudes(
        num_qubits=n_qubits, entanglement="pairwise", reps=DEFAULT_REPS
    ),
}

# Order matches the paper's Figure 2 legend, top to bottom.
ANSATZ_ORDER = list(ansatz_configs.keys())

optimizer_factories = {
    "Cobyla": lambda: COBYLA(maxiter=ANSATZ_MAXITER),
    "SPSA":   lambda: SPSA(maxiter=ANSATZ_MAXITER // 2),
    "NFT":    lambda: NFT(maxiter=ANSATZ_MAXITER // 2),
}

# Leading evaluations each optimizer spends before it starts descending.
calibration_evals = {name: optimizer_calibration_evals(factory())
                     for name, factory in optimizer_factories.items()}

results_ansatz_opt = {opt_name: {} for opt_name in OPTIMIZER_ORDER}

print("Running VQE: 9 ansatzes x 3 optimizers (noiseless)...\n")
print(f"reps={DEFAULT_REPS} | maxiter={ANSATZ_MAXITER}|{ANSATZ_MAXITER // 2}|{ANSATZ_MAXITER // 2}")

for opt_name in OPTIMIZER_ORDER:
    for name in ANSATZ_ORDER:
        ansatz = ansatz_configs[name]
        optimizer = optimizer_factories[opt_name]()
        label = f"{name} / {opt_name}"
        res, hist, elapsed = run_vqe_cached(
            hamiltonian=ising_op,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=ideal_sampler,
            label=label,
            progress_every=ANSATZ_PROGRESS_EVERY,
        )
        results_ansatz_opt[opt_name][name] = {
            "result": res, "hist": hist, "elapsed": elapsed,
            "decode": decode_vqe_result(res, ansatz, ideal_sampler, PROBLEM),
        }

# Ranked on the shot-weighted objective, per section 9.1.
best_opt_name, best_ansatz_name = max(
    ((o, a) for o in results_ansatz_opt for a in results_ansatz_opt[o]),
    key=lambda t: results_ansatz_opt[t[0]][t[1]]["decode"]["obj_mean"]
)
print(f"Best (ansatz, optimizer) by shot-weighted objective: "
      f"{best_ansatz_name}, {best_opt_name}")
```

    Running VQE: 9 ansatzes x 3 optimizers (noiseless)...
    
    reps=3 | maxiter=100|50|50
    Best (ansatz, optimizer) by shot-weighted objective: PauliTwo, NFT


<div dir="rtl" class="farsi-text">

نتایج بالا در قالب سه پنل (به تفکیک بهینه‌ساز)، مطابق قالب شکل ۲ مقاله، رسم می‌شوند. محور افقی تعداد *ارزیابی تابع هدف* است و ناحیه‌ی خاکستری در پنل SPSA مرحله‌ی کالیبراسیون آن را نشان می‌دهد. خط‌چین افقی، انرژی جواب بهینه‌ی کلاسیک است.

در جدول زیر، ستون `E/E*` همان نسبت کیفیت جواب است که مقاله در شکل ۶ خود گزارش می‌کند (۱.۰ یعنی رسیدن به بهینه‌ی کلاسیک).

</div>


```python
# ---- Plot convergence: ansatzes comparison (Figure 2 style, 3 subplots) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# One fixed color per ansatz across all three subplots, as in the paper.
colors_map = plt.cm.tab10.colors
ansatz_colors = {name: colors_map[i % 10] for i, name in enumerate(ANSATZ_ORDER)}

for ax, opt_name in zip(axes, OPTIMIZER_ORDER):
    shade_calibration(ax, calibration_evals.get(opt_name, 0))
    for name in ANSATZ_ORDER:
        hist = results_ansatz_opt[opt_name][name]["hist"]
        ax.plot(range(len(hist)), hist, label=name,
                color=ansatz_colors[name], linewidth=1.2, alpha=0.85)
    ax.axhline(y=E_classical_ising, color="black", linestyle=":", linewidth=1.5)
    ax.set_title(opt_name, fontsize=13)
    ax.set_xlabel("Objective evaluations", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r"$E_{min}$", fontsize=13)
axes[-1].legend(loc="upper right", fontsize=8, ncol=1)

plt.suptitle("Convergence: 9 Ansatz Configurations x 3 Optimizers (noiseless)\n"
             "[Replication of Figure 2, Buonaiuto et al. 2023]", fontsize=12, y=1.03)
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/convergence_ansatzes.png", bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

print_run_table(
    [(f"{name} / {opt_name}",
      results_ansatz_opt[opt_name][name]["decode"],
      float(np.real(results_ansatz_opt[opt_name][name]["result"].eigenvalue)),
      len(results_ansatz_opt[opt_name][name]["hist"]))
     for opt_name in OPTIMIZER_ORDER for name in ANSATZ_ORDER],
    header="Ansatz / Optimizer",
)
```

    
    ================================================================================================================
    Ansatz / Optimizer                     E_vqe   E/E*   obj(vqe)   obj(mean)   obj(top)  match   top%  evals
    ----------------------------------------------------------------------------------------------------------------
    PauliTwo / Cobyla                    -18.366  0.881   0.001605     -2.6311    -2.7315      -   6.7%    100
    TwoLocal linear / Cobyla             -18.020  0.864   0.001626     -2.8123    -1.2928    vqe   4.2%    122
    TwoLocal full / Cobyla               -11.574  0.555   0.001626     -9.2601    -1.9057    vqe   0.3%    122
    TwoLocal circular / Cobyla           -17.226  0.826   0.001534     -3.7589    -1.4020      -   2.3%    122
    TwoLocal pairwise / Cobyla           -18.020  0.864   0.001626     -2.8123    -1.2928    vqe   4.2%    122
    RealAmplit. linear / Cobyla          -14.930  0.716   0.001626     -6.3360    -0.3468    vqe   0.7%    100
    RealAmplit. full / Cobyla            -17.405  0.835   0.001626     -3.7318    -0.7513    vqe   1.0%    100
    RealAmplit. circular / Cobyla        -10.074  0.483   0.001626    -11.5206   -10.6421    vqe   1.4%    100
    RealAmplit. pairwise / Cobyla        -17.468  0.838   0.001626     -3.5891    -1.5921    vqe   2.3%    100
    PauliTwo / SPSA                      -18.214  0.874   0.001605     -2.7308    -2.8621      -   1.6%    151
    TwoLocal linear / SPSA               -17.535  0.841   0.001626     -3.4318    -0.2329    vqe   0.5%    151
    TwoLocal full / SPSA                 -15.270  0.733   0.001626     -5.6402    -1.4271    vqe   0.8%    151
    TwoLocal circular / SPSA             -18.177  0.872   0.001605     -2.7721    -0.3841      -   1.4%    151
    TwoLocal pairwise / SPSA             -17.535  0.841   0.001626     -3.4318    -0.2329    vqe   0.5%    151
    RealAmplit. linear / SPSA            -16.697  0.801   0.001626     -4.3170    -2.8684    vqe   1.7%    151
    RealAmplit. full / SPSA              -15.907  0.763   0.001605     -5.1386    -1.5441      -   0.6%    151
    RealAmplit. circular / SPSA          -15.050  0.722   0.001626     -5.6840     0.0014    vqe   0.7%    151
    RealAmplit. pairwise / SPSA          -18.141  0.870   0.001626     -2.7828    -5.3541    vqe   1.6%    151
    PauliTwo / NFT                       -20.235  0.971   0.001626     -0.7013    -0.0051    vqe  16.2%    103
    TwoLocal linear / NFT                -19.361  0.929   0.001626     -1.4824    -1.6827    vqe   4.4%    103
    TwoLocal full / NFT                  -16.296  0.782   0.001626     -4.6205    -1.5375    vqe   0.8%    103
    TwoLocal circular / NFT              -19.246  0.923   0.001626     -1.7309    -1.7692    vqe   8.2%    103
    TwoLocal pairwise / NFT              -19.361  0.929   0.001626     -1.4824    -1.6827    vqe   4.4%    103
    RealAmplit. linear / NFT             -15.653  0.751   0.001626     -5.2668    -2.8375    vqe   2.5%    103
    RealAmplit. full / NFT               -19.213  0.922   0.001626     -1.6955    -0.3752    vqe   2.6%    103
    RealAmplit. circular / NFT           -12.328  0.591   0.001626     -8.5361    -0.2185    vqe   1.9%    103
    RealAmplit. pairwise / NFT           -18.878  0.906   0.001534     -1.9549    -1.0727      -   4.0%    103
    ----------------------------------------------------------------------------------------------------------------
    Classical optimum                    -20.845  1.000   0.001626


<div dir="rtl" class="farsi-text">

## ۱۱. مقایسه Optimizer های مختلف

مطابق بخش ۳.۲ مقاله (Figure 3): COBYLA، SPSA، NFT

- **COBYLA**: بهترین عملکرد در شبیه‌ساز بدون نویز (مقاله)
- **SPSA**: مناسب برای سخت‌افزار واقعی با نویز
- **NFT** (Nakanishi-Fujii-Todo): سریع‌ترین همگرایی در موارد خاص

در این مقایسه، ansatz ثابت نگه داشته می‌شود تا تنها متغیر، بهینه‌ساز باشد؛ انتخاب `TwoLocal` با entanglement خطی از همان توصیه‌ی مقاله می‌آید که linear/pairwise را مصالحه‌ی کارآمد بین expressivity و trainability می‌داند.

</div>


```python
# ---- Compare optimizers (Section 3.2, Figure 3 of paper) ----
opt_progress_every = 25

optimizer_configs = {
    "COBYLA": COBYLA(maxiter=200),
    "SPSA":   SPSA(maxiter=200),
    "NFT":    NFT(maxiter=200),
}
optimizer_calibration = {name: optimizer_calibration_evals(o)
                         for name, o in optimizer_configs.items()}

results_optimizer = {}
print("Running VQE with different optimizers (TwoLocal linear, noiseless)...\n")

for opt_name, optimizer in optimizer_configs.items():
    ansatz = TwoLocal(
        num_qubits=n_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="linear", reps=DEFAULT_REPS
    )
    res, hist, elapsed = run_vqe_cached(
        hamiltonian=ising_op,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=ideal_sampler,
        label=f"TwoLocal+{opt_name}",
        progress_every=opt_progress_every,
    )
    results_optimizer[opt_name] = {
        "result": res, "hist": hist, "elapsed": elapsed,
        "decode": decode_vqe_result(res, ansatz, ideal_sampler, PROBLEM),
    }
```

    Running VQE with different optimizers (TwoLocal linear, noiseless)...
    


<div dir="rtl" class="farsi-text">

منحنی‌های همگرایی سه بهینه‌ساز، روی یک محور مشترک، مطابق قالب شکل ۳ مقاله رسم می‌شوند. محور افقی تعداد ارزیابی تابع هدف است — نه تعداد تکرار — چون هر بهینه‌ساز به ازای هر تکرار تعداد متفاوتی ارزیابی مصرف می‌کند؛ ناحیه‌ی خاکستری ابتدای نمودار، مرحله‌ی کالیبراسیون SPSA است.

</div>


```python
# ---- Plot convergence: optimizers comparison (Figure 3 style) ----
fig, ax = plt.subplots(figsize=(11, 5))

opt_colors = {"COBYLA": "#1f77b4", "SPSA": "#ff7f0e", "NFT": "#2ca02c"}
opt_styles = {"COBYLA": "-", "SPSA": "--", "NFT": "-."}

# Band marking SPSA's calibration phase, per the note above.
shade_calibration(ax, max(optimizer_calibration.values()), label="SPSA calibration")

for opt_name, data in results_optimizer.items():
    hist = data["hist"]
    ax.plot(range(len(hist)), hist,
            label=opt_name,
            color=opt_colors[opt_name],
            linestyle=opt_styles[opt_name],
            linewidth=2.0)

# Classical optimum in Ising energy space
ax.axhline(y=E_classical_ising, color="black", linestyle=":",
           linewidth=2.5, label="Classical optimum (Ising energy)")

ax.set_xlabel("Objective evaluations", fontsize=12)
ax.set_ylabel("Energy (Hamiltonian eigenvalue)", fontsize=12)
ax.set_title("Convergence: Different Optimizers (TwoLocal linear, noiseless)\n"
             "[Section 3.2, Buonaiuto et al. 2023]", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/convergence_optimizers.png", bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

print_run_table(
    [(name,
      data["decode"],
      float(np.real(data["result"].eigenvalue)),
      len(data["hist"]))
     for name, data in results_optimizer.items()],
    header="Optimizer",
)
print("\nObjective evaluations per optimizer (same nominal maxiter=200): "
      + ", ".join(f"{name}={len(d['hist'])}"
                  + (f" of which {optimizer_calibration[name]} calibration"
                     if optimizer_calibration[name] else "")
                  for name, d in results_optimizer.items()))
```

    
    ================================================================================================================
    Optimizer                              E_vqe   E/E*   obj(vqe)   obj(mean)   obj(top)  match   top%  evals
    ----------------------------------------------------------------------------------------------------------------
    COBYLA                               -19.481  0.935   0.001626     -1.3957    -0.2365    vqe   5.9%    200
    SPSA                                 -19.371  0.929   0.001626     -1.5421    -3.0002    vqe   1.1%    451
    NFT                                  -20.754  0.996   0.001626     -0.0903    -0.0411    vqe  45.9%    408
    ----------------------------------------------------------------------------------------------------------------
    Classical optimum                    -20.845  1.000   0.001626
    
    Objective evaluations per optimizer (same nominal maxiter=200): COBYLA=200, SPSA=451 of which 50 calibration, NFT=408


<div dir="rtl" class="farsi-text">

## ۱۲. مقایسه ضریب جریمه λ

مطابق بخش ۳.۲ مقاله (**Figure 4**): تأثیر λ بر کیفیت جواب

**توجه:** Figure 6 مقاله نتایج سخت‌افزار واقعی (box plot QV) را نشان می‌دهد. Figure 4 تأثیر λ روی frontier مارکویتز را نشان می‌دهد.

مقاله نشان می‌دهد بازه

<div dir="ltr">

**$$1 ≤ λ ≤ 10$$**

</div>

نتایج خوبی می‌دهد — مقدار خیلی کم محدودیت بودجه را نقض می‌کند و مقدار خیلی زیاد ترم جریمه را بر تابع هدف غالب می‌کند.

 برخلاف بخش‌های ۱۰ و ۱۱، ansatz این بخش با `reps=1` و روی sampler نویزی (نه ideal) اجرا می‌شود — رجوع شود به بخش «اثر ضریب جریمه λ» گزارش برای دلیل و پیامدهای این تفاوت.



</div>

<div dir="rtl" class="farsi-text">

### ۱۲.۱. ساخت مدل نویز IBM Cairo

مدل نویز کالیبراسیون‌شده‌ی دستگاه واقعی IBM Cairo (یا معادل تقریبی آن، در صورت نبود دسترسی) ساخته می‌شود و sampler نویزی این بخش با آن پیکربندی می‌شود. مقاله همین کار را چنین توصیف می‌کند: «وارد کردن مدل نویز از مشخصات کامپیوتر کوانتومی IBM Cairo».

تابع زیر سه منبع را به ترتیب امتحان می‌کند: `FakeCairoV2` (از `qiskit-ibm-runtime`)، سپس `FakeCairo` (رابط قدیمی‌تر)، و در نهایت یک تقریب depolarizing کالیبره‌شده با مشخصات Cairo. نقشه‌ی اتصال‌پذیری برگشتی فقط در ارزیابی توپولوژی زیر استفاده می‌شود؛ خودِ sampler عمدا بدون آن اجرا می‌شود (دلیل و هزینه‌اش در ادامه).

</div>


```python
# ---- Build IBM Cairo noise model (Figure 3 of paper) ----

def build_cairo_noise_model():
    """Return (noise_model, coupling_map, source).

    The coupling map is used only by the topology audit below; the sampler
    itself deliberately runs without one, see that cell for what that costs.
    """
    # FakeCairoV2 (qiskit >= 0.45)
    try:
        from qiskit_ibm_runtime.fake_provider import FakeCairoV2
        backend = FakeCairoV2()
        return NoiseModel.from_backend(backend), backend.coupling_map, "FakeCairoV2"
    except Exception:
        pass
    # FakeCairo (older API)
    try:
        from qiskit.providers.fake_provider import FakeCairo
        backend = FakeCairo()
        coupling = getattr(backend.configuration(), "coupling_map", None)
        return NoiseModel.from_backend(backend), coupling, "FakeCairo"
    except Exception:
        pass
    # fallback: depolarizing Cairo (p1q~0.03%, p2q~0.8%)
    import qiskit_aer.noise as noise_mod
    nm = noise_mod.NoiseModel()
    nm.add_all_qubit_quantum_error(noise_mod.depolarizing_error(0.0003, 1), ["rz", "sx", "x", "ry"])
    nm.add_all_qubit_quantum_error(noise_mod.depolarizing_error(0.008, 2), ["cz", "cx"])
    nm.add_all_qubit_readout_error(noise_mod.ReadoutError([[0.985, 0.015], [0.015, 0.985]]))
    return nm, None, "Cairo-calibrated depolarizing approximation"

cairo_nm, cairo_coupling_map, cairo_noise_source = build_cairo_noise_model()
print(f"Noise model source: {cairo_noise_source}")

noisy_sampler = AerSampler(
    seed=RANDOM_SEED,
    options={
        "backend_options": {
            "noise_model": cairo_nm,
            "method": AER_NOISY_METHOD,
            "device": AER_DEVICE,
            "batched_shots_gpu": True,
            "batched_shots_gpu_max_qubits": 16,
            "max_parallel_threads": 0,
            "max_parallel_experiments": 0,
        }
    },
)
noisy_sampler.options.default_shots = VQE_SHOTS
```

    Noise model source: FakeCairoV2


<div dir="rtl" class="farsi-text">

**ارزیابی توپولوژی**

مدل نویز Cairo کانال‌های خطا را بر اساس *نام گیت* اعمال می‌کند، اما شبیه‌ساز Aer هیچ نقشه‌ی اتصال‌پذیری (coupling map) ندارد؛ یعنی هر جفت کیوبیت طوری شبیه‌سازی می‌شود که انگار مستقیما به هم متصل است. روی دستگاه واقعی، جفت‌های غیرمجاور به زنجیره‌ای از گیت‌های SWAP نیاز دارند و مقاله بخش عمده‌ی افت کیفیت `full` را در حالت نویزی دقیقا به همین سربار نسبت می‌دهد.

سلول زیر این شکاف را کمّی می‌کند: هر ansatz یک‌بار همان‌طور که واقعا شبیه‌سازی می‌شود و یک‌بار روی نقشه‌ی اتصال‌پذیری واقعی Cairo ترنسپایل می‌شود و تعداد گیت دو-کیوبیتی و عمق مدار مقایسه می‌شود.

مهم است که این را یک کمبود نسبت به مقاله ندانیم. شکل‌های ۲ و ۳ مقاله نیز شبیه‌سازی‌اند و شواهد مربوط به توپولوژی در آن مقاله از شکل ۶ می‌آید، که روی سخت‌افزار واقعی اجرا شده — نه از شبیه‌سازی. بنابراین اجرای مسیریابی‌شده، اگر انجام می‌شد، فراتر از شبیه‌سازی‌های خودِ مقاله بود و نه جبران عقب‌ماندگی از آن‌ها. چنین اجرایی در بودجه‌ی محاسباتی این پروژه نمی‌گنجید (سربار حدود ۲.۲ برابری روی کل جاروب نویزی) و به مدیریت جابه‌جایی کیوبیت‌ها پس از مسیریابی نیاز داشت. آنچه در این سلول گزارش می‌شود، اندازه‌ی دقیق همان چیزی است که لحاظ نشده — که نتیجه‌گیری بخش ۱۳ را به یک کران پایین تبدیل می‌کند، نه یک ادعای بی‌پشتوانه.

</div>


```python
# ---- Topology audit: the SWAP cost this simulation does not pay (transpile-only) ----
topology_audit = None
if RUN_TOPOLOGY_DIAGNOSTIC and cairo_coupling_map is not None:

    def transpilation_cost(ansatz, coupling_map=None, seed=11):
        tqc = transpile(ansatz.decompose(reps=10), basis_gates=cairo_nm.basis_gates,
                        coupling_map=coupling_map, optimization_level=1,
                        seed_transpiler=seed)
        two_qubit = sum(v for k, v in tqc.count_ops().items()
                        if k in ("cx", "cz", "ecr", "swap"))
        return two_qubit, tqc.depth()

    topology_audit = {}
    print(f"{'Ansatz':<24} {'2q simulated':>13} {'2q on Cairo':>12} {'factor':>8} "
          f"{'depth sim':>10} {'depth Cairo':>12}")
    print("-" * 84)
    for name in ANSATZ_ORDER:
        sim_2q, sim_depth = transpilation_cost(ansatz_configs[name])
        dev_2q, dev_depth = transpilation_cost(ansatz_configs[name], cairo_coupling_map)
        overhead = dev_2q / sim_2q if sim_2q else float("nan")
        topology_audit[name] = {
            "two_qubit_simulated": sim_2q, "two_qubit_on_device": dev_2q,
            "depth_simulated": sim_depth, "depth_on_device": dev_depth,
            "two_qubit_overhead": overhead,
        }
        print(f"{name:<24} {sim_2q:>13d} {dev_2q:>12d} {overhead:>7.1f}x "
              f"{sim_depth:>10d} {dev_depth:>12d}")
    print("-" * 84)
else:
    print("Topology audit skipped (flag off, or no coupling map available).")
```

    Ansatz                    2q simulated  2q on Cairo   factor  depth sim  depth Cairo
    ------------------------------------------------------------------------------------
    PauliTwo                            42           42     1.0x         39           37
    TwoLocal linear                     42           42     1.0x         94           79
    TwoLocal full                      315         1276     4.1x        141         1673
    TwoLocal circular                   45          177     3.9x        200          629
    TwoLocal pairwise                   42           42     1.0x         46           48
    RealAmplit. linear                  42           42     1.0x         34           61
    RealAmplit. full                   315         1239     3.9x         73         1577
    RealAmplit. circular                45          177     3.9x         61          574
    RealAmplit. pairwise                42           42     1.0x         22           37
    ------------------------------------------------------------------------------------


<div dir="rtl" class="farsi-text">

### ۱۲.۲. پیمایش λ

برای هر یک از شش مقدار $\lambda \in \{0, 1, 10, 10^2, 10^3, 10^4\}$، مسئله‌ی QUBO و هامیلتونین متناظر از نو ساخته و VQE (تحت همین sampler نویزی، با `maxiter=250`) اجرا می‌شود. همین شبکه‌ی مقادیر از خودِ مقاله می‌آید: «جوابِ متناظر با $\lambda=0$ لزوما قیدهای بودجه را رعایت نمی‌کند... و با ادامه‌ی افزایش $\lambda$ در توان‌های ۱۰، جواب‌های زیربهینه به دست می‌آیند.»

جواب هر λ دو بار امتیاز داده می‌شود: یک‌بار زیر جریمه‌ی خودش (همان تابع هدفی که در برابرش بهینه شده) و یک‌بار زیر جریمه‌ی مرجع مشترک `LAMBDA` — و همین دومی است که شش اجرا را با یکدیگر قابل‌مقایسه می‌کند.

</div>


```python
# ---- Penalty coefficient lambda sweep (Section 3.2, Figure 4 of paper) ----
lambda_values = [0, 1, 10, 10**2, 10**3, 10**4]
lambda_maxiter = 250
lambda_progress_every = 25
results_lambda = {}

print("Sweeping penalty coefficient lambda...\n")
print(f"maxiter={lambda_maxiter}")

for lam in lambda_values:
    # Rebuild QUBO and Hamiltonian for this lambda
    qp_lam = build_qubo(mu_pp, Sigma_pp, P_pp, q=RISK_Q, lam=lam)
    qubo_lam = converter.convert(qp_lam)
    ham_lam, offset_lam = to_ising(qubo_lam)
    ham_lam = ham_lam.simplify(atol=1e-12)

    ansatz_lam = TwoLocal(
        num_qubits=ham_lam.num_qubits, rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz", entanglement="linear", reps=1   # not DEFAULT_REPS; see above
    )
    optimizer_lam = COBYLA(maxiter=lambda_maxiter)

    res, hist, elapsed = run_vqe_cached(
        hamiltonian=ham_lam,
        ansatz=ansatz_lam,
        optimizer=optimizer_lam,
        sampler=noisy_sampler,
        label=f"lambda={lam}",
        progress_every=lambda_progress_every,
    )

    # Scored under its own penalty and, via reference_lam, under the common one.
    problem_lam = make_problem(mu_pp, Sigma_pp, P_pp, RISK_Q, lam, C,
                               b_opt_classical, mu, Sigma, P, BUDGET)
    decode = decode_vqe_result(res, ansatz_lam, noisy_sampler, problem_lam,
                               reference_lam=LAMBDA)

    results_lambda[lam] = {
        "result": res, "hist": hist, "elapsed": elapsed, "ansatz": ansatz_lam,
        "decode": decode,
        "eigenvalue": float(np.real(res.eigenvalue)),
        "offset": float(offset_lam),
        "constraint_ok": abs(decode["budget_mean"] - 1.0) < BUDGET_TOL,
    }
```

    Sweeping penalty coefficient lambda...
    
    maxiter=250


<div dir="rtl" class="farsi-text">

### ۱۲.۳. نمودار مرز کارا برای هر λ

جواب هر λ (میانگین‌گیری‌شده روی توزیع کامل شات‌ها) در فضای اصلی پورتفولیو رمزگشایی و روی مرز کارای مارکوویتز رسم می‌شود — همراه با ابر پورتفولیوهای تصادفی کاملا سرمایه‌گذاری‌شده، مطابق شکل ۴ مقاله.

</div>


```python
# ---- Section 3.2, Figure 4 — Effect of Penalty Coefficient λ ----

def sample_random_portfolios(mu, Sigma, n_samples, seed):
    """Fully-invested random allocations (the paper's "Random Allocations" cloud).

    Dirichlet weights sum to exactly 1, which is the signature of the paper's
    Fig. 4 cloud not touching the origin.
    """
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(mu)), size=n_samples)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, Sigma, w))
    return vols, rets


def matches_softly(ret, vol, r_classical, v_classical, tol=SOFT_MATCH_TOL):
    """Paper-faithful notion of 'reaches classical': is the decoded
    (return, volatility) point within `tol` relative distance of the classical
    point, rather than requiring a bit-for-bit identical bitstring."""
    rel = np.hypot((ret - r_classical) / r_classical,
                    (vol - v_classical) / v_classical)
    return bool(rel < tol)


# ---- 1. Random fully-invested portfolios, and the classical reference point ----
cloud_vols, cloud_rets = sample_random_portfolios(mu, Sigma, N_RANDOM_PORTFOLIOS, RANDOM_SEED)
r_classical, v_classical = portfolio_return_risk_original(
    b_opt_classical, C, mu, Sigma, P, BUDGET
)

# ---- 2. Position of each lambda in return/volatility space (shot-weighted mean) ----
lambda_frontier_points = {}
for lam in lambda_values:
    dec = results_lambda[lam]["decode"]
    lambda_frontier_points[lam] = {
        "ret": dec["ret_mean"], "vol": dec["vol_mean"],
        "budget_val": dec["budget_mean"],
        "match": dec["match_vqe"],
        "soft_match": matches_softly(dec["ret_mean"], dec["vol_mean"],
                                      r_classical, v_classical),
    }

# ---- 3. Plot: Figure 4 style — single scatter panel (Effect of λ) ----
fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(cloud_vols, cloud_rets, alpha=0.15, s=10, color="lightblue",
           label="Random allocations (fully invested)", zorder=1)

# Classical answer in original space (black square, matches paper)
ax.scatter([v_classical], [r_classical], s=250, color="black",
           marker="s", zorder=5, label="Classical optimal (brute-force)",
           linewidths=1.5)

# VQE points for each lambda — marker/color scheme matching paper's Fig 4 legend
marker_map = {0: "*", 1: "p", 10: "o", 100: "^", 1000: "D", 10000: "h"}
color_map = {
    0:     "#c0504d",   # red-ish star, as in paper
    1:     "#33cc33",   # bright green pentagon
    10:    "#4d804d",   # darker green circle
    100:   "#ff9900",   # orange triangle
    1000:  "#e6197a",   # pink/magenta diamond
    10000: "#ff00ff",   # magenta hexagon
}

for lam in lambda_values:
    d = lambda_frontier_points[lam]
    label = f"λ={lam}" if lam not in (100, 1000, 10000) else f"λ=10^{len(str(lam))-1}"
    ax.scatter(
        [d["vol"]], [d["ret"]],
        s=200,
        color=color_map.get(lam, "gray"),
        marker=marker_map.get(lam, "o"),
        zorder=4,
        label=f"{label}  (P''b={d['budget_val']:.3f})",
        edgecolors="black", linewidths=0.8,
    )

ax.set_xlabel("Portfolio Volatility (σ)", fontsize=12)
ax.set_ylabel("Portfolio Expected Return (μ)", fontsize=12)
ax.set_title("Effect of λ on Solution Quality\n[Figure 4, Buonaiuto et al. 2023]", fontsize=12)
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/lambda_frontier.png", bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# ---- 4. Summary table of λ results ----
budget_col = "P''b"
obj_col = f"obj @λ={LAMBDA:g}"
print("\n" + "=" * 100)
print(f"{'λ':>7}  {'Return μ':>10}  {'Volatility σ':>12}  {budget_col:>8}  {'budget ok':>9}  "
      f"{obj_col:>13}  {'match':>6}  {'soft':>5}")
print("-" * 100)
for lam in lambda_values:
    d = lambda_frontier_points[lam]
    dec = results_lambda[lam]["decode"]
    print(f"{lam:>7}  {d['ret']:>10.6f}  {d['vol']:>12.6f}  {d['budget_val']:>8.3f}  "
          f"{str(results_lambda[lam]['constraint_ok']):>9}  {dec['obj_ref_vqe']:>13.6f}  "
          f"{str(d['match']):>6}  {str(d['soft_match']):>5}")
print("-" * 100)
print(f"{'Classical':>7}  {r_classical:>10.6f}  {v_classical:>12.6f}  {'1.000':>8}  "
      f"{'—':>9}  {val_opt_classical:>13.6f}")
print("=" * 100)

# ---- 5. Pick a lambda, on two stated criteria rather than one implicit one ----
feasible_lambdas = [lam for lam in lambda_values
                    if abs(lambda_frontier_points[lam]["budget_val"] - 1.0) <= BUDGET_TOL]
best_lambda_by_objective = (
    max(feasible_lambdas, key=lambda l: results_lambda[l]["decode"]["obj_ref_vqe"])
    if feasible_lambdas else None
)
best_lambda_by_budget = min(
    lambda_values, key=lambda l: abs(lambda_frontier_points[l]["budget_val"] - 1.0)
)
best_lambda_val = best_lambda_by_objective

print(f"\nBudget constraint satisfied (|P''b - 1| <= {BUDGET_TOL}) for λ in {feasible_lambdas}")
print(f"Best λ by objective at the common reference penalty λ={LAMBDA:g}: {best_lambda_by_objective}")
print(f"Closest budget usage to 1: λ = {best_lambda_by_budget}")
```

    
    ====================================================================================================
          λ    Return μ  Volatility σ      P''b  budget ok      obj @λ=10   match   soft
    ----------------------------------------------------------------------------------------------------
          0    0.005751      0.072166     3.276      False     -60.584292   False  False
          1    0.001435      0.019614     1.019       True       0.001534   False  False
         10    0.001550      0.022121     1.028       True       0.001473   False  False
        100    0.001282      0.018954     0.940       True       0.001605   False  False
       1000    0.001554      0.021436     1.017       True       0.001605   False  False
      10000    0.001659      0.025195     1.020       True       0.000901   False  False
    ----------------------------------------------------------------------------------------------------
    Classical    0.002093      0.028465     1.000          —       0.001626
    ====================================================================================================
    
    Budget constraint satisfied (|P''b - 1| <= 0.15) for λ in [1, 10, 100, 1000, 10000]
    Best λ by objective at the common reference penalty λ=10: 100
    Closest budget usage to 1: λ = 1000


<div dir="rtl" class="farsi-text">

## ۱۳. شبیه‌سازی با نویز

مطابق بخش ۲.۳ مقاله (NISQ devices): اضافه کردن noise model سفارشی  
شامل: depolarizing error روی گیت‌های یک‌کیوبیتی و دو‌کیوبیتی، و readout error

**مهم:** مطابق بخش ۳.۲ مقاله (Figure 3)، COBYLA بهترین عملکرد را **حتی در شبیه‌سازی با نویز** دارد. SPSA برای سخت‌افزار واقعی NISQ توصیه می‌شود که گرادیان‌ها دارای نویز سخت‌افزاری هستند.

**نکته‌ی پیاده‌سازی:** SPSA برای هر تخمین گرادیان، دو مدار را به‌صورت دسته‌ای (batch) به sampler می‌دهد؛ این کار مسیر batched-shots روی GPU را دچار مشکل می‌کند. به همین دلیل SPSA sampler اختصاصی خودش را می‌گیرد که در آن این گزینه خاموش است، و بقیه‌ی بهینه‌سازها روی همان sampler نویزی بخش ۱۲.۱ اجرا می‌شوند.

**قید مهم بر تفسیر این بخش:** شبیه‌ساز بدون نقشه‌ی اتصال‌پذیری اجرا می‌شود، پس ansatzهای با درهم‌تنیدگی متراکم (به‌ویژه `full`) هیچ‌یک از هزینه‌ی مسیریابی/SWAP خود را نمی‌پردازند — دقیقا همان سازوکاری که مقاله افت کیفیت آن‌ها را به آن نسبت می‌دهد. اندازه‌ی دقیق این هزینه در ارزیابی توپولوژی بخش ۱۲.۱ محاسبه شده و در انتهای جدول همین بخش نیز یادآوری می‌شود.

</div>


```python
# ---- Compare ansatzes x optimizers under noise (Figure 3 of paper) ----

spsa_sampler = AerSampler(
    seed=RANDOM_SEED,
    options={"backend_options": {
        "noise_model": cairo_nm,
        "method": AER_NOISY_METHOD,
        "device": AER_DEVICE,
        "batched_shots_gpu": False,
        "max_parallel_threads": 0,
        "max_parallel_experiments": 0
    }
  },
)
spsa_sampler.options.default_shots = VQE_SHOTS

# Reuse the exact same 9 ansatz configs and ordering as Figure 2
optimizer_factories_noisy = {
    "Cobyla": lambda: COBYLA(maxiter=ANSATZ_MAXITER),
    "SPSA":   lambda: SPSA(maxiter=ANSATZ_MAXITER // 2),
    "NFT":    lambda: NFT(maxiter=ANSATZ_MAXITER // 2),
}
calibration_evals_noisy = {name: optimizer_calibration_evals(factory())
                           for name, factory in optimizer_factories_noisy.items()}

results_ansatz_opt_noisy = {opt_name: {} for opt_name in OPTIMIZER_ORDER_NOISY}
noisy_failures = {}

print(f"Running VQE: 9 ansatzes x {len(OPTIMIZER_ORDER_NOISY)} optimizer(s) (IBM Cairo noise model)...\n")
print(f"reps={DEFAULT_REPS} | maxiter={ANSATZ_MAXITER}|{ANSATZ_MAXITER // 2}|{ANSATZ_MAXITER // 2}")

for opt_name in OPTIMIZER_ORDER_NOISY:
    for name in ANSATZ_ORDER:
        ansatz = ansatz_configs[name]
        optimizer = optimizer_factories_noisy[opt_name]()
        # SPSA gets its own sampler, with the GPU batched-shots path off.
        sampler = spsa_sampler if opt_name == "SPSA" else noisy_sampler
        label = f"{name} / {opt_name} [Cairo]"
        try:
            res, hist, elapsed = run_vqe_cached(
                hamiltonian=ising_op,
                ansatz=ansatz,
                optimizer=optimizer,
                sampler=sampler,
                label=label,
                progress_every=ANSATZ_PROGRESS_EVERY,
            )
        except Exception as exc:
            # Record and move on; the plot and summary below render it as a gap.
            print(f"[WARN] {label} failed: {type(exc).__name__}: {exc}")
            noisy_failures[f"{opt_name}/{name}"] = f"{type(exc).__name__}: {exc}"
            continue

        # Decode under the same sampler the optimisation ran on.
        results_ansatz_opt_noisy[opt_name][name] = {
            "result": res, "hist": hist, "elapsed": elapsed,
            "decode": decode_vqe_result(res, ansatz, sampler, PROBLEM),
        }

if noisy_failures:
    print(f"\n[INFO] {len(noisy_failures)} combination(s) failed and were skipped.")
```

    Running VQE: 9 ansatzes x 3 optimizer(s) (IBM Cairo noise model)...
    
    reps=3 | maxiter=100|50|50


<div dir="rtl" class="farsi-text">

نتایج بالا، مشابه بخش ۱۰، در قالب سه پنل رسم می‌شوند.

</div>


```python
# ---- Plot: 9 ansatzes x 3 optimizers under Cairo noise (Figure 3 style) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

n_missing_total = 0

for ax, opt_name in zip(axes, OPTIMIZER_ORDER):
    shade_calibration(ax, calibration_evals_noisy.get(opt_name, 0))
    opt_results = results_ansatz_opt_noisy.get(opt_name, {})
    missing_this_opt = []
    for name in ANSATZ_ORDER:
        entry = opt_results.get(name)
        if entry is None or "hist" not in entry:
            missing_this_opt.append(name)
            continue
        hist = entry["hist"]
        ax.plot(range(len(hist)), hist, label=name,
                color=ansatz_colors[name], linewidth=1.2, alpha=0.85)

    n_missing_total += len(missing_this_opt)
    ax.axhline(y=E_classical_ising, color="black", linestyle=":", linewidth=1.5)
    title = opt_name
    n_done = len(ANSATZ_ORDER) - len(missing_this_opt)
    if n_done < len(ANSATZ_ORDER):
        title += f"  ({n_done}/{len(ANSATZ_ORDER)} ansatzes)"
    if n_done == 0:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray", style="italic")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Objective evaluations", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r"$E_{min}$", fontsize=13)
for ax in reversed(axes):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, ncol=1)
        break

subtitle = "Convergence: 9 Ansatz Configurations x 3 Optimizers (IBM Cairo noise model)"
if n_missing_total:
    subtitle += "  [partial run]"
plt.suptitle(subtitle + "\n[Replication of Figure 3, Buonaiuto et al. 2023]",
             fontsize=12, y=1.03)
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/noisy_convergence.png", bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

rows = []
for opt_name in OPTIMIZER_ORDER:
    for name in ANSATZ_ORDER:
        data = results_ansatz_opt_noisy.get(opt_name, {}).get(name)
        label = f"{name} / {opt_name}"
        if data is None:
            rows.append((label, None, float("nan"), 0))
        else:
            rows.append((label, data["decode"],
                         float(np.real(data["result"].eigenvalue)), len(data["hist"])))
print_run_table(rows, header="Ansatz / Optimizer [Cairo]")

if n_missing_total:
    print(f"\n[INFO] {n_missing_total} ansatz/optimizer combo(s) not computed "
          f"-- shown as '--' rather than erroring.")
if topology_audit:
    worst = max(topology_audit.items(), key=lambda kv: kv[1]["two_qubit_overhead"])
    print(f"\n[CAVEAT] no coupling map here: '{worst[0]}' would need "
          f"{worst[1]['two_qubit_overhead']:.1f}x more two-qubit gates on Cairo "
          f"(topology audit, section 12.1).")
```

    
    ================================================================================================================
    Ansatz / Optimizer [Cairo]             E_vqe   E/E*   obj(vqe)   obj(mean)   obj(top)  match   top%  evals
    ----------------------------------------------------------------------------------------------------------------
    PauliTwo / Cobyla                    -17.858  0.857   0.001605     -2.8850    -0.1357      -   2.4%    100
    TwoLocal linear / Cobyla             -16.932  0.812   0.001473     -3.9693    -2.7042      -   1.5%    122
    TwoLocal full / Cobyla               -11.700  0.561   0.001534     -8.9767    -7.1840      -   0.5%    122
    TwoLocal circular / Cobyla           -14.326  0.687   0.001626     -6.3981    -2.2691    vqe   1.1%    122
    TwoLocal pairwise / Cobyla           -17.671  0.848   0.001534     -3.3349    -3.7267      -   5.3%    122
    RealAmplit. linear / Cobyla          -15.280  0.733   0.001626     -5.7067    -1.2975    vqe   0.8%    100
    RealAmplit. full / Cobyla            -15.858  0.761   0.001605     -5.0514    -0.9946      -   1.4%    100
    RealAmplit. circular / Cobyla        -11.586  0.556   0.001626     -9.2450    -0.0928    vqe   1.3%    100
    RealAmplit. pairwise / Cobyla        -16.365  0.785   0.001626     -4.4461    -0.0041    vqe   0.6%    100
    PauliTwo / SPSA                      -17.869  0.857   0.001534     -3.1286    -0.3149      -   1.4%    151
    TwoLocal linear / SPSA               -18.067  0.867   0.001626     -2.7532    -0.0484    vqe   1.4%    151
    TwoLocal full / SPSA                 -15.340  0.736   0.001605     -5.7096    -4.5545      -   0.7%    151
    TwoLocal circular / SPSA             -17.509  0.840   0.001626     -3.4235    -0.3409    vqe   0.9%    151
    TwoLocal pairwise / SPSA             -18.333  0.879   0.001626     -2.5177    -0.0371    vqe   0.6%    151
    RealAmplit. linear / SPSA            -16.322  0.783   0.001626     -4.4877    -0.2375    vqe   2.4%    151
    RealAmplit. full / SPSA              -15.979  0.767   0.001626     -4.8446     0.0007    vqe   0.7%    151
    RealAmplit. circular / SPSA          -14.480  0.695   0.001626     -6.3503    -6.1638    vqe   1.3%    151
    RealAmplit. pairwise / SPSA          -17.007  0.816   0.001626     -3.8113    -0.0110    vqe   1.6%    151
    PauliTwo / NFT                       -19.714  0.946   0.001626     -1.1242    -1.6827    vqe   5.3%    103
    TwoLocal linear / NFT                -19.323  0.927   0.001626     -1.4847    -1.6827    vqe   1.6%    103
    TwoLocal full / NFT                  -15.305  0.734   0.001626     -5.2725    -0.1154    vqe   0.4%    103
    TwoLocal circular / NFT              -19.064  0.915   0.001626     -1.8200    -0.8106    vqe   5.0%    103
    TwoLocal pairwise / NFT              -19.341  0.928   0.001626     -1.4198    -1.6827    vqe   2.9%    103
    RealAmplit. linear / NFT             -15.971  0.766   0.001626     -4.8026     0.0010    vqe   1.4%    103
    RealAmplit. full / NFT               -16.011  0.768   0.001626     -4.6200    -2.6751    vqe   1.9%    103
    RealAmplit. circular / NFT           -11.014  0.528   0.001626     -9.6104    -5.9261    vqe   1.6%    103
    RealAmplit. pairwise / NFT           -17.866  0.857   0.001626     -3.1749    -4.0541    vqe   2.0%    103
    ----------------------------------------------------------------------------------------------------------------
    Classical optimum                    -20.845  1.000   0.001626
    
    [CAVEAT] no coupling map here: 'TwoLocal full' would need 4.1x more two-qubit gates on Cairo (topology audit, section 12.1).


<div dir="rtl" class="farsi-text">

## ۱۴. Efficient Frontier کلاسیک (مقایسه اضافی)

نمایش مرز کارا (Efficient Frontier) برای مقایسه جواب‌های مختلف با جواب بهینه.

این نمودار همان ابر تخصیص‌های تصادفی و همان نقطه‌ی مرجع کلاسیکِ محاسبه‌شده برای شکل λ (بخش ۱۲.۳) را دوباره به کار می‌برد — با seed و sampler یکسان، پس تولید دوباره‌ی آن‌ها تنها یک رونوشت مو‌به‌مو می‌ساخت. جواب هر بهینه‌ساز نیز، مانند همان شکل، در میانگین وزن‌دار روی توزیع کامل شات‌ها رسم می‌شود.

</div>


```python
# ---- Plot Efficient Frontier (Section 2.2 / Figure 5 analogue) ----
print(f"Random fully-invested portfolios sampled: {len(cloud_vols)}")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(cloud_vols, cloud_rets, alpha=0.2, s=12, color="lightblue",
           label="Random allocations (fully invested)", zorder=1)

# Classical optimal
ax.scatter([v_classical], [r_classical], s=250, color="black", marker="*",
           zorder=5, label="Classical optimal (brute-force)", linewidths=1.5)

# VQE solutions, one per optimizer, at the shot-weighted mean of their output.
colors_pts = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, (opt_name, data) in enumerate(list(results_optimizer.items())[:3]):
    # Whether the reported solution equals b* goes in the label, not the marker.
    dec = data["decode"]
    suffix = " \u2713" if dec["match_vqe"] else ""
    ax.scatter([dec["vol_mean"]], [dec["ret_mean"]], s=150, color=colors_pts[k],
               marker="o", zorder=4, label=f"VQE ({opt_name}){suffix}", linewidths=2)

ax.set_xlabel("Portfolio Volatility (σ)", fontsize=12)
ax.set_ylabel("Portfolio Expected Return (μ)", fontsize=12)
ax.set_title("Markowitz Efficient Frontier — Original Portfolio Space\n"
             "[Figure 5 analogue, Buonaiuto et al. 2023]", fontsize=13)
ax.legend(fontsize=10, title="\u2713 = reported solution equals b*", title_fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig(f"{PLOT_DIR}/efficient_frontier.png", bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)
```

    Random fully-invested portfolios sampled: 30000


<div dir="rtl" class="farsi-text">

## ۱۵. ذخیره‌سازی نتایج در فایل yaml

 وضعیت کامل این اجرا — تمام پارامترهای ورودی و خروجی هر بخش بالا — در یک فایل YAML خوانا ذخیره می‌شود، برای بازبینی یا مقایسه با اجراهای دیگر بدون نیاز به اجرای مجدد کل نوت‌بوک.

</div>


```python
# ---- Save complete run state to a single readable file ----

g = globals()
missing_names = []       # human-readable notes on what wasn't found
section_errors = {}      # section_name -> error string, for anything that raised


def _to_native(obj):
    """Recursively convert numpy/pandas/complex values into plain
    YAML-safe Python types. Anything unrecognized (Qiskit circuits,
    results, samplers, ...) becomes a short descriptive string instead
    of crashing the dump."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else str(v)
    if isinstance(obj, (complex, np.complexfloating)):
        c = complex(obj)
        return float(c.real) if abs(c.imag) < 1e-9 else {"re": float(c.real), "im": float(c.imag)}
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.isoformat()
    return f"<{type(obj).__name__}: {str(obj)[:120]}>"


def _safe(section_name, builder):
    """Run a section-builder; on failure, record the error and move on
    rather than losing the whole file over one bad section."""
    try:
        return builder()
    except Exception as e:
        section_errors[section_name] = f"{type(e).__name__}: {e}"
        return None


def _summarize_history(hist):
    if not hist:
        return {"n_iters": 0}
    arr = np.asarray(hist, dtype=float)
    out = {
        "n_iters": int(len(arr)),
        "first_energy": float(arr[0]),
        "last_energy": float(arr[-1]),
        "min_energy": float(arr.min()),
        "min_at_iter": int(arr.argmin()),
    }
    if INCLUDE_FULL_HISTORIES:
        out["energy_history"] = [float(x) for x in arr]
    return out


def _describe_ansatz(ansatz):
    if ansatz is None:
        return None
    fp = g.get("_ansatz_fingerprint")
    if callable(fp):
        try:
            return _to_native(fp(ansatz))
        except Exception:
            pass
    return {"class": type(ansatz).__name__, "num_qubits": getattr(ansatz, "num_qubits", None)}


def _describe_optimizer(optimizer):
    if optimizer is None:
        return None
    fp = g.get("_optimizer_fingerprint")
    if callable(fp):
        try:
            return _to_native(fp(optimizer))
        except Exception:
            pass
    return {"class": type(optimizer).__name__}


def _describe_sampler(sampler):
    if sampler is None:
        return None
    fp = g.get("_sampler_fingerprint")
    if callable(fp):
        try:
            return _to_native(fp(sampler))
        except Exception:
            pass
    return {"class": type(sampler).__name__}


def _describe_noise_model(nm):
    if nm is None:
        return None
    try:
        d = nm.to_dict(serializable=True)
        return {
            "class": type(nm).__name__,
            "source": g.get("cairo_noise_source"),
            "num_error_channels": len(d.get("errors", [])),
            "basis_gates": _to_native(getattr(nm, "basis_gates", None)),
            "coupling_map_applied_to_simulator": False,
        }
    except Exception as e:
        return {"class": type(nm).__name__, "note": f"introspection failed: {e}"}


# decode_vqe_result() scalars, fixed order so successive run_states diff cleanly.
_READOUT_SCALARS = (
    "shots", "n_distinct", "top_count", "runner_up_count",
    "obj_top", "obj_best", "obj_vqe", "obj_mean",
    "obj_ref_top", "obj_ref_best", "obj_ref_vqe", "obj_ref_mean",
    "budget_top", "budget_best", "budget_vqe", "budget_mean",
    "ret_top", "ret_best", "ret_vqe", "ret_mean",
    "vol_top", "vol_best", "vol_vqe", "vol_mean",
    "match_top", "match_best", "match_vqe",
)


def _capture_readout(decode):
    """Flatten one decode_vqe_result() dict into the YAML. All three read-out
    rules are recorded side by side; see section 9.1 for why they differ."""
    if not decode:
        return None
    out = {k: _to_native(decode[k]) for k in _READOUT_SCALARS if k in decode}
    for tag in ("top", "best", "vqe"):
        if f"b_{tag}" in decode:
            out[f"bits_{tag}"] = _to_native(decode[f"b_{tag}"])
            out[f"units_{tag}"] = _to_native(decode[f"n_{tag}"])
    return out


def _capture_run_entry(entry):
    """Common shape shared by results_ansatz_opt[*][*], results_optimizer[*],
    results_lambda[*], results_ansatz_opt_noisy[*][*]."""
    if not entry:
        return None
    out = {}
    if "elapsed" in entry:
        out["elapsed_seconds"] = _to_native(entry["elapsed"])
    if "eigenvalue" in entry:
        out["eigenvalue"] = _to_native(entry["eigenvalue"])
    elif entry.get("result") is not None:
        try:
            out["eigenvalue"] = _to_native(entry["result"].eigenvalue)
        except Exception:
            pass
    if out.get("eigenvalue") is not None and g.get("E_classical_ising"):
        out["energy_ratio_vs_classical"] = float(out["eigenvalue"]) / float(g["E_classical_ising"])
    if "decode" in entry:
        out["readout"] = _capture_readout(entry["decode"])
    if "hist" in entry:
        out["convergence"] = _summarize_history(entry["hist"])
    if "constraint_ok" in entry:
        out["budget_constraint_satisfied"] = _to_native(entry["constraint_ok"])
    if INCLUDE_FULL_HISTORIES and entry.get("result") is not None:
        try:
            out["optimal_parameters"] = _to_native(entry["result"].optimal_parameters)
        except Exception:
            pass
    return out


def _capture_flat_sweep(flat_dict):
    return {str(k): _capture_run_entry(v) for k, v in flat_dict.items()}


def _capture_nested_sweep(nested_dict):
    return {
        str(opt_name): {str(a): _capture_run_entry(e) for a, e in ansatz_dict.items()}
        for opt_name, ansatz_dict in nested_dict.items()
    }


def _pkg_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


# ---- Assemble the state dict, section by section ----
state = {}

state["meta"] = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "paper": ("Buonaiuto, G., Gargiulo, F., De Pietro, G., Esposito, M., & Pota, M. (2023). "
              "Best practices for portfolio optimization by quantum computing, experimented "
              "on real quantum devices. Scientific Reports, 13, 19434. "
              "https://doi.org/10.1038/s41598-023-45392-w"),
    "note": ("Auto-generated snapshot of this notebook's configuration and results. "
             "Best-effort: sections not executed in this kernel session are listed "
             "under missing_top_level_names / section_errors instead of raising."),
}

state["environment"] = _safe("environment", lambda: {
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "in_colab": bool(g.get("IN_COLAB", False)),
    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    "packages": {
        pkg: _pkg_version(pkg)
        for pkg in ["qiskit", "qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11", "qiskit-algorithms",
                    "qiskit-optimization", "qiskit-ibm-runtime", "numpy", "pandas",
                    "matplotlib", "scipy", "joblib", "yfinance", "pyyaml"]
    },
})

state["config"] = _safe("config", lambda: {
    "tickers": g.get("TICKERS"),
    "start_date": g.get("START_DATE"),
    "end_date": g.get("END_DATE"),
    "end_date_is_exclusive": True,
    "price_column": "Corrected_Adj_Close",
    "budget": g.get("BUDGET"),
    "risk_aversion_q": g.get("RISK_Q"),
    "penalty_lambda": g.get("LAMBDA"),
    "random_seed": g.get("RANDOM_SEED"),
    "vqe_shots": g.get("VQE_SHOTS"),
    "default_reps": g.get("DEFAULT_REPS"),
    "default_entanglement": g.get("DEFAULT_ENTANGLEMENT"),
    "default_maxiter": g.get("DEFAULT_MAXITER"),
    "ansatz_maxiter": g.get("ANSATZ_MAXITER"),
    "lambda_maxiter": g.get("lambda_maxiter"),
    "spsa_calibration_evals": g.get("SPSA_CALIBRATION_EVALS"),
    "budget_tolerance": g.get("BUDGET_TOL"),
    "soft_match_tolerance": g.get("SOFT_MATCH_TOL"),
    "n_random_portfolios": g.get("N_RANDOM_PORTFOLIOS"),
    "default_progress_every": g.get("DEFAULT_PROGRESS_EVERY"),
    "ansatz_progress_every": g.get("ANSATZ_PROGRESS_EVERY"),
    "price_data_dir": g.get("PRICE_DATA_DIR"),
    "cache_dir": g.get("CACHE_DIR"),
    "plot_dir": g.get("PLOT_DIR"),
    "show_plots": g.get("SHOW_PLOTS"),
    "save_plots": g.get("SAVE_PLOTS"),
})

if "prices_df" in g:
    state["market_data"] = _safe("market_data", lambda: {
        "tickers": _to_native(g.get("TICKERS")),
        "n_trading_days": int(len(g["prices_df"])),
        "date_range": [g["prices_df"].index.min().date().isoformat(),
                        g["prices_df"].index.max().date().isoformat()],
        "last_close_prices": {c: float(g["prices_df"][c].iloc[-1]) for c in g["prices_df"].columns},
        "note": ("Full daily series intentionally not embedded -- already persisted "
                  "verbatim in <TICKER>.csv under price_data_dir (see config). Prices are "
                  "the split/dividend-corrected adjusted closes, not the raw Close."),
        "N_assets": _to_native(g.get("N")),
        "P_current_prices": _to_native(g.get("P")),
        "mu_expected_daily_returns": _to_native(g.get("mu")),
        "Sigma_covariance": _to_native(g.get("Sigma")),
    })
else:
    missing_names.append("prices_df / P / mu / Sigma (section 3 not run)")

if "P_prime" in g:
    state["integer_formulation"] = _safe("integer_formulation", lambda: {
        "P_prime": _to_native(g.get("P_prime")),
        "mu_prime": _to_native(g.get("mu_prime")),
        "Sigma_prime": _to_native(g.get("Sigma_prime")),
    })
else:
    missing_names.append("P_prime / mu_prime / Sigma_prime (section 4 not run)")

if "C" in g:
    state["binary_encoding"] = _safe("binary_encoding", lambda: {
        "n_max_per_asset": _to_native(g.get("n_max")),
        "d_i_per_asset": _to_native(g.get("d_list")),
        "dim_b_total_qubits": _to_native(g.get("dim_b")),
        "paper_reported_qubits": _to_native(g.get("PAPER_QUBIT_COUNT")),
        "budgets_reproducing_paper_qubit_count": _to_native(
            [g["budgets_matching_paper"][0], g["budgets_matching_paper"][-1]]
            if g.get("budgets_matching_paper") else None),
        "encoding_matrix_C": _to_native(g.get("C")),
        "mu_pp": _to_native(g.get("mu_pp")),
        "Sigma_pp": _to_native(g.get("Sigma_pp")),
        "P_pp": _to_native(g.get("P_pp")),
    })
else:
    missing_names.append("C / dim_b / mu_pp / Sigma_pp / P_pp (section 5 not run)")

if "qp" in g or "ising_op" in g:
    state["qubo_and_hamiltonian"] = _safe("qubo_and_hamiltonian", lambda: {
        "qubo_num_binary_vars": g["qp"].get_num_vars() if g.get("qp") is not None else None,
        "ising_num_qubits": getattr(g.get("ising_op"), "num_qubits", None),
        "ising_num_pauli_terms": len(g["ising_op"]) if g.get("ising_op") is not None else None,
        "ising_energy_offset": _to_native(g.get("offset")),
        "hamiltonian_fingerprint": (
            g["_hamiltonian_fingerprint"](g["ising_op"])
            if g.get("ising_op") is not None and callable(g.get("_hamiltonian_fingerprint"))
            else None
        ),
    })
else:
    missing_names.append("qp / ising_op (sections 6-7 not run)")

if "b_opt_classical" in g:
    state["classical_benchmark"] = _safe("classical_benchmark", lambda: {
        "optimal_bitstring": _to_native(g.get("b_opt_classical")),
        "optimal_units_per_asset": (
            dict(zip(g.get("TICKERS", []), _to_native(g.get("n_opt"))))
            if g.get("n_opt") is not None else None
        ),
        "objective_value": _to_native(g.get("val_opt_classical")),
        "ising_energy": _to_native(g.get("E_classical_ising")),
        "brute_force_search_time_seconds": _to_native(g.get("t_classical")),
        "total_spend_usd": _to_native(g.get("total_spend")),
        "search_space_size_2pow_dimb": (2 ** g["dim_b"]) if "dim_b" in g else None,
    })
else:
    missing_names.append("b_opt_classical (section 8 not run)")

if "result_ideal" in g:
    def _build_vqe_ideal():
        out = {
            "ansatz": _describe_ansatz(g.get("ansatz_default")),
            "optimizer": _describe_optimizer(g.get("cobyla_default")),
            "sampler": _describe_sampler(g.get("ideal_sampler")),
            "n_evaluations": len(g["hist_ideal"]) if "hist_ideal" in g else None,
            "elapsed_seconds": _to_native(g.get("t_ideal")),
            "eigenvalue": _to_native(g["result_ideal"].eigenvalue),
            "readout": _capture_readout(g.get("decode_ideal")),
            "convergence": _summarize_history(g.get("hist_ideal", [])),
        }
        if g.get("E_classical_ising"):
            out["energy_ratio_vs_classical"] = (
                float(np.real(g["result_ideal"].eigenvalue)) / float(g["E_classical_ising"]))
        if INCLUDE_FULL_HISTORIES:
            try:
                out["optimal_parameters"] = _to_native(g["result_ideal"].optimal_parameters)
            except Exception:
                pass
        return out
    state["vqe_ideal_single_run"] = _safe("vqe_ideal_single_run", _build_vqe_ideal)
else:
    missing_names.append("result_ideal (section 9 not run)")

if "results_ansatz_opt" in g:
    state["ansatz_optimizer_sweep_noiseless"] = _safe("ansatz_optimizer_sweep_noiseless", lambda: {
        "ansatz_configs": {n: _describe_ansatz(a) for n, a in g.get("ansatz_configs", {}).items()},
        "ansatz_order": _to_native(g.get("ANSATZ_ORDER")),
        "optimizer_order": _to_native(g.get("OPTIMIZER_ORDER")),
        "calibration_evaluations": _to_native(g.get("calibration_evals")),
        "results": _capture_nested_sweep(g["results_ansatz_opt"]),
    })
else:
    missing_names.append("results_ansatz_opt (section 10 not run)")

if "results_optimizer" in g:
    state["optimizer_sweep_noiseless"] = _safe(
        "optimizer_sweep_noiseless", lambda: _capture_flat_sweep(g["results_optimizer"])
    )
else:
    missing_names.append("results_optimizer (section 11 not run)")

if "results_lambda" in g:
    def _build_lambda_sweep():
        lam_out = {}
        frontier = g.get("lambda_frontier_points", {})
        for lam, entry in g["results_lambda"].items():
            captured = _capture_run_entry(entry)
            fp = frontier.get(lam)
            if fp:
                captured["frontier_return"] = _to_native(fp.get("ret"))
                captured["frontier_volatility"] = _to_native(fp.get("vol"))
                captured["frontier_soft_match"] = _to_native(fp.get("soft_match"))
            lam_out[str(lam)] = captured
        return {
            "lambda_values_tested": _to_native(g.get("lambda_values")),
            "reference_lambda_for_comparison": _to_native(g.get("LAMBDA")),
            "noisy_sampler": _describe_sampler(g.get("noisy_sampler")),
            "results": lam_out,
            "feasible_lambdas": _to_native(g.get("feasible_lambdas")),
            "best_lambda_by_objective": _to_native(g.get("best_lambda_by_objective")),
            "best_lambda_by_budget": _to_native(g.get("best_lambda_by_budget")),
        }
    state["penalty_lambda_sweep"] = _safe("penalty_lambda_sweep", _build_lambda_sweep)
else:
    missing_names.append("results_lambda (section 12 not run)")

if "results_ansatz_opt_noisy" in g:
    state["ansatz_optimizer_sweep_noisy"] = _safe("ansatz_optimizer_sweep_noisy", lambda: {
        "optimizers_actually_run": _to_native(g.get("OPTIMIZER_ORDER_NOISY")),
        "noise_model": _describe_noise_model(g.get("cairo_nm")),
        "calibration_evaluations": _to_native(g.get("calibration_evals_noisy")),
        "failures": _to_native(g.get("noisy_failures")),
        "results": _capture_nested_sweep(g["results_ansatz_opt_noisy"]),
    })
else:
    missing_names.append("results_ansatz_opt_noisy (section 13 not run)")

if g.get("topology_audit"):
    state["topology_audit"] = _safe("topology_audit", lambda: {
        "note": ("Two-qubit gate count and depth of each ansatz as actually simulated "
                 "(no coupling map) versus routed onto the real IBM Cairo connectivity "
                 "graph. The noisy runs pay none of this routing cost, which is the "
                 "effect the paper attributes to hardware topology."),
        "results": _to_native(g["topology_audit"]),
    })
else:
    missing_names.append("topology_audit (diagnostic skipped or coupling map unavailable)")

if "cloud_vols" in g:
    def _build_frontier():
        vols = np.asarray(g["cloud_vols"], dtype=float)
        rets = np.asarray(g["cloud_rets"], dtype=float)
        return {
            "n_random_portfolios_sampled": int(len(vols)),
            "sampling_seed": _to_native(g.get("RANDOM_SEED")),
            "note": ("Raw per-point cloud not embedded (tens of thousands of numbers); "
                      "fully reproducible via "
                      "np.random.default_rng(RANDOM_SEED).dirichlet(np.ones(N), size=...) "
                      "as coded in this notebook. Saved PNGs show the full cloud."),
            "return_stats": {"min": float(rets.min()), "max": float(rets.max()),
                              "mean": float(rets.mean()), "std": float(rets.std())},
            "volatility_stats": {"min": float(vols.min()), "max": float(vols.max()),
                                  "mean": float(vols.mean()), "std": float(vols.std())},
            "classical_optimal_return": _to_native(g.get("r_classical")),
            "classical_optimal_volatility": _to_native(g.get("v_classical")),
        }
    state["efficient_frontier_monte_carlo"] = _safe("efficient_frontier_monte_carlo", _build_frontier)
else:
    missing_names.append("cloud_vols / cloud_rets (efficient-frontier cells not run)")

state["final_conclusions"] = _safe("final_conclusions", lambda: {
    "best_ansatz": g.get("best_ansatz_name"),
    "best_optimizer": g.get("best_opt_name"),
    "ranking_metric": "shot-weighted objective (obj_mean), i.e. the energy the optimizer minimised",
    "best_lambda_by_objective": _to_native(g.get("best_lambda_by_objective")),
    "best_lambda_by_budget": _to_native(g.get("best_lambda_by_budget")),
    "qubit_count_this_run": _to_native(g.get("dim_b")),
    "qubit_count_paper_reported": _to_native(g.get("PAPER_QUBIT_COUNT", 12)),
    "encoding_savings_vs_one_hot": (
        f"{g['dim_b']} qubits vs {sum(n + 1 for n in g['n_max'])} with one-hot"
        if "dim_b" in g and "n_max" in g else None
    ),
})


def _list_plot_files():
    plot_dir = g.get("PLOT_DIR")
    if not plot_dir or not os.path.isdir(plot_dir):
        return []
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(plot_dir, "*.png")))


state["artifacts"] = _safe("artifacts", lambda: {
    "plot_dir": g.get("PLOT_DIR"),
    "plot_files": _list_plot_files(),
    "price_data_dir": g.get("PRICE_DATA_DIR"),
    "joblib_cache_dir": g.get("CACHE_DIR"),
})

state["missing_top_level_names"] = missing_names
state["section_errors"] = section_errors

# ---- Write out ----
os.makedirs(STATE_DIR, exist_ok=True)

_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
state_path = os.path.join(STATE_DIR, f"{_timestamp}.yaml")

with open(state_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(state, f, sort_keys=False, allow_unicode=True,
                    default_flow_style=False, width=100)

print(f"Run state saved to: {state_path}  ({os.path.getsize(state_path)/1024:.1f} KB)")
if missing_names:
    print(f"\n[INFO] {len(missing_names)} section(s) not captured (not run in this kernel session):")
    for m in missing_names:
        print(f"  - {m}")
if section_errors:
    print(f"\n[WARN] {len(section_errors)} section(s) hit an error while capturing:")
    for k, v in section_errors.items():
        print(f"  - {k}: {v}")
```

    Run state saved to: /content/drive/MyDrive/quantum-portfolio-optimization/run_states/20260828_000406.yaml  (639.0 KB)


<style>
	.farsi-text{
		direction: rtl;
		font-family: Vazirmatn;
		text-align: justify;
	}
	.jp-RenderedHTMLCommon p{
		text-align: unset;
	}
</style>
