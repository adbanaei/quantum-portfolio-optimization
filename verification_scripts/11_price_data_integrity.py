# Data-quality sweep over the price CSVs.
import pandas as pd

TICKERS = ["AAPL", "IBM", "NFLX", "TSLA"]
for t in TICKERS:
    df = pd.read_csv(f"./price_data/{t}.csv", index_col=0, parse_dates=True)
    print(f"\n{'='*60}\n{t}\n{'='*60}")
    print(f"  rows: {len(df)}   date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  duplicated dates: {df.index.duplicated().sum()}")
    print(f"  monotonic increasing: {df.index.is_monotonic_increasing}")
    print(f"  NaNs in Corrected_Adj_Close: {df['Corrected_Adj_Close'].isna().sum()}")
    print(f"  any non-positive prices: {(df['Corrected_Adj_Close'] <= 0).sum()}")

    # a >15% single-day move would suggest a residual, uncorrected adjustment
    pct = df["Corrected_Adj_Close"].pct_change()
    big_moves = pct[pct.abs() > 0.15]
    print(f"  single-day moves > 15% in Corrected_Adj_Close: {len(big_moves)}")
    if len(big_moves) > 0:
        for date, move in big_moves.items():
            print(f"    {date.date()}: {move*100:+.1f}%")

    ratio = df["Close"] / df["Corrected_Adj_Close"]
    print(f"  Close/Corrected_Adj_Close ratio: min={ratio.min():.4f} max={ratio.max():.4f}  "
          f"(1.0 = no adjustment; drift indicates cumulative dividend/split correction)")
    print(f"  ratio at START ({df.index.min().date()}): {ratio.iloc[0]:.4f}")
    print(f"  ratio at END   ({df.index.max().date()}): {ratio.iloc[-1]:.4f}")

# IBM around the Kyndryl spinoff
print(f"\n{'='*60}\nIBM around Kyndryl spinoff (2021-11-03)\n{'='*60}")
ibm = pd.read_csv("./price_data/IBM.csv", index_col=0, parse_dates=True)
window = ibm.loc["2021-10-25":"2021-11-10"]
print(window[["Close", "Adj Close", "Corrected_Adj_Close"]])
