import os, math, time
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import requests

# -------------------------
# Config
# -------------------------
BENCH = "^SP500-30"
PERIOD = "5y"
LOOKBACK = 20
SEQ = 20
HORIZON = 5
MIN_VOL = 250_000
LIQ_DAYS = 20
LIQ_PASS = 16
TRAIN_FRAC = 0.70
BATCH = 256
EPOCHS = 15
LR = 1e-3
WD = 1e-5
HIDDEN = 32
TOPN = 10
WEEK_RULE = "W-FRI"
TICKERS_CSV = "staples_tickers.csv"
OUT_CSV = "weekly_top10_rankings.csv"

# -------------------------
# Universe (S&P 500 staples)
# -------------------------
def _normalize_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def get_staples_tickers(cache_csv: str) -> list[str]:
    if os.path.exists(cache_csv):
        t = pd.read_csv(cache_csv)["Symbol"].astype(str).tolist()
        return sorted(list(dict.fromkeys([_normalize_symbol(x) for x in t])))

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # NOTE: pandas warns that passing literal HTML may be deprecated later,
    # but it's fine for now and not part of the tz-fix.
    sp = pd.read_html(resp.text)[0]

    sp["Symbol"] = sp["Symbol"].astype(str).map(_normalize_symbol)
    staples = sp[sp["GICS Sector"] == "Consumer Staples"][["Symbol"]].copy()
    staples.to_csv(cache_csv, index=False)
    return sorted(staples["Symbol"].tolist())

def adj_close_or_close(df: pd.DataFrame) -> pd.Series:
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

# -------------------------
# Data download
# -------------------------
def download_histories(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    out = {}
    for i, t in enumerate(tickers, 1):
        try:
            h = yf.Ticker(t).history(period=period, auto_adjust=False).sort_index()
            if h is not None and not h.empty and "Volume" in h.columns:
                out[t] = h
        except Exception as e:
            print(f"[WARN] {t} download failed: {e}")

        if i % 25 == 0:
            time.sleep(0.4)
    return out

# -------------------------
# Features + target + liquidity gate
# -------------------------
def build_frames(hist_by_ticker: dict[str, pd.DataFrame], bench_hist: pd.DataFrame):
    prices = {}
    vols = {}
    for t, h in hist_by_ticker.items():
        prices[t] = adj_close_or_close(h).rename(t)
        vols[t] = h["Volume"].rename(t)

    px = pd.concat(prices.values(), axis=1).sort_index()
    vol = pd.concat(vols.values(), axis=1).sort_index()

    bpx = adj_close_or_close(bench_hist).rename("BENCH").sort_index()

    common = px.index.intersection(bpx.index)
    px = px.loc[common]
    vol = vol.loc[common]
    bpx = bpx.loc[common]
    return px, vol, bpx

def compute_features_long(px: pd.DataFrame, vol: pd.DataFrame, bpx: pd.Series) -> pd.DataFrame:
    r = px.pct_change()
    br = bpx.pct_change()

    r_ma = r.rolling(LOOKBACK).mean()
    br_ma = br.rolling(LOOKBACK).mean()

    r_sd = r.rolling(LOOKBACK).std()
    br_sd = br.rolling(LOOKBACK).std()

    mom = (r_ma.sub(br_ma, axis=0)) * 100.0
    relvol = ((r_sd.div(br_sd, axis=0)) - 1.0) * 100.0
    z = (r_ma.sub(br_ma, axis=0)).div(r_sd, axis=0)

    v_ma = vol.rolling(LOOKBACK).mean()
    v_sd = vol.rolling(LOOKBACK).std()
    vz = (vol - v_ma).div(v_sd)

    frames = []
    for t in px.columns:
        df = pd.DataFrame(
            {
                "mom_vs_sp": mom[t],
                "relvol_vs_sp": relvol[t],
                "zscore_vs_sp": z[t],
                "volume_z": vz[t],
                "Ticker": t,
            }
        )
        frames.append(df)

    long = pd.concat(frames, axis=0).reset_index().rename(columns={"index": "Date"})
    return long.set_index(["Date", "Ticker"]).sort_index()

def liquidity_gate(vol: pd.DataFrame) -> pd.DataFrame:
    meets = (vol >= MIN_VOL).astype(int)
    pass_ct = meets.rolling(LIQ_DAYS).sum()
    return pass_ct >= float(LIQ_PASS)

def target_excess(px: pd.DataFrame, bpx: pd.Series) -> pd.DataFrame:
    fwd_px = px.shift(-HORIZON)
    fwd_b = bpx.shift(-HORIZON)
    r5 = (fwd_px / px) - 1.0
    br5 = (fwd_b / bpx) - 1.0
    return r5.sub(br5, axis=0)

# -------------------------
# Sequences: (20,4) -> y
# -------------------------
FEATURES = ["mom_vs_sp", "relvol_vs_sp", "zscore_vs_sp", "volume_z"]

def build_xy(features_long: pd.DataFrame, y_df: pd.DataFrame, elig: pd.DataFrame):
    Xs, ys, end_dates, tickers = [], [], [], []

    for t in y_df.columns:
        try:
            f = features_long.xs(t, level="Ticker")[FEATURES].copy()
        except KeyError:
            continue

        common = f.index.intersection(y_df.index).intersection(elig.index)
        f = f.loc[common]
        y = y_df.loc[common, t]
        e = elig.loc[common, t]

        if len(common) < SEQ + 1:
            continue

        fv = f.values
        yv = y.values
        ev = e.values

        for end in range(SEQ - 1, len(common)):
            if not bool(ev[end]):
                continue
            if not np.isfinite(yv[end]):
                continue

            start = end - (SEQ - 1)
            window = fv[start : end + 1, :]
            if not np.all(np.isfinite(window)):
                continue

            Xs.append(window.astype(np.float32))
            ys.append(float(yv[end]))
            end_dates.append(np.datetime64(common[end].date()))
            tickers.append(t)

    if not Xs:
        raise RuntimeError("No samples created. Check data, liquidity gate, or lookback windows.")

    return np.stack(Xs), np.array(ys, dtype=np.float32), np.array(end_dates), tickers

def fit_scaler(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu = flat.mean(axis=0, keepdims=True).astype(np.float32)
    sd = flat.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return mu.reshape(1, 1, -1), sd.reshape(1, 1, -1)

# -------------------------
# PyTorch bits
# -------------------------
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMReg(nn.Module):
    def __init__(self, input_size=4, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

def train_once(model, train_ld, test_ld, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    model.to(device)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr = []
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr.append(loss.item())

        model.eval()
        te = []
        with torch.no_grad():
            for Xb, yb in test_ld:
                Xb, yb = Xb.to(device), yb.to(device)
                te.append(loss_fn(model(Xb), yb).item())

        print(f"Epoch {ep:02d}/{EPOCHS} | Train MSE {np.mean(tr):.6f} | Test MSE {np.mean(te):.6f}")

# -------------------------
# Weekly ranking
# -------------------------
def weekly_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dummy = pd.Series(1, index=trading_dates)
    return dummy.resample(WEEK_RULE).last().dropna().index

@torch.no_grad()
def rank_weekly_top10(model, features_long, elig, mu, sd, dates, tickers, device):
    model.eval()
    model.to(device)

    feat_by_t = {}
    for t in tickers:
        try:
            feat_by_t[t] = features_long.xs(t, level="Ticker")[FEATURES].copy()
        except KeyError:
            pass

    rows = []
    for d in dates:
        preds = []
        for t in tickers:
            if t not in feat_by_t:
                continue
            if d not in elig.index or t not in elig.columns:
                continue
            if not bool(elig.at[d, t]):
                continue

            f = feat_by_t[t]
            if d not in f.index:
                continue

            end = f.index.get_indexer([d])[0]
            start = end - (SEQ - 1)
            if start < 0:
                continue

            X = f.iloc[start : end + 1].values.astype(np.float32)
            if X.shape != (SEQ, 4) or not np.all(np.isfinite(X)):
                continue

            X = ((X.reshape(1, SEQ, 4) - mu) / sd).astype(np.float32)
            pred = model(torch.from_numpy(X).to(device)).cpu().numpy().reshape(-1)[0]
            preds.append((t, float(pred)))

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:TOPN]

        print(f"\n{pd.Timestamp(d).date()} Top {TOPN}:")
        for i, (t, p) in enumerate(top, 1):
            print(f"  {i:02d}. {t:>6} | pred_excess_5d = {p:+.6f}")
            rows.append({"Date": pd.Timestamp(d).date(), "Rank": i, "Ticker": t, "PredictedExcessReturn": p})

    return pd.DataFrame(rows)

# -------------------------
# Evaluation (TZ-AWARE FIXED)
# -------------------------
def evaluate_weekly_predictions(ranks: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    For each test Friday Date in ranks:
      - collect the realized 5-day excess returns from y for the Top-10 tickers
      - compute average realized excess return and hit rate

    Fix: y.index may be tz-aware; localize the evaluation dates to match y's tz.
    """

    y_by_date = y.copy()
    y_by_date.index = pd.to_datetime(y_by_date.index).normalize()
    y_tz = getattr(y_by_date.index, "tz", None)

    rows = []

    for d, g in ranks.groupby("Date"):
        d_norm = pd.Timestamp(d).normalize()

        # IMPORTANT FIX: make d_norm tz-aware if y index is tz-aware
        if y_tz is not None and d_norm.tzinfo is None:
            d_norm = d_norm.tz_localize(y_tz)

        if d_norm not in y_by_date.index:
            continue

        realized_vals = []
        for _, r in g.iterrows():
            t = r["Ticker"]
            if t in y_by_date.columns:
                val = y_by_date.at[d_norm, t]
                if np.isfinite(val):
                    realized_vals.append(float(val))

        if not realized_vals:
            continue

        realized_arr = np.array(realized_vals, dtype=float)

        rows.append({
            "Date": d_norm.date(),
            "N": int(len(realized_arr)),
            "AvgRealizedExcessReturn": float(realized_arr.mean()),
            "HitRate": float((realized_arr > 0.0).mean()),
        })

    if not rows:
        print("\n===== WEEKLY TOP-10 EVALUATION (TEST PERIOD) =====")
        print("No weeks could be evaluated (date mismatch or missing realized values).")
        return pd.DataFrame(columns=["Date", "N", "AvgRealizedExcessReturn", "HitRate"])

    eval_df = pd.DataFrame(rows).sort_values("Date")

    print("\n===== WEEKLY TOP-10 EVALUATION (TEST PERIOD) =====")
    print(f"Weeks evaluated: {len(eval_df)}")
    print(f"Average weekly realized excess return: {eval_df['AvgRealizedExcessReturn'].mean():+.4%}")
    print(f"Average hit rate: {eval_df['HitRate'].mean():.2%}")

    return eval_df

# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tickers = get_staples_tickers(TICKERS_CSV)
    print(f"Staples tickers: {len(tickers)}")

    hist_by = download_histories(tickers, PERIOD)
    tickers_ok = sorted(hist_by.keys())
    print(f"Downloaded: {len(tickers_ok)}")
    if len(tickers_ok) < 10:
        raise RuntimeError("Too few tickers downloaded to rank Top 10.")

    bench_hist = yf.Ticker(BENCH).history(period=PERIOD, auto_adjust=False)
    if bench_hist is None or bench_hist.empty:
        raise RuntimeError(f"Failed to download benchmark {BENCH}")

    px, vol, bpx = build_frames(hist_by, bench_hist)

    feats_long = compute_features_long(px, vol, bpx)
    elig = liquidity_gate(vol)
    y = target_excess(px, bpx)

    X, yv, end_dates, _ = build_xy(feats_long, y, elig)
    print(f"Samples: {X.shape[0]}")

    udates = np.sort(np.unique(end_dates))
    cut_i = int(math.floor(len(udates) * TRAIN_FRAC))
    cut_i = max(1, min(cut_i, len(udates) - 1))
    cut_date = udates[cut_i]
    print(f"Cutoff end-date: {cut_date}")

    tr = end_dates < cut_date
    te = ~tr
    Xtr, ytr = X[tr], yv[tr]
    Xte, yte = X[te], yv[te]
    print(f"Train: {Xtr.shape[0]} | Test: {Xte.shape[0]}")

    mu, sd = fit_scaler(Xtr)
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)

    train_ld = DataLoader(SeqDS(Xtr, ytr), batch_size=BATCH, shuffle=True)
    test_ld = DataLoader(SeqDS(Xte, yte), batch_size=BATCH, shuffle=False)

    model = LSTMReg(input_size=4, hidden=HIDDEN)
    train_once(model, train_ld, test_ld, device)

    rdates = weekly_dates(px.index)

    # -------------------------
    # TZ fix for your earlier comparison error
    # -------------------------
    cutoff_ts = pd.Timestamp(pd.to_datetime(str(cut_date)).date())
    if getattr(rdates, "tz", None) is not None and cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize(rdates.tz)

    rdates_test = rdates[rdates >= cutoff_ts]
    print(f"Ranking weeks (test): {len(rdates_test)}")

    ranks = rank_weekly_top10(model, feats_long, elig, mu, sd, rdates_test, list(px.columns), device)
    ranks.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # -------------------------
    # Evaluation (ADDED)
    # -------------------------
    eval_df = evaluate_weekly_predictions(ranks, y)
    eval_df.to_csv("weekly_top10_evaluation.csv", index=False)
    print("Saved: weekly_top10_evaluation.csv")

if __name__ == "__main__":
    main()