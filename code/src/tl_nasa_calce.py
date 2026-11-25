# %% [markdown]
# # Transfer Learning for SOH and RUL: NASA (source) -> CALCE CS2 (target)
# LSTM, GRU, Transformer, CNN-LSTM, CNN-GRU CNN-GRU-RC and the proposed GRU-SDR (GRU + SOH-derived RUL) are pretrained on NASA cells and fine-tuned with a small part of one
# CALCE cell, then tested on unseen CALCE cells. Run the cells in order; edit Cell 1 only.

# %%
# ===== Cell 1: Connect Google Drive and find the data =====
import os, re, glob, time, json, zipfile, shutil, warnings
warnings.filterwarnings("ignore")

# 1) Connect Google Drive (accept the popup)
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        shutil.rmtree("/content/drive", ignore_errors=True)   # only a local leftover folder, Drive is not connected
        drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DRIVE = "/content/drive/MyDrive" if IN_COLAB else os.getcwd()
OUT_DIR = os.environ.get("TL_OUT", os.path.join(DRIVE, "TL_NASA_CALCE_Results"))

NASA_CELLS  = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]   # all 1C discharge
FT_CELL     = "CS2_35"                                   # cell used for fine-tuning
EOL         = 0.80          # end of life at 80% SOH
SEQ         = 10            # cycles per input window
FRACTIONS   = [0.10, 0.25, 0.50, 1.00]   # share of the fine-tuning cell's life used
SEEDS       = [42, 7, 123]
EPOCHS_SRC, EPOCHS_TGT, BATCH, PATIENCE = 150, 150, 16, 15
RC_GAP, RC_LAMBDA = 10, 0.05   # proposed CNN-GRU-RC: pair windows 10 cycles apart, weight of the RUL-consistency term
RESUME = True                  # keep models already saved in OUT_DIR, train only the missing ones

# 2) Find the datasets anywhere in your Drive
def _search(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(DRIVE, "**", pat), recursive=True))
        hits = [h for h in hits if "TL_NASA_CALCE_Results" not in h]
        if hits:
            return hits[0]
    return None

hit = _search(["BatteryAgingARC*.zip", "B0005.mat"])
assert hit, "NASA data not found in Google Drive (BatteryAgingARC-FY08Q4.zip or B0005.mat)."
NASA_PATH = os.path.dirname(hit) if hit.endswith(".mat") else hit

hit = _search(["CS2_35*.zip", "CS2_35*.xlsx"])
assert hit, "CALCE data not found in Google Drive (CS2_35.zip ... CS2_38.zip)."
CALCE_PATH = os.path.dirname(hit)
if os.path.basename(CALCE_PATH).upper().startswith("CS2_"):
    CALCE_PATH = os.path.dirname(CALCE_PATH)

os.makedirs(OUT_DIR, exist_ok=True)
print("NASA :", NASA_PATH); print("CALCE:", CALCE_PATH); print("OUT  :", OUT_DIR)

# %%
# ===== Cell 2: Imports and plot style =====
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
keras, layers = tf.keras, tf.keras.layers

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 300, "figure.dpi": 110})
# Distinct but not colourful: black, blue, brown, grey + different markers
MODELS = ["LSTM", "GRU", "Transformer", "CNN-LSTM", "CNN-GRU", "CNN-GRU-RC"]   # baselines; proposed GRU-SDR is in Step 8b
MSTYLE = {"LSTM":        dict(color="#1f4e9c", marker="s"),
          "GRU":         dict(color="#7f7f7f", marker="D"),
          "Transformer": dict(color="#000000", marker="^"),
          "CNN-LSTM":    dict(color="#8b4513", marker="o")}
LS = {"Fine-tuned": "-", "Scratch": "--"}      # solid = fine-tuned, dashed = from scratch
print("TensorFlow", tf.__version__)

def _unzip(path, dest):
    if path.lower().endswith(".zip"):
        if not os.path.isdir(dest):
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        return dest
    return path

# %%
# ===== Cell 3: NASA -> one row per discharge cycle =====
V_HI, V_LO = 3.8, 3.6      # voltage window for the partial-discharge time

def curve_features(t, v):
    """Features from the early part of one discharge curve (not the full discharge time,
    which at constant current is the capacity itself)."""
    t = np.asarray(t, float) - float(t[0]); v = np.asarray(v, float)
    hi, lo = np.where(v <= V_HI)[0], np.where(v <= V_LO)[0]
    if len(hi) == 0 or len(lo) == 0 or t[-1] < 600:
        return None
    v300 = float(np.interp(300, t, v))
    return dict(tv=t[lo[0]] - t[hi[0]], v300=v300, vm600=float(v[t <= 600].mean()), dv0=float(v[0]) - v300)

def nasa_cycles(root, cell):
    f = glob.glob(os.path.join(root, "**", cell + ".mat"), recursive=True)
    assert f, f"{cell}.mat not found under {root}"
    m = sio.loadmat(f[0], squeeze_me=True, struct_as_record=False)
    rows = []
    for c in np.atleast_1d(m[cell].cycle):
        if str(c.type).strip().lower() != "discharge":
            continue
        d = c.data
        cap = float(np.atleast_1d(d.Capacity)[0])
        t, v, i = (np.asarray(x, float).ravel() for x in (d.Time, d.Voltage_measured, d.Current_measured))
        on = np.abs(i) > 0.5 * np.abs(np.median(i))          # under load only
        if on.sum() < 5 or not np.isfinite(cap):
            continue
        f_ = curve_features(t[on], v[on])
        if f_ is not None:
            rows.append(dict(cap=cap, **f_))
    return pd.DataFrame(rows)

# %%
# ===== Cell 4: CALCE CS2 -> one row per discharge cycle (Arbin xlsx/txt files) =====
def _calce_files(root, cell):
    root = _unzip(root, root[:-4]) if root.lower().endswith(".zip") else root
    for z in glob.glob(os.path.join(root, "**", cell + "*.zip"), recursive=True):
        _unzip(z, z[:-4])
    fs = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
          if os.path.basename(p).upper().startswith(cell.upper() + "_") and p.lower().endswith((".xlsx", ".xls", ".txt"))]
    def key(p):                                   # file names end with the test date, e.g. CS2_35_1_10_11
        n = re.findall(r"\d+", os.path.basename(p)[len(cell):])
        if len(n) >= 3:
            mth, day, yr = (int(x) for x in n[-3:])
            return (yr % 100, mth, day, p)
        return (99, 99, 99, p)
    return sorted(set(fs), key=key)

def _read_arbin(p):
    if p.lower().endswith(".txt"):
        df = pd.read_csv(p, sep="\t")
    else:
        xl = pd.ExcelFile(p)
        sh = [s for s in xl.sheet_names if s.lower().startswith("channel")] or xl.sheet_names[-1:]
        df = pd.concat([xl.parse(s) for s in sh], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    col = lambda k: next(c for c in df.columns if c.lower().startswith(k.lower()))
    out = pd.DataFrame({"t": df[col("Test_Time")], "cyc": df[col("Cycle_Index")],
                        "i": df[col("Current")], "v": df[col("Voltage")],
                        "qd": df[col("Discharge_Capacity")]}).apply(pd.to_numeric, errors="coerce").dropna()
    return out

def calce_cycles(root, cell, cache_dir=OUT_DIR):
    cache = os.path.join(cache_dir, f"cycles_v2_{cell}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    files = _calce_files(root, cell)
    assert files, f"No files for {cell} under {root}"
    rows = []
    for p in files:
        try:
            df = _read_arbin(p)
        except Exception as e:
            print("  skip", os.path.basename(p), "->", e); continue
        for _, g in df.groupby("cyc"):
            dch = g[g.i < -0.05]
            if len(dch) < 5:
                continue
            f_ = curve_features(dch.t.values, dch.v.values)
            if f_ is not None:
                rows.append(dict(cap=dch.qd.max() - dch.qd.min(), **f_))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    print(f"  {cell}: {len(files)} files -> {len(out)} discharge cycles")
    return out

# %%
# ===== Cell 5: Clean cycles, SOH/RUL labels, relative features =====
FEATS = ["tv_rel", "v300", "vm600", "dv0"]

def prepare(raw, cell, dataset):
    d = raw.copy().reset_index(drop=True)
    d = d[(d.cap > 0.1 * d.cap.max()) & (d.tv > 0)].reset_index(drop=True)
    med = d.cap.rolling(9, center=True, min_periods=1).median()
    d = d[(d.cap - med).abs() <= 0.08 * med].reset_index(drop=True)     # drop glitch cycles
    q0 = d.cap.iloc[:5].median(); t0 = d.tv.iloc[:5].median()
    d["SOH"] = d.cap / q0
    below = np.where(d.SOH.values < EOL)[0]
    n_eol = int(below[0]) if len(below) else len(d) - 1
    d = d.iloc[:n_eol + 1].reset_index(drop=True)                        # keep cycles up to EOL
    d["RUL"] = n_eol - np.arange(len(d))
    d["tv_rel"] = d.tv / t0                                              # unit-free feature
    d["cell"], d["dataset"], d["cycle"] = cell, dataset, np.arange(len(d))
    return d, n_eol

nasa_root = _unzip(NASA_PATH, "/content/NASA_data" if os.path.isdir("/content") else NASA_PATH[:-4])
data, n_eol = {}, {}
for c in NASA_CELLS:
    data[c], n_eol[c] = prepare(nasa_cycles(nasa_root, c), c, "NASA")
for c in CALCE_CELLS:
    data[c], n_eol[c] = prepare(calce_cycles(CALCE_PATH, c), c, "CALCE")

summ = pd.DataFrame([dict(Dataset=data[c].dataset[0], Cell=c, Cycles_to_EOL=n_eol[c],
                          Initial_capacity_Ah=round(data[c].cap.iloc[:5].median(), 3)) for c in data])
summ.to_csv(os.path.join(OUT_DIR, "Table_cells.csv"), index=False)
print(summ.to_string(index=False))

# %%
# ===== Cell 6: Windows and scaling =====
fx = StandardScaler().fit(pd.concat([data[c] for c in NASA_CELLS])[FEATS].values)
RUL_SCALE_SRC = float(np.mean([n_eol[c] for c in NASA_CELLS]))
RUL_SCALE_TGT = float(n_eol[FT_CELL])          # known only for the fine-tuning cell

def win(d, rul_scale, upto=None):
    d = d if upto is None else d.iloc[:upto]
    X = fx.transform(d[FEATS].values).astype("float32")
    Y = np.c_[(d.SOH.values - EOL) / (1 - EOL), d.RUL.values / rul_scale].astype("float32")   # both ~0..1
    if len(d) < SEQ:
        return None, None
    idx = np.arange(SEQ - 1, len(d))
    W = np.stack([X[k - SEQ + 1:k + 1] for k in idx])
    pos = np.broadcast_to(np.linspace(0, 1, SEQ, dtype="float32")[None, :, None], (len(W), SEQ, 1))
    return np.concatenate([W, pos], axis=2), Y[idx]          # last channel = position in window

def cat(parts):
    parts = [p for p in parts if p[0] is not None]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

Xs, Ys = cat([win(data[c], RUL_SCALE_SRC) for c in NASA_CELLS])
print("Source (NASA) windows:", Xs.shape)

# %%
# ===== Cell 7: Models =====
N_IN = len(FEATS) + 1      # features + position channel

def build(name):
    inp = keras.Input(shape=(SEQ, N_IN))
    if name == "GRU-SDR":
        # Proposed dual-branch, health-informed GRU:
        #  health branch  : GRU on the raw window -> SOH head
        #  trend branch   : 2 causal Conv1D -> GRU -> RUL head
        #  the RUL head also sees the health features and the predicted SOH
        h = layers.GRU(32)(inp)
        c = layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
        c = layers.Conv1D(32, 3, padding="causal", activation="relu")(c)
        c = layers.GRU(32)(c)
        s = layers.Dense(16, activation="relu")(layers.Dropout(0.2)(h))
        soh = layers.Dense(1)(s)
        r = layers.Concatenate()([h, c, soh])
        r = layers.Dense(16, activation="relu")(layers.Dropout(0.2)(r))
        rul = layers.Dense(1)(r)
        return keras.Model(inp, layers.Concatenate()([soh, rul]))
    if name == "Transformer":                       # one small encoder block
        x = layers.Dense(32)(inp)
        a = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, a]))
        f = layers.Dense(64, activation="relu")(x)
        f = layers.Dense(32)(f)
        x = layers.LayerNormalization()(layers.Add()([x, f]))
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = inp
        if name.startswith("CNN-"):                 # CNN-LSTM, CNN-GRU, CNN-GRU-RC
            x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
            x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
        x = layers.GRU(32)(x) if "GRU" in name else layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(2)(x)                        # [SOH, scaled RUL]
    return keras.Model(inp, out)

def train(m, X, Y, epochs, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    n_val = max(1, int(0.2 * len(X))) if len(X) >= 10 else 0
    kw = dict(epochs=epochs, batch_size=BATCH, verbose=0, shuffle=True)
    if n_val:   # chronological validation: the last 20% of windows
        Xtr, Ytr, Xv, Yv = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        m.fit(Xtr, Ytr, validation_data=(Xv, Yv),
              callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)], **kw)
    else:
        m.fit(X, Y, **kw)
    return m

# ----- variant: CNN-GRU trained with an RUL-consistency term -----
# Two windows of the same cell, RC_GAP cycles apart, pass through the same network.
# Besides the usual MSE, the loss asks that the predicted RUL drops between them as much as
# the true RUL does (one cycle per cycle), which counters flat, overestimated RUL curves.
def rc_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    d_true = y_true[:, 1] - y_true[:, 3]
    d_pred = y_pred[:, 1] - y_pred[:, 3]
    return mse + RC_LAMBDA * tf.reduce_mean(tf.square(d_pred / d_true - 1.0))

def rc_pairs(parts):
    A, B, YA, YB = [], [], [], []
    for X, Y in parts:
        if X is None or len(X) <= RC_GAP:
            continue
        A.append(X[:-RC_GAP]); B.append(X[RC_GAP:]); YA.append(Y[:-RC_GAP]); YB.append(Y[RC_GAP:])
    return np.concatenate(A), np.concatenate(B), np.concatenate(YA), np.concatenate(YB)

def train_rc(base, parts, epochs, lr):
    A, B, YA, YB = rc_pairs(parts)
    ia, ib = keras.Input(shape=(SEQ, N_IN)), keras.Input(shape=(SEQ, N_IN))
    pm = keras.Model([ia, ib], layers.Concatenate()([base(ia), base(ib)]))
    pm.compile(optimizer=keras.optimizers.Adam(lr), loss=rc_loss)
    Y = np.c_[YA, YB]
    n_val = max(1, int(0.2 * len(A))) if len(A) >= 10 else 0
    kw = dict(epochs=epochs, batch_size=BATCH, verbose=0, shuffle=True)
    if n_val:   # chronological validation, as for the other models
        pm.fit([A[:-n_val], B[:-n_val]], Y[:-n_val], validation_data=([A[-n_val:], B[-n_val:]], Y[-n_val:]),
               callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)], **kw)
    else:
        pm.fit([A, B], Y, **kw)
    return base

def fit_any(name, base, parts, epochs, lr):
    if name.endswith("-RC"):
        return train_rc(base, parts, epochs, lr)
    X, Y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return train(base, X, Y, epochs, lr)

# %%
# ===== Cell 8: Pretrain on NASA, then scratch vs. fine-tuned on CALCE =====
TEST_CELLS = [c for c in CALCE_CELLS if c != FT_CELL]
test_w = {c: win(data[c], RUL_SCALE_TGT) for c in TEST_CELLS}
records, preds = [], {}
old_runs = old_pred = None
f_runs, f_pred = os.path.join(OUT_DIR, "Table_all_runs.csv"), os.path.join(OUT_DIR, "Table_predictions.csv")
if RESUME and os.path.exists(f_runs) and os.path.exists(f_pred):
    old_runs, old_pred = pd.read_csv(f_runs), pd.read_csv(f_pred)
    done = [m for m in MODELS if m in set(old_runs.Model) and m in set(old_pred.Model)]
    old_runs, old_pred = old_runs[old_runs.Model.isin(done)], old_pred[old_pred.Model.isin(done)]
else:
    done = []
TRAIN_MODELS = [m for m in MODELS if m not in done]
print("Loaded from earlier run:", done or "none", "| training now:", TRAIN_MODELS)
src_parts = [win(data[c], RUL_SCALE_SRC) for c in NASA_CELLS]

def score(name, method, frac, seed, m):
    for c in TEST_CELLS:
        X, Y = test_w[c]
        P = m.predict(X, verbose=0)
        P[:, 1] = np.maximum(P[:, 1], 0)
        P[:, 0] = P[:, 0] * (1 - EOL) + EOL; Y = Y.copy(); Y[:, 0] = Y[:, 0] * (1 - EOL) + EOL
        soh_e = P[:, 0] - Y[:, 0]
        rul_e = (P[:, 1] - Y[:, 1]) * RUL_SCALE_TGT
        records.append(dict(Model=name, Method=method, Fraction=frac, Seed=seed, Cell=c,
                            SOH_RMSE=np.sqrt(np.mean(soh_e ** 2)), SOH_MAE=np.mean(np.abs(soh_e)),
                            RUL_RMSE=np.sqrt(np.mean(rul_e ** 2)), RUL_MAE=np.mean(np.abs(rul_e))))
        preds[(name, method, frac, seed, c)] = P

for name in TRAIN_MODELS:
    for s in SEEDS:
        keras.backend.clear_session(); keras.utils.set_random_seed(s)
        src = fit_any(name, build(name), src_parts, EPOCHS_SRC, 1e-3)
        w_src = src.get_weights()
        # zero-shot (NASA model used directly; SOH only is meaningful)
        score(name, "Zero-shot", 0.0, s, src)
        for frac in FRACTIONS:
            upto = max(SEQ + 2, int(round(frac * (n_eol[FT_CELL] + 1))))
            tgt_parts = [win(data[FT_CELL], RUL_SCALE_TGT, upto)]
            keras.utils.set_random_seed(s)
            score(name, "Scratch", frac, s, fit_any(name, build(name), tgt_parts, EPOCHS_TGT, 1e-3))
            keras.utils.set_random_seed(s)
            ft = build(name); ft.set_weights(w_src)
            score(name, "Fine-tuned", frac, s, fit_any(name, ft, tgt_parts, EPOCHS_TGT, 1e-4))
        print(f"{name} seed {s} done")

res = pd.concat([old_runs, pd.DataFrame(records)], ignore_index=True) if old_runs is not None else pd.DataFrame(records)
res.to_csv(os.path.join(OUT_DIR, "Table_all_runs.csv"), index=False)

# per-cycle predictions (mean over seeds) for the digital-twin figure
rows = [old_pred] if old_pred is not None else []
for name in TRAIN_MODELS:
    for method, fracs in [("Zero-shot", [0.0]), ("Scratch", FRACTIONS), ("Fine-tuned", FRACTIONS)]:
        for frac in fracs:
            for c in TEST_CELLS:
                Pm = np.mean([preds[(name, method, frac, s, c)] for s in SEEDS], axis=0)
                Yt = test_w[c][1]
                rows.append(pd.DataFrame({"Model": name, "Method": method, "Fraction": frac, "Cell": c,
                                          "Cycle": data[c].cycle.values[SEQ - 1:],
                                          "SOH_true": Yt[:, 0] * (1 - EOL) + EOL, "SOH_pred": Pm[:, 0],
                                          "RUL_true": Yt[:, 1] * RUL_SCALE_TGT, "RUL_pred": Pm[:, 1] * RUL_SCALE_TGT}))
pd.concat(rows).round(5).to_csv(os.path.join(OUT_DIR, "Table_predictions.csv"), index=False)

# %%
# ===== Cell 8b: Proposed method - GRU with SOH-derived RUL (GRU-SDR) =====
# The fine-tuned GRU predicts SOH. Instead of learning RUL directly, RUL is derived from the
# filtered SOH prediction, assuming the cell fades at the same average rate as the training cell:
#     RUL(k) = N_eol_train * (mean of last 5 predicted SOH - 0.8) / (1 - 0.8)
# N_eol_train = cycles to 80% of the training cell (CS2_35, same RUL scale as all models);
# the 5-cycle mean is the same filter the digital twin uses. Nothing is tuned on the test cells.
# Retrains the GRU with the same seeds and scores GRU-SDR per seed (about 10 minutes).
SDR_WIN = 5
N_EOL_TRAIN = RUL_SCALE_TGT

def sdr_rul(soh_pred):
    s = pd.Series(soh_pred).rolling(SDR_WIN, min_periods=1).mean().values
    return np.maximum(N_EOL_TRAIN * (s - EOL) / (1 - EOL), 0)

sdr_records, sdr_preds = [], {}
def score_sdr(method, frac, seed, m):
    for c in TEST_CELLS:
        X, Y = test_w[c]
        P = m.predict(X, verbose=0)
        soh_p = P[:, 0] * (1 - EOL) + EOL
        soh_t = Y[:, 0] * (1 - EOL) + EOL
        rul_t = Y[:, 1] * RUL_SCALE_TGT
        rul_p = sdr_rul(soh_p)
        se, re = soh_p - soh_t, rul_p - rul_t
        sdr_records.append(dict(Model="GRU-SDR", Method=method, Fraction=frac, Seed=seed, Cell=c,
                                SOH_RMSE=np.sqrt(np.mean(se ** 2)), SOH_MAE=np.mean(np.abs(se)),
                                RUL_RMSE=np.sqrt(np.mean(re ** 2)), RUL_MAE=np.mean(np.abs(re)),
                                RUL_bias=np.mean(re), RUL_over=np.mean(re > 0)))
        sdr_preds[(method, frac, seed, c)] = (soh_p, rul_p, soh_t, rul_t)

f_prop = os.path.join(OUT_DIR, "Table_proposed_runs.csv")
_skip = RESUME and os.path.exists(f_prop) and os.path.exists(os.path.join(OUT_DIR, "Table_proposed_predictions.csv"))
for s in ([] if _skip else SEEDS):
    keras.backend.clear_session(); keras.utils.set_random_seed(s)
    src = fit_any("GRU", build("GRU"), src_parts, EPOCHS_SRC, 1e-3)
    w_src = src.get_weights()
    score_sdr("Zero-shot", 0.0, s, src)
    for frac in FRACTIONS:
        upto = max(SEQ + 2, int(round(frac * (n_eol[FT_CELL] + 1))))
        tgt_parts = [win(data[FT_CELL], RUL_SCALE_TGT, upto)]
        keras.utils.set_random_seed(s)
        score_sdr("Scratch", frac, s, fit_any("GRU", build("GRU"), tgt_parts, EPOCHS_TGT, 1e-3))
        keras.utils.set_random_seed(s)
        ft = build("GRU"); ft.set_weights(w_src)
        score_sdr("Fine-tuned", frac, s, fit_any("GRU", ft, tgt_parts, EPOCHS_TGT, 1e-4))
    print(f"GRU-SDR seed {s} done")

if _skip:
    print("GRU-SDR loaded from earlier run")
    sdr = pd.read_csv(f_prop)
else:
  sdr = pd.DataFrame(sdr_records)
  sdr.to_csv(f_prop, index=False)
  rows = []
  for method, fracs in [("Zero-shot", [0.0]), ("Scratch", FRACTIONS), ("Fine-tuned", FRACTIONS)]:
      for frac in fracs:
          for c in TEST_CELLS:
              sp = np.mean([sdr_preds[(method, frac, s, c)][0] for s in SEEDS], axis=0)
              rp = np.mean([sdr_preds[(method, frac, s, c)][1] for s in SEEDS], axis=0)
              _, _, st, rt = sdr_preds[(method, frac, SEEDS[0], c)]
              rows.append(pd.DataFrame({"Model": "GRU-SDR", "Method": method, "Fraction": frac, "Cell": c,
                                        "Cycle": data[c].cycle.values[SEQ - 1:], "SOH_true": st, "SOH_pred": sp,
                                        "RUL_true": rt, "RUL_pred": rp}))
  pd.concat(rows).round(5).to_csv(os.path.join(OUT_DIR, "Table_proposed_predictions.csv"), index=False)

per_seed = sdr.groupby(["Method", "Fraction", "Seed"])[["SOH_RMSE", "RUL_RMSE", "RUL_bias", "RUL_over"]].mean()
summary = per_seed.groupby(["Method", "Fraction"]).agg(["mean", "std"])
summary[("SOH_RMSE", "mean")] *= 100; summary[("SOH_RMSE", "std")] *= 100
print(summary.round(2))

# add GRU-SDR to the main tables, so Tables and Figures include it
res = pd.concat([res[res.Model != "GRU-SDR"], sdr[res.columns]], ignore_index=True)
res.to_csv(os.path.join(OUT_DIR, "Table_all_runs.csv"), index=False)
_pp = pd.read_csv(os.path.join(OUT_DIR, "Table_predictions.csv"))
_pp = pd.concat([_pp[_pp.Model != "GRU-SDR"], pd.read_csv(os.path.join(OUT_DIR, "Table_proposed_predictions.csv"))[_pp.columns]])
_pp.to_csv(os.path.join(OUT_DIR, "Table_predictions.csv"), index=False)

# %%
# ===== Cell 9: Tables =====
agg = (res.groupby(["Model", "Method", "Fraction", "Seed"])[["SOH_RMSE", "SOH_MAE", "RUL_RMSE", "RUL_MAE"]]
          .mean().groupby(["Model", "Method", "Fraction"]).agg(["mean", "std"]))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.round(4).to_csv(os.path.join(OUT_DIR, "Table_main_results.csv"), index=False)
show = agg.copy()
for k in ["SOH_RMSE", "SOH_MAE"]:
    show[k] = show[k + "_mean"].map("{:.4f}".format) + " ± " + show[k + "_std"].map("{:.4f}".format)
for k in ["RUL_RMSE", "RUL_MAE"]:
    show[k] = show[k + "_mean"].map("{:.1f}".format) + " ± " + show[k + "_std"].map("{:.1f}".format)
show.loc[show.Method == "Zero-shot", ["RUL_RMSE", "RUL_MAE"]] = "–"
print("Mean over seeds and test cells (± std over seeds):")
print(show[["Model", "Method", "Fraction", "SOH_RMSE", "SOH_MAE", "RUL_RMSE", "RUL_MAE"]].to_string(index=False))

per_cell = (res[res.Fraction == 0.25].groupby(["Model", "Method", "Cell"])[["SOH_RMSE", "RUL_RMSE"]]
            .mean().reset_index())
per_cell.round(4).to_csv(os.path.join(OUT_DIR, "Table_per_cell_25pct.csv"), index=False)

# %%
# ===== Cell 10: Figure 1 - workflow diagram =====
FIG_OUT = OUT_DIR
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
EC = "#1a1a1a"
W, H = 3.5, 5.0
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, ls="-", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec=EC, lw=lw, ls=ls))

def t(x, y, s, size=6.8, weight="normal", ha="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va="center", linespacing=1.2, style=style)

def arrow(p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EC, ls=ls, shrinkA=0, shrinkB=0))

L, R = 0.74, 3.46          # content area (left column holds stage labels)
CW = R - L
def stage(y, n, name):
    t(0.06, y, f"{n}", 9, "bold", ha="left"); t(0.2, y, name, 6.8, "bold", ha="left")

# 1 Data
y1 = 4.38
stage(y1 + 0.28, 1, "Data")
box(L, y1, CW / 2 - 0.05, 0.52); t(L + CW / 4 - 0.025, y1 + 0.37, "NASA (source)", 7, "bold")
t(L + CW / 4 - 0.025, y1 + 0.15, "B0005, B0006, B0007, B0018\n2 Ah, 2 A discharge", 6.1)
box(L + CW / 2 + 0.05, y1, CW / 2 - 0.05, 0.52); t(L + 3 * CW / 4 + 0.025, y1 + 0.37, "CALCE CS2 (target)", 7, "bold")
t(L + 3 * CW / 4 + 0.025, y1 + 0.15, "CS2_35 to CS2_38\nLiCoO$_2$, 1.1 Ah, 1C", 6.1)

# 2 Features
y2 = 3.5
stage(y2 + 0.28, 2, "Features")
box(L, y2, CW, 0.62)
t(L + 0.08, y2 + 0.4, "Early part of each discharge (to 3.6 V):\n• 3.8→3.6 V time   • voltage at 300 s\n• mean V, 0–600 s   • early voltage drop", 6.2, ha="left")
t(L + CW - 0.08, y2 + 0.4, "SOH $= Q/Q_0$\nRUL to 80% SOH\nlast 10 cycles", 6.2, ha="right")
arrow((L + CW / 4, y1), (L + CW / 4, y2 + 0.62)); arrow((L + 3 * CW / 4, y1), (L + 3 * CW / 4, y2 + 0.62))

# 3 Learning
y3 = 1.86
stage(y3 + 0.7, 3, "Learning")
bw = CW / 2 - 0.05
xs = L + bw + 0.1
box(L, y3 + 0.96, bw, 0.44); t(L + bw / 2, y3 + 1.28, "Pretraining", 7, "bold"); t(L + bw / 2, y3 + 1.08, "all NASA cells\n(also tested zero-shot)", 5.9)
box(L, y3 + 0.24, bw, 0.46); t(L + bw / 2, y3 + 0.55, "Fine-tuning", 7, "bold")
t(L + bw / 2, y3 + 0.35, "10/25/50/100% of CS2_35\nlearning rate 10$^{-4}$", 5.9)
arrow((L + bw / 2, y3 + 0.96), (L + bw / 2, y3 + 0.7)); t(L + bw / 2 + 0.05, y3 + 0.84, "weights", 5.8, ha="left")
box(xs, y3 + 0.24, bw, 0.46, ls="--"); t(xs + bw / 2, y3 + 0.55, "From scratch", 7, "bold"); t(xs + bw / 2, y3 + 0.35, "same CS2_35 share", 6.1)
box(L, y3 - 0.08, CW, 0.22); t(L + CW / 2, y3 + 0.03, "Models: LSTM  \u00b7  GRU  \u00b7  Transformer  \u00b7  CNN-LSTM (joint SOH + RUL)", 6.0)
arrow((L + bw / 2, y2), (L + bw / 2, y3 + 1.4))                      # NASA windows -> pretraining
xc = xs + bw / 2; yb = y3 + 0.86                                      # CALCE windows: down, then split
ax.plot([xc, xc], [y2, yb + 0.12], color=EC, lw=0.8)
ax.plot([xc, xc], [yb + 0.12, yb], color=EC, lw=0.8)
arrow((xc, yb), (xc, y3 + 0.7), ls="--")
ax.plot([xc, L + bw * 0.82], [yb, yb], color=EC, lw=0.8)
arrow((L + bw * 0.82, yb), (L + bw * 0.82, y3 + 0.7))

# 4 Evaluation
y4 = 1.02
stage(y4 + 0.27, 4, "Evaluation")
box(L, y4, CW, 0.5)
t(L + CW / 2, y4 + 0.25, "Unseen cells CS2_36, CS2_37, CS2_38\nSOH and RUL RMSE, MAE; mean ± SD over 3 seeds", 6.3)
arrow((L + CW / 2, y3 - 0.08), (L + CW / 2, y4 + 0.5))

# 5 Digital twin
y5 = 0.06
stage(y5 + 0.3, 5, "Digital\ntwin")
box(L, y5, CW, 0.6, lw=1.1)
t(L + CW / 2, y5 + 0.3, "Offline DT with the selected model (GRU): replays an\nunseen cell and shows predicted SOH and RUL, a health\nstatus (healthy / warning / maintenance), and an event log", 6.2)
arrow((L + CW / 2, y4), (L + CW / 2, y5 + 0.6))

fig.savefig(os.path.join(FIG_OUT, "Fig1_workflow.pdf")); fig.savefig(os.path.join(FIG_OUT, "Fig1_workflow.png"), dpi=300)

plt.show()

# %%
# ===== Cell 11: Figures 2-5 (from the saved CSV files) =====
FIG_RES, FIG_NASA, FIG_OUT = OUT_DIR, nasa_root, OUT_DIR
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

RES, NASA, OUT = FIG_RES, FIG_NASA, FIG_OUT
EOL = 0.80
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# every curve: its own colour, line style and marker (muted colours); proposed model in solid black
MSTY = {"LSTM":        ("#1f4e9c", (0, (5, 2)),           "s"),
        "GRU":         ("#7f7f7f", "-.",                  "D"),
        "Transformer": ("#2e6b3a", (0, (1, 1.5)),         "^"),
        "CNN-LSTM":    ("#8b4513", (0, (6, 1.5, 1, 1.5)), "v"),
        "CNN-GRU":     ("#6a5a8c", (0, (3, 1, 1, 1, 1, 1)), "P"),
        "CNN-GRU-RC":  ("#3d7a7a", (0, (8, 2)),           "X"),
        "GRU-SDR":     ("#000000", "-",                   "o")}
LABEL = {m: m for m in MSTY}; LABEL["GRU-SDR"] = "GRU-SDR (proposed)"
CELLSTYLE = [("#1f4e9c", "-", "o"), ("#000000", "--", "s"), ("#8b4513", "-.", "^"), ("#7f7f7f", ":", "D")]
BARSTYLE = {"LSTM": ("#1f4e9c", "white", "////"), "GRU": ("#7f7f7f", "#d9d9d9", ""),
            "Transformer": ("#2e6b3a", "white", "xxxx"), "CNN-LSTM": ("#8b4513", "white", "...."),
            "CNN-GRU": ("#6a5a8c", "white", "\\\\\\\\"), "CNN-GRU-RC": ("#3d7a7a", "white", "++++"),
            "GRU-SDR": ("#000000", "#000000", "")}

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png")); plt.show()

def prepare_soh(cap):
    d = pd.DataFrame({"cap": cap})
    d = d[d.cap > 0.1 * d.cap.max()].reset_index(drop=True)
    med = d.cap.rolling(9, center=True, min_periods=1).median()
    d = d[(d.cap - med).abs() <= 0.08 * med].reset_index(drop=True)
    soh = (d.cap / d.cap.iloc[:5].median()).values
    k = np.where(soh < EOL)[0]
    return soh[:k[0] + 1] if len(k) else soh

agg = pd.read_csv(os.path.join(RES, "Table_main_results.csv"))
MODELS = [m for m in MSTY if m in set(agg.Model)]
fr = sorted(agg[agg.Method != "Zero-shot"].Fraction.unique())
xp = np.arange(len(fr))

# ---------- Fig. 2: capacity fade of all cells ----------
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
if NASA:
    import scipy.io as sio
    for k, c in enumerate(["B0005", "B0006", "B0007", "B0018"]):
        m = sio.loadmat(glob.glob(os.path.join(NASA, "**", c + ".mat"), recursive=True)[0],
                        squeeze_me=True, struct_as_record=False)
        cap = [float(np.atleast_1d(x.data.Capacity)[0]) for x in np.atleast_1d(m[c].cycle)
               if str(x.type).strip() == "discharge"]
        soh = prepare_soh(np.array(cap)); col, ls, mk = CELLSTYLE[k]
        axs[0].plot(np.arange(len(soh)), soh * 100, color=col, ls=ls, lw=0.9, marker=mk,
                    markevery=max(1, len(soh) // 7), ms=3.2, mfc="white", mew=0.8, label=c)
for k, c in enumerate(["CS2_35", "CS2_36", "CS2_37", "CS2_38"]):
    f = os.path.join(RES, f"cycles_v2_{c}.csv")
    if os.path.exists(f):
        soh = prepare_soh(pd.read_csv(f).cap.values); col, ls, mk = CELLSTYLE[k]
        axs[1].plot(np.arange(len(soh)), soh * 100, color=col, ls=ls, lw=0.8, marker=mk,
                    markevery=max(1, len(soh) // 7), ms=3.2, mfc="white", mew=0.8,
                    label=c + (" (fine-tuning)" if c == "CS2_35" else " (test)"))
for ax, t in zip(axs, ["(a) NASA cells (source)", "(b) CALCE CS2 cells (target)"]):
    ax.axhline(EOL * 100, color="#555555", lw=0.7, ls=":")
    ax.set_xlabel("Cycle"); ax.set_title(t, fontsize=8); ax.legend(loc="upper right", handlelength=2.4)
axs[0].set_ylabel("SOH (%)")
axs[0].text(0.02, 0.06, "EOL = 80%", transform=axs[0].transAxes, fontsize=6.5, color="#555555")
fig.tight_layout(w_pad=1.0); save(fig, "Fig2_SOH_all_cells")

# ---------- Fig. 3: error vs. share of target data (fine-tuned models) ----------
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5))
for name in MODELS:
    col, ls, mk = MSTY[name]
    a = agg[(agg.Model == name) & (agg.Method == "Fine-tuned")].set_index("Fraction").loc[fr]
    for ax, k, f in [(axs[0], "SOH_RMSE", 100), (axs[1], "RUL_RMSE", 1)]:
        ax.plot(xp, a[k + "_mean"].values * f, color=col, ls=ls, lw=1.6 if name == "GRU-SDR" else 1.1,
                marker=mk, ms=4.5, mfc=col if name == "GRU-SDR" else "white", mec=col, mew=0.9, label=LABEL[name])
for ax, yl, t in zip(axs, ["SOH RMSE (%)", "RUL RMSE (cycles)"], ["(a) SOH", "(b) RUL"]):
    ax.set_xticks(xp); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
    ax.set_xlabel("Share of CS2_35 life used for fine-tuning"); ax.set_ylabel(yl); ax.set_title(t, fontsize=8)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(MODELS), bbox_to_anchor=(0.5, -0.1), columnspacing=1.0, handlelength=3.0)
fig.tight_layout(w_pad=1.5); save(fig, "Fig3_error_vs_target_data")

# ---------- Fig. 4: gain from transfer learning ----------
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
w = 0.8 / len(MODELS)
for j, name in enumerate(MODELS):
    ec, fc, ht = BARSTYLE[name]
    ft = agg[(agg.Model == name) & (agg.Method == "Fine-tuned")].set_index("Fraction").loc[fr]
    sc = agg[(agg.Model == name) & (agg.Method == "Scratch")].set_index("Fraction").loc[fr]
    for ax, k in [(axs[0], "SOH_RMSE"), (axs[1], "RUL_RMSE")]:
        gain = (1 - ft[k + "_mean"].values / sc[k + "_mean"].values) * 100
        ax.bar(xp + (j - (len(MODELS) - 1) / 2) * w, gain, w * 0.92, color=fc, edgecolor=ec, hatch=ht,
               lw=0.9, label=name)
for ax, t in zip(axs, ["(a) SOH", "(b) RUL"]):
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xp); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
    ax.set_xlabel("Share of CS2_35 life used for training"); ax.set_title(t, fontsize=8)
axs[0].set_ylabel("RMSE reduction from\nNASA pretraining (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, [LABEL[x] for x in l], loc="lower center", ncol=len(MODELS), bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=1.5); save(fig, "Fig4_transfer_gain")

# ---------- Fig. 5: digital twin dashboard (offline replay of an unseen cell) ----------
pf = os.path.join(RES, "Table_predictions.csv")
if os.path.exists(pf):
    P = pd.read_csv(pf)
    FR_DT = 0.25
    cand = agg[(agg.Method == "Fine-tuned") & (agg.Fraction == FR_DT)].sort_values("SOH_RMSE_mean")
    best = cand.Model.iloc[0]
    cell = sorted(P.Cell.unique())[0]
    d = P[(P.Model == best) & (P.Method == "Fine-tuned") & (P.Fraction == FR_DT) & (P.Cell == cell)].sort_values("Cycle")
    def status(s):
        return ("HEALTHY", "#3c763d") if s >= 0.90 else (("WARNING", "#a66a00") if s >= 0.85 else ("MAINTENANCE", "#9b2c2c"))
    snaps = [d.iloc[int(len(d) * 0.25)], d.iloc[int(len(d) * 0.80)]]
    fig = plt.figure(figsize=(7.16, 3.25))
    for j, now in enumerate(snaps):
        x0 = 0.005 + j * 0.5
        bg = fig.add_axes([x0, 0.01, 0.485, 0.98]); bg.axis("off")
        bg.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="#fafafa", ec="#555555", lw=0.8, transform=bg.transAxes))
        bg.text(0.03, 0.94, "Battery Digital Twin", fontsize=8.5, fontweight="bold", transform=bg.transAxes)
        bg.text(0.03, 0.875, f"Cell {cell} (unseen)  •  cycle {int(now.Cycle)}\n{best} fine-tuned on {int(FR_DT*100)}% of CS2_35",
                fontsize=6.2, color="#444444", transform=bg.transAxes, va="center", linespacing=1.3)
        lab, col = status(now.SOH_pred)
        bg.add_patch(FancyBboxPatch((0.73, 0.925), 0.24, 0.05, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=col, ec=col, transform=bg.transAxes))
        bg.text(0.85, 0.95, lab, fontsize=7, color="white", fontweight="bold", ha="center", va="center", transform=bg.transAxes)
        # battery gauge
        ga = fig.add_axes([x0 + 0.02, 0.43, 0.09, 0.36]); ga.axis("off"); ga.set_xlim(0, 1); ga.set_ylim(0, 1.12)
        ga.add_patch(Rectangle((0.32, 1.0), 0.36, 0.08, fc="#555555", ec="#555555"))
        ga.add_patch(FancyBboxPatch((0.05, 0.0), 0.9, 1.0, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec="#333333", lw=1))
        lvl = float(np.clip((now.SOH_pred - 0.7) / 0.3, 0, 1))
        ga.add_patch(Rectangle((0.1, 0.04), 0.8, 0.92 * lvl, fc=col, ec="none", alpha=0.85))
        ga.text(0.5, 0.5, f"{now.SOH_pred*100:.1f}%", ha="center", va="center", fontsize=8, fontweight="bold")
        # KPI tiles
        for k, (name_, val, sub) in enumerate([("Predicted SOH", f"{now.SOH_pred*100:.1f}%", f"actual {now.SOH_true*100:.1f}%"),
                                               ("Predicted RUL", f"{max(now.RUL_pred,0):.0f} cyc", f"actual {now.RUL_true:.0f} cyc")]):
            yb = 0.66 - k * 0.24
            bg.add_patch(FancyBboxPatch((0.25, yb), 0.2, 0.18, boxstyle="round,pad=0,rounding_size=0.015",
                                        fc="white", ec="#999999", lw=0.6, transform=bg.transAxes))
            bg.text(0.35, yb + 0.14, name_, fontsize=6.2, ha="center", color="#444444", transform=bg.transAxes)
            bg.text(0.35, yb + 0.075, val, fontsize=9, ha="center", fontweight="bold", transform=bg.transAxes)
            bg.text(0.35, yb + 0.025, sub, fontsize=5.8, ha="center", color="#666666", transform=bg.transAxes)
        bg.text(0.035, 0.30, "Status rule (predicted SOH):\n≥ 90% healthy\n85–90% warning\n< 85% maintenance",
                fontsize=5.8, color="#444444", va="top", transform=bg.transAxes, linespacing=1.3)
        # SOH trend
        a1 = fig.add_axes([x0 + 0.275, 0.49, 0.195, 0.29])
        past = d[d.Cycle <= now.Cycle]
        a1.plot(d.Cycle, d.SOH_true * 100, color="#bbbbbb", lw=0.8, ls=":", label="actual (future)")
        a1.plot(past.Cycle, past.SOH_true * 100, color="#000000", lw=1.0, ls="-", label="actual")
        a1.plot(past.Cycle, past.SOH_pred * 100, color="#8b4513", lw=1.0, ls="--", label="predicted")
        for th, c_ in [(90, "#3c763d"), (85, "#a66a00"), (80, "#9b2c2c")]:
            a1.axhline(th, color=c_, lw=0.6, ls="-.")
        a1.axvline(now.Cycle, color="#555555", lw=0.6)
        a1.set_ylim(77, 102); a1.set_xlim(0, d.Cycle.max())
        a1.set_title("SOH (%)", fontsize=6.8, pad=2); a1.tick_params(labelsize=5.8)
        a1.legend(fontsize=5.0, loc="lower left", handlelength=2.2, frameon=True, framealpha=0.95, edgecolor="#cccccc")
        # RUL trend
        a2 = fig.add_axes([x0 + 0.275, 0.1, 0.195, 0.27])
        a2.plot(d.Cycle, d.RUL_true, color="#bbbbbb", lw=0.8, ls=":")
        a2.plot(past.Cycle, past.RUL_true, color="#000000", lw=1.0, ls="-")
        a2.plot(past.Cycle, past.RUL_pred.clip(lower=0), color="#1f4e9c", lw=1.0, ls="--", label="predicted")
        a2.legend(fontsize=5.0, loc="upper right", handlelength=2.2, frameon=True, framealpha=0.95, edgecolor="#cccccc")
        a2.axvline(now.Cycle, color="#555555", lw=0.6)
        a2.set_xlim(0, d.Cycle.max()); a2.set_ylim(0, max(d.RUL_true.max(), d.RUL_pred.max()) * 1.08)
        a2.set_title("RUL (cycles)", fontsize=6.8, pad=2); a2.tick_params(labelsize=5.8); a2.set_xlabel("Cycle", fontsize=6.2, labelpad=1)
        bg.text(0.03, 0.035, f"({'ab'[j]})", fontsize=8, transform=bg.transAxes)
    save(fig, "Fig5_digital_twin")
print("figures saved to", OUT)

# %%
# ===== Cell 12: Save everything as one zip =====
with open(os.path.join(OUT_DIR, "settings.json"), "w") as f:
    json.dump(dict(NASA_CELLS=NASA_CELLS, CALCE_CELLS=CALCE_CELLS, FT_CELL=FT_CELL, EOL=EOL, SEQ=SEQ,
                   FRACTIONS=FRACTIONS, SEEDS=SEEDS, FEATS=FEATS, EPOCHS_SRC=EPOCHS_SRC,
                   EPOCHS_TGT=EPOCHS_TGT, BATCH=BATCH, PATIENCE=PATIENCE,
                   RUL_SCALE_SRC=RUL_SCALE_SRC, RUL_SCALE_TGT=RUL_SCALE_TGT), f, indent=2)
zpath = OUT_DIR.rstrip("/") + ".zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT_DIR)):
        z.write(os.path.join(OUT_DIR, fn), arcname=os.path.join(os.path.basename(OUT_DIR), fn))
print("Saved:", zpath)
try:
    from google.colab import files as _f
    _f.download(zpath)
except Exception:
    pass

# %%
# ===== Step 1: Battery Digital Twin (interactive) =====
# Replays the unseen CALCE cells cycle by cycle with the trained models' predictions.
# Needs the TL_NASA_CALCE_Results folder (Table_predictions.csv + cycles_v2_*.csv) in Google Drive.
import os, glob, json, shutil
import numpy as np, pandas as pd
from IPython.display import HTML, display

# 1) find the results folder
try:
    RES_DIR = OUT_DIR                                   # defined if the training notebook ran in this session
except NameError:
    RES_DIR = os.environ.get("TL_RESULTS", "")
if not os.path.isfile(os.path.join(RES_DIR, "Table_predictions.csv")):
    try:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            shutil.rmtree("/content/drive", ignore_errors=True)
            drive.mount("/content/drive")
    except ImportError:
        pass
    hits = sorted(glob.glob("/content/drive/MyDrive/**/TL_NASA_CALCE_Results/Table_predictions.csv", recursive=True))
    assert hits, "Table_predictions.csv not found. Run the training notebook first."
    RES_DIR = os.path.dirname(hits[0])
print("Results:", RES_DIR)

# 2) predictions + per-cycle measurements (same cleaning as in training, so cycle numbers match)
P = pd.read_csv(os.path.join(RES_DIR, "Table_predictions.csv"))
_fp = os.path.join(RES_DIR, "Table_proposed_predictions.csv")     # proposed GRU-SDR (Step 8b), if present
if os.path.exists(_fp):
    P = pd.concat([P[P.Model != "GRU-SDR"], pd.read_csv(_fp)[P.columns]], ignore_index=True)
EOL_DT = 0.80
def clean(d):
    d = d[(d.cap > 0.1 * d.cap.max()) & (d.tv > 0)].reset_index(drop=True)
    med = d.cap.rolling(9, center=True, min_periods=1).median()
    d = d[(d.cap - med).abs() <= 0.08 * med].reset_index(drop=True)
    soh = d.cap / d.cap.iloc[:5].median()
    k = np.where(soh.values < EOL_DT)[0]
    return d.iloc[:(k[0] + 1 if len(k) else len(d))].reset_index(drop=True)

payload = {"runs": {}, "meas": {}}
for c in sorted(P.Cell.unique()):
    f = os.path.join(RES_DIR, f"cycles_v2_{c}.csv")
    if os.path.exists(f):
        m = clean(pd.read_csv(f))
        payload["meas"][c] = {"cap": m.cap.round(4).tolist(), "tv": m.tv.round(1).tolist(),
                              "v300": m.v300.round(4).tolist(), "vm600": m.vm600.round(4).tolist(),
                              "dv0": m.dv0.round(4).tolist()}
for (mdl, meth, fr, c), g in P.groupby(["Model", "Method", "Fraction", "Cell"]):
    g = g.sort_values("Cycle")
    payload["runs"][f"{mdl}|{meth}|{fr:g}|{c}"] = {
        "cyc": g.Cycle.astype(int).tolist(),
        "sT": (g.SOH_true * 100).round(2).tolist(), "sP": (g.SOH_pred * 100).round(2).tolist(),
        "rT": g.RUL_true.round(0).astype(int).tolist(), "rP": g.RUL_pred.clip(lower=0).round(0).astype(int).tolist()}
# recommended combination: GRU for SOH and CNN-GRU for RUL (best models for each task)
COMBO = "GRU (SOH) + CNN-GRU (RUL)"
for k in list(payload["runs"]):
    mdl, meth, fr, c = k.split("|")
    k2 = f"CNN-GRU|{meth}|{fr}|{c}"
    if mdl == "GRU" and k2 in payload["runs"] and payload["runs"][k2]["cyc"] == payload["runs"][k]["cyc"]:
        g, r = payload["runs"][k], payload["runs"][k2]
        payload["runs"][f"{COMBO}|{meth}|{fr}|{c}"] = {"cyc": g["cyc"], "sT": g["sT"], "sP": g["sP"], "rT": r["rT"], "rP": r["rP"]}
order = ["GRU-SDR", "GRU", "CNN-GRU", COMBO, "LSTM", "Transformer", "CNN-LSTM", "CNN-GRU-RC", "DB-GRU"]
present = {k.split("|")[0] for k in payload["runs"]}
payload["models"] = [m for m in order if m in present]
payload["cells"] = sorted(P.Cell.unique().tolist())
payload["fracs"] = sorted(float(x) for x in P[P.Method != "Zero-shot"].Fraction.unique())

HTML_DT = r"""
<div id="dt-root">
<style>
#dt-root{font-family:"Segoe UI",Roboto,Arial,sans-serif;color:#1d2733;background:#eef2f6;padding:14px;border-radius:12px;max-width:1180px}
#dt-root *{box-sizing:border-box}
.dt-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.dt-title{font-size:20px;font-weight:700;color:#14324f}
.dt-sub{font-size:12px;color:#5b6b7b;margin-top:2px}
.dt-ctrl{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.dt-ctrl label{font-size:11px;color:#5b6b7b;display:flex;flex-direction:column;gap:2px}
.dt-ctrl select{padding:5px 6px;border:1px solid #c6d0db;border-radius:6px;background:#fff;font-size:12px}
.dt-btn{border:none;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer;font-size:12px}
.dt-play{background:#14324f;color:#fff}.dt-reset{background:#d9e1ea;color:#14324f}
.dt-grid{display:grid;grid-template-columns:200px 1fr 260px;gap:12px}
.dt-card{background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(20,50,79,.12)}
.dt-card h4{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#5b6b7b}
.dt-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.dt-kpi .v{font-size:24px;font-weight:700}.dt-kpi .s{font-size:11px;color:#6b7b8b;margin-top:2px}
.dt-status{font-size:20px;font-weight:800;color:#fff;border-radius:8px;padding:12px;text-align:center;letter-spacing:.05em}
.dt-rule{font-size:11px;color:#5b6b7b;margin-top:8px;line-height:1.5}
.dt-charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.dt-log{font-size:11.5px;max-height:150px;overflow:auto;line-height:1.6}
.dt-log div{border-bottom:1px solid #eef1f4;padding:2px 0}
.dt-meas{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px}
.dt-meas b{display:block;font-size:15px;color:#14324f}
.dt-slider{width:100%;margin-top:10px}
.dt-foot{font-size:11px;color:#6b7b8b;margin-top:8px}
canvas{width:100%;height:220px;display:block}
</style>
<div class="dt-head">
  <div><div class="dt-title">&#128267; Battery Digital Twin</div>
       <div class="dt-sub">Offline replay of an unseen CALCE cell &middot; models pretrained on NASA, adapted to CALCE</div></div>
  <div class="dt-ctrl">
    <label>Cell<select id="dt-cell"></select></label>
    <label>Model<select id="dt-model"></select></label>
    <label>Strategy<select id="dt-meth"><option>Fine-tuned</option><option>Scratch</option><option>Zero-shot</option></select></label>
    <label>Target data<select id="dt-frac"></select></label>
    <label>Speed<select id="dt-speed"><option value="300">slow</option><option value="80" selected>normal</option><option value="20">fast</option></select></label>
    <button class="dt-btn dt-play" id="dt-play">&#9654; Play</button>
    <button class="dt-btn dt-reset" id="dt-reset">&#8635; Reset</button>
  </div>
</div>
<div class="dt-grid">
  <div class="dt-card" style="text-align:center"><h4>Battery</h4><canvas id="dt-bat" style="height:250px"></canvas></div>
  <div class="dt-card">
    <h4>Live state &middot; <span id="dt-cycle">cycle 0</span></h4>
    <div class="dt-kpis">
      <div class="dt-kpi"><div class="v" id="k-sp">--</div><div class="s">Predicted SOH</div></div>
      <div class="dt-kpi"><div class="v" id="k-st">--</div><div class="s">Measured SOH</div></div>
      <div class="dt-kpi"><div class="v" id="k-rp">--</div><div class="s">Predicted RUL (cycles)</div></div>
      <div class="dt-kpi"><div class="v" id="k-rt">--</div><div class="s">Actual RUL (cycles)</div></div>
    </div>
    <h4 style="margin-top:14px">Measured this cycle (early discharge, to 3.6 V)</h4>
    <div class="dt-meas">
      <div>Capacity<b id="m-cap">--</b></div><div>3.8&rarr;3.6 V time<b id="m-tv">--</b></div>
      <div>Voltage at 300 s<b id="m-v300">--</b></div><div>Mean voltage 0&ndash;600 s<b id="m-vm">--</b></div>
    </div>
    <input type="range" id="dt-slider" class="dt-slider" min="0" max="0" value="0">
  </div>
  <div class="dt-card">
    <h4>Health status</h4>
    <div class="dt-status" id="dt-status">--</div>
    <div class="dt-rule">Rule on predicted SOH (5-cycle average, change confirmed after 3 cycles, status never moves back): &ge; 90% healthy &middot; 85&ndash;90% warning &middot; &lt; 85% maintenance &middot; 80% = end of life</div>
    <h4 style="margin-top:12px">Event log</h4>
    <div class="dt-log" id="dt-log"></div>
  </div>
</div>
<div class="dt-charts">
  <div class="dt-card"><h4>State of health (%)</h4><canvas id="dt-soh"></canvas></div>
  <div class="dt-card"><h4>Remaining useful life (cycles)</h4><canvas id="dt-rul"></canvas></div>
</div>
<div class="dt-foot" id="dt-foot"></div>
<script>
(function(){
const D = __PAYLOAD__;
const $ = id => document.getElementById(id);
const fill = (sel, arr, fmt) => { sel.innerHTML = arr.map(v => `<option value="${v}">${fmt ? fmt(v) : v}</option>`).join(""); };
fill($("dt-cell"), D.cells); fill($("dt-model"), D.models);
fill($("dt-frac"), D.fracs, v => Math.round(v * 100) + "% of CS2_35");
$("dt-frac").value = D.fracs.includes(0.25) ? 0.25 : D.fracs[0];
let run = null, meas = null, i = 0, timer = null, ST = [], SM = [], EV = [];
const COL = {ok:"#2f7d4f", warn:"#c27c0e", bad:"#b23a3a"};
const status = s => s >= 90 ? ["HEALTHY", COL.ok] : (s >= 85 ? ["WARNING", COL.warn] : ["MAINTENANCE", COL.bad]);

function key(){
  const m = $("dt-meth").value; const f = m === "Zero-shot" ? "0" : String(+$("dt-frac").value);
  return `${$("dt-model").value}|${m}|${f}|${$("dt-cell").value}`;
}
function load(){
  stop(); run = D.runs[key()]; meas = D.meas[$("dt-cell").value] || null; i = 0; events = [];
  $("dt-frac").disabled = $("dt-meth").value === "Zero-shot";
  if (!run) { $("dt-foot").textContent = "No predictions for this combination."; return; }
  $("dt-slider").max = run.cyc.length - 1; $("dt-slider").value = 0;
  // filtered status: 5-cycle moving average of predicted SOH, change confirmed after 3 cycles
  SM = run.sP.map((_, k) => { const a = run.sP.slice(Math.max(0, k - 4), k + 1); return a.reduce((x, y) => x + y, 0) / a.length; });
  ST = []; EV = []; let cur = status(SM[0])[0], cnt = 0;
  SM.forEach((v, k) => { const c = status(v)[0];
    const RK = {HEALTHY: 0, WARNING: 1, MAINTENANCE: 2};   // degradation is irreversible: status only moves forward
    if (RK[c] > RK[cur]) { cnt++; if (cnt >= 3) { cur = c; cnt = 0; EV.push([k, `cycle ${run.cyc[k]}: status changed to ${c}`]); } } else cnt = 0;
    ST.push(cur); });
  const e = run.sP.map((p, k) => p - run.sT[k]); const rmse = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / e.length);
  const er = run.rP.map((p, k) => p - run.rT[k]); const rr = Math.sqrt(er.reduce((a, b) => a + b * b, 0) / er.length);
  $("dt-foot").textContent = `Whole-life error for this cell: SOH RMSE ${rmse.toFixed(2)}%, RUL RMSE ${rr.toFixed(0)} cycles. Recorded data replayed offline; live BMS/ESP32 input is future work.`;
  draw();
}
function setup(cv){ const r = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * r; cv.height = h * r; const c = cv.getContext("2d"); c.setTransform(r, 0, 0, r, 0, 0); return [c, w, h]; }
function chart(cv, series, yMin, yMax, lines){
  const [c, w, h] = setup(cv); const L = 42, R = 10, T = 10, B = 26; const xMax = run.cyc[run.cyc.length - 1];
  const X = v => L + (w - L - R) * v / xMax, Y = v => T + (h - T - B) * (1 - (v - yMin) / (yMax - yMin));
  c.clearRect(0, 0, w, h); c.font = "11px Arial"; c.fillStyle = "#6b7b8b"; c.strokeStyle = "#e6ebf0"; c.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = yMin + (yMax - yMin) * k / 4; c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke();
    c.fillText(v.toFixed(0), 4, Y(v) + 4); }
  for (let k = 0; k <= 5; k++) { const v = xMax * k / 5; c.fillText(v.toFixed(0), X(v) - 8, h - 8); }
  (lines || []).forEach(([v, col, lab]) => { c.strokeStyle = col; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke();
    c.setLineDash([]); c.fillStyle = col; c.fillText(lab, w - R - 70, Y(v) - 3); });
  series.forEach(s => { c.strokeStyle = s.col; c.lineWidth = s.w || 2; c.setLineDash(s.dash || []); c.beginPath();
    const n = s.upto == null ? s.y.length : s.upto + 1;
    for (let k = 0; k < n; k++) { const x = X(run.cyc[k]), y = Y(s.y[k]); k ? c.lineTo(x, y) : c.moveTo(x, y); } c.stroke(); c.setLineDash([]); });
  c.strokeStyle = "#14324f"; c.lineWidth = 1; c.beginPath(); c.moveTo(X(run.cyc[i]), T); c.lineTo(X(run.cyc[i]), h - B); c.stroke();
  const labs = series.filter(s => s.lab); let lw = labs.reduce((a, s) => a + c.measureText(s.lab).width + 34, 0);
  c.fillStyle = "rgba(255,255,255,0.92)"; c.fillRect(w - R - lw - 6, T, lw + 4, 16);
  let lx = w - R - lw; labs.forEach(s => { c.fillStyle = s.col; c.fillRect(lx, T + 6, 14, 3); c.fillStyle = "#333";
    c.fillText(s.lab, lx + 18, T + 11); lx += c.measureText(s.lab).width + 34; });
}
function battery(s, col){
  const [c, w, h] = setup($("dt-bat")); c.clearRect(0, 0, w, h);
  const bw = 90, bh = 190, x = (w - bw) / 2, y = 34;
  c.fillStyle = "#4a5866"; c.fillRect(x + bw / 2 - 18, y - 12, 36, 12);
  c.lineWidth = 4; c.strokeStyle = "#4a5866"; c.beginPath(); c.roundRect ? c.roundRect(x, y, bw, bh, 10) : c.rect(x, y, bw, bh); c.stroke();
  const lvl = Math.max(0, Math.min(1, (s - 70) / 30)); c.fillStyle = col;
  c.fillRect(x + 7, y + 7 + (bh - 14) * (1 - lvl), bw - 14, (bh - 14) * lvl);
  c.fillStyle = "#1d2733"; c.font = "bold 20px Arial"; c.textAlign = "center"; c.fillText(s.toFixed(1) + "%", w / 2, y + bh / 2 + 6);
  c.font = "11px Arial"; c.fillStyle = "#5b6b7b"; c.fillText("predicted SOH (filtered)", w / 2, y + bh + 20); c.textAlign = "left";
}
function draw(){
  if (!run) return; const sP = run.sP[i], cyc = run.cyc[i], lab = ST[i];
  const col = lab === "HEALTHY" ? COL.ok : (lab === "WARNING" ? COL.warn : COL.bad);
  $("dt-cycle").textContent = `cycle ${cyc} of ${run.cyc[run.cyc.length - 1]}`;
  $("k-sp").textContent = sP.toFixed(1) + "%"; $("k-st").textContent = run.sT[i].toFixed(1) + "%";
  $("k-rp").textContent = run.rP[i]; $("k-rt").textContent = run.rT[i];
  $("k-sp").style.color = col; const st = $("dt-status"); st.textContent = lab; st.style.background = col;
  if (meas && meas.cap[cyc] !== undefined) {
    $("m-cap").textContent = meas.cap[cyc].toFixed(3) + " Ah"; $("m-tv").textContent = meas.tv[cyc].toFixed(0) + " s";
    $("m-v300").textContent = meas.v300[cyc].toFixed(3) + " V"; $("m-vm").textContent = meas.vm600[cyc].toFixed(3) + " V"; }
  const log = EV.filter(e => e[0] <= i).map(e => e[1]).reverse();
  log.push(`cycle ${run.cyc[0]}: monitoring started (${ST[0].toLowerCase()})`);
  $("dt-log").innerHTML = log.map(e => `<div>${e}</div>`).join("");
  battery(SM[i], col);
  chart($("dt-soh"), [{y: run.sT, col: "#c9d1d9", w: 1.5, dash: [3, 3]}, {y: run.sT, col: "#1d2733", upto: i, lab: "measured"},
                      {y: run.sP, col: "#8b4513", upto: i, dash: [6, 3], lab: "predicted"}], 75, Math.max(102, Math.ceil(Math.max(...run.sP, ...run.sT) + 2)),
        [[90, COL.ok, "healthy 90%"], [85, COL.warn, "warning 85%"], [80, COL.bad, "EOL 80%"]]);
  const rMax = Math.max(...run.rT, ...run.rP) * 1.08;
  chart($("dt-rul"), [{y: run.rT, col: "#c9d1d9", w: 1.5, dash: [3, 3]}, {y: run.rT, col: "#1d2733", upto: i, lab: "actual"},
                      {y: run.rP, col: "#1f4e9c", upto: i, dash: [6, 3], lab: "predicted"}], 0, rMax, []);
  $("dt-slider").value = i;
}
function stop(){ if (timer) { clearInterval(timer); timer = null; } $("dt-play").innerHTML = "&#9654; Play"; }
function play(){ if (timer) { stop(); return; } if (i >= run.cyc.length - 1) i = 0; $("dt-play").innerHTML = "&#10074;&#10074; Pause";
  timer = setInterval(() => { if (i >= run.cyc.length - 1) { stop(); return; } i++; draw(); }, +$("dt-speed").value); }
["dt-cell", "dt-model", "dt-meth", "dt-frac"].forEach(id => $(id).addEventListener("change", load));
$("dt-speed").addEventListener("change", () => { if (timer) { stop(); play(); } });
$("dt-play").addEventListener("click", play);
$("dt-reset").addEventListener("click", () => { stop(); i = 0; draw(); });
$("dt-slider").addEventListener("input", e => { stop(); i = +e.target.value; draw(); });
window.addEventListener("resize", draw);
load();
})();
</script>
</div>
"""
html_dt = HTML_DT.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
with open(os.path.join(RES_DIR, "Battery_Digital_Twin.html"), "w") as f:
    f.write("<!doctype html><html><head><meta charset='utf-8'><title>Battery Digital Twin</title></head><body>" + html_dt + "</body></html>")
print("Saved standalone copy:", os.path.join(RES_DIR, "Battery_Digital_Twin.html"))
display(HTML(html_dt))

