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
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType fonts (IEEE PDF eXpress rejects Type 3)

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
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.4))
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
    ax.set_ylabel("SOH (%)"); ax.set_ylim(77, 102)
axs[0].text(0.02, 0.06, "EOL = 80%", transform=axs[0].transAxes, fontsize=6.5, color="#555555")
fig.tight_layout(h_pad=0.6); save(fig, "Fig2_SOH_all_cells")

# ---------- Fig. 3: error vs. share of target data (fine-tuned models) ----------
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.4))
for name in MODELS:
    col, ls, mk = MSTY[name]
    a = agg[(agg.Model == name) & (agg.Method == "Fine-tuned")].set_index("Fraction").loc[fr]
    for ax, k, f in [(axs[0], "SOH_RMSE", 100), (axs[1], "RUL_RMSE", 1)]:
        ax.plot(xp, a[k + "_mean"].values * f, color=col, ls=ls, lw=1.6 if name == "GRU-SDR" else 1.1,
                marker=mk, ms=4.5, mfc=col if name == "GRU-SDR" else "white", mec=col, mew=0.9, label=LABEL[name])
for ax, yl, t in zip(axs, ["SOH RMSE (%)", "RUL RMSE (cycles)"], ["(a) SOH", "(b) RUL"]):
    ax.set_xticks(xp); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
    ax.set_ylabel(yl); ax.set_title(t, fontsize=8)
axs[1].set_xlabel("Share of CS2_35 life used for fine-tuning")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=3.0)
fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=0.8); save(fig, "Fig3_error_vs_target_data")

# ---------- Fig. 4: gain from transfer learning ----------
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.4))
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
    ax.set_title(t, fontsize=8); ax.set_ylabel("RMSE reduction (%)")
axs[1].set_xlabel("Share of CS2_35 life used for training")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, [LABEL[x] for x in l], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=0.8); save(fig, "Fig4_transfer_gain")

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
