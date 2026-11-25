# ===== Battery Digital Twin (interactive, runs in Colab) =====
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
