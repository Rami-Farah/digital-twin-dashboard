# =============================================================================
# Lean–Digital Model Stress Dashboard  —  Thesis Chapter 4
# Improvements over previous version:
#   1. Queue evolution uses rolling average (5-min window) → readable plot
#   2. Sensitivity chart shows 4 bars: baseline vs modified, side by side
#   3. Full OP2 sensitivity sweep (185→210 sec) as a continuous line chart
#   4. Thesis-ready text block removed from UI (kept as internal helper only)
#   5. Consistent color scheme throughout all charts
#   6. Chart captions match thesis figure numbering
# =============================================================================

import simpy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants ─────────────────────────────────────────────────────────────────
WARMUP_SEC  = 3600
MEASURE_SEC = 8 * 3600
TOTAL_SEC   = WARMUP_SEC + MEASURE_SEC
QUEUE_INTERVAL = 10   # seconds between queue snapshots

STATIONS  = ["OP1", "OP2", "OP3", "OP4"]
LEAN_BASE = [178, 197, 179, 172]
DT_BASE   = [183, 187, 184, 172]

LEAN_COLOR = "#2E4057"
DT_COLOR   = "#1A936F"
HL_COLOR   = "#E63946"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       120,
})


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class PacedLineModel:
    def __init__(self, env, station_names, station_times, release_interval):
        self.env              = env
        self.station_names    = station_names
        self.station_times    = station_times
        self.release_interval = release_interval

        self.resources = {n: simpy.Resource(env, capacity=1) for n in station_names}

        self.completed    = 0
        self.system_times = []
        self.station_busy = {n: 0.0  for n in station_names}
        self.station_waits= {n: []   for n in station_names}
        self.queue_snap   = {n: []   for n in station_names}
        self.queue_t      = {n: []   for n in station_names}
        self.queue_v      = {n: []   for n in station_names}

    def monitor_queues(self):
        while True:
            t = self.env.now
            if WARMUP_SEC <= t <= WARMUP_SEC + MEASURE_SEC:
                for n in self.station_names:
                    q = len(self.resources[n].queue)
                    self.queue_snap[n].append(q)
                    self.queue_t[n].append(t - WARMUP_SEC)
                    self.queue_v[n].append(q)
            yield self.env.timeout(QUEUE_INTERVAL)

    def visit(self, name, proc_time):
        arrival = self.env.now
        with self.resources[name].request() as req:
            yield req
            wait  = self.env.now - arrival
            if WARMUP_SEC <= self.env.now <= WARMUP_SEC + MEASURE_SEC:
                self.station_waits[name].append(wait)
            start = self.env.now
            yield self.env.timeout(proc_time)
            end   = self.env.now
            overlap = max(0.0,
                          min(end,   WARMUP_SEC + MEASURE_SEC) -
                          max(start, WARMUP_SEC))
            self.station_busy[name] += overlap

    def part_flow(self):
        t0 = self.env.now
        for n, pt in zip(self.station_names, self.station_times):
            yield self.env.process(self.visit(n, pt))
        t1 = self.env.now
        if WARMUP_SEC <= t1 <= WARMUP_SEC + MEASURE_SEC:
            self.completed += 1
            self.system_times.append(t1 - t0)

    def source(self):
        while True:
            self.env.process(self.part_flow())
            yield self.env.timeout(self.release_interval)


def run_sim(station_times, release_interval, station_names=None):
    """Run one simulation replicate and return a result dict."""
    if station_names is None:
        station_names = STATIONS
    env   = simpy.Environment()
    model = PacedLineModel(env, station_names, station_times, release_interval)
    env.process(model.source())
    env.process(model.monitor_queues())
    env.run(until=TOTAL_SEC)

    hours = MEASURE_SEC / 3600
    util  = {n: model.station_busy[n] / MEASURE_SEC for n in station_names}

    return {
        "throughput":   model.completed / hours,
        "system_time":  float(np.mean(model.system_times)) if model.system_times else 0.0,
        "utilization":  util,
        "avg_wait":     {n: float(np.mean(model.station_waits[n]))
                         if model.station_waits[n] else 0.0
                         for n in station_names},
        "avg_queue":    {n: float(np.mean(model.queue_snap[n]))
                         if model.queue_snap[n] else 0.0
                         for n in station_names},
        "max_queue":    {n: int(np.max(model.queue_snap[n]))
                         if model.queue_snap[n] else 0
                         for n in station_names},
        "bottleneck":   max(util, key=util.get),
        "queue_t":      model.queue_t,
        "queue_v":      model.queue_v,
    }


def theoretical_cap(times):
    return min(3600 / t for t in times)


# =============================================================================
# HELPERS
# =============================================================================

def rolling_avg(values, times_sec, window_sec=300):
    """Return smoothed queue series using a rolling time window."""
    t = np.array(times_sec)
    v = np.array(values)
    out_t, out_v = [], []
    step = window_sec
    for centre in range(0, int(MEASURE_SEC) + step, step):
        mask = (t >= centre) & (t < centre + window_sec)
        if mask.any():
            out_t.append((centre + window_sec / 2) / 3600)
            out_v.append(v[mask].mean())
    return np.array(out_t), np.array(out_v)


def warn_flags(r, target, focus="OP2"):
    flags = []
    tp = r["throughput"]
    bn = r["bottleneck"]
    u  = r["utilization"][bn]
    q  = r["avg_queue"].get(focus, 0.0)
    st_= r["system_time"]

    if tp < target - 0.1:
        flags.append(("error",
            f"Target throughput not achieved. Achieved {tp:.2f} vs target {target:.0f}."))
    if u >= 0.99:
        flags.append(("error",
            f"Critical bottleneck loading at {bn} ({u:.4f})."))
    elif u >= 0.95:
        flags.append(("warning",
            f"High bottleneck loading at {bn} ({u:.4f})."))
    if q > 1.0:
        flags.append(("error",
            f"Queue accumulation at {focus}: avg = {q:.2f} units."))
    elif q > 0.0:
        flags.append(("warning",
            f"Minor queue at {focus}: avg = {q:.2f} units."))
    if st_ > 1200:
        flags.append(("warning",
            f"Average system time is very high ({st_:.0f} sec)."))
    return flags


# =============================================================================
# CHART FUNCTIONS  — each returns a matplotlib Figure
# =============================================================================

def chart_throughput(lean_r, dt_r, target, label=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals  = [lean_r["throughput"], dt_r["throughput"]]
    bars  = ax.bar(["Lean baseline", "DT-assisted"], vals,
                   color=[LEAN_COLOR, DT_COLOR], width=0.5, edgecolor="white")
    ax.axhline(target, color=HL_COLOR, linewidth=1.5,
               linestyle="--", label=f"Target ({target} u/h)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.08,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (units/hour)")
    ax.set_ylim(0, target + 2.5)
    ax.set_title(f"Throughput comparison — {target} u/h demand{label}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def chart_system_time(lean_r, dt_r, label=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [lean_r["system_time"], dt_r["system_time"]]
    bars = ax.bar(["Lean baseline", "DT-assisted"], vals,
                  color=[LEAN_COLOR, DT_COLOR], width=0.5, edgecolor="white")
    ax.axhline(726, color=HL_COLOR, linewidth=1.5,
               linestyle="--", label="Stable reference (726 sec)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 15,
                f"{v:.0f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average system time (seconds)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title(f"Average system time comparison{label}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def chart_utilization(lean_r, dt_r, label=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    x  = np.arange(len(STATIONS))
    w  = 0.35
    lu = [lean_r["utilization"][n] for n in STATIONS]
    du = [dt_r["utilization"][n]   for n in STATIONS]
    ax.bar(x - w/2, lu, w, label="Lean baseline",
           color=LEAN_COLOR, edgecolor="white")
    ax.bar(x + w/2, du, w, label="DT-assisted",
           color=DT_COLOR,   edgecolor="white")
    ax.axhline(0.95, color=HL_COLOR, linewidth=1.2,
               linestyle="--", label="Critical threshold (0.95)")
    ax.set_xticks(x)
    ax.set_xticklabels(STATIONS)
    ax.set_ylabel("Resource utilization")
    ax.set_ylim(0.75, 1.08)
    ax.set_title(f"Resource utilization comparison{label}")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    return fig


def chart_queue_evolution(lean_r, dt_r, label=""):
    """Rolling-average queue at OP2 over the 8-hour measurement period."""
    fig, ax = plt.subplots(figsize=(9, 4))

    t_l, q_l = rolling_avg(lean_r["queue_v"]["OP2"],
                            lean_r["queue_t"]["OP2"], window_sec=300)
    t_d, q_d = rolling_avg(dt_r["queue_v"]["OP2"],
                            dt_r["queue_t"]["OP2"],  window_sec=300)

    ax.plot(t_l, q_l, color=LEAN_COLOR, linewidth=2,
            label=f"Lean baseline (avg {lean_r['avg_queue']['OP2']:.2f})")
    ax.plot(t_d, q_d, color=DT_COLOR,   linewidth=2,
            linestyle="--",
            label=f"DT-assisted (avg {dt_r['avg_queue']['OP2']:.2f})")
    ax.fill_between(t_l, q_l, alpha=0.10, color=LEAN_COLOR)
    ax.fill_between(t_d, q_d, alpha=0.10, color=DT_COLOR)

    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Queue length at OP2 (5-min rolling avg)")
    ax.set_title(f"Queue evolution at OP2{label}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def chart_sensitivity_bars(lean_base_r, dt_base_r,
                            lean_mod_r,  dt_mod_r,
                            extra_sec, target_label):
    """
    4-bar chart: baseline Lean / baseline DT / modified Lean / modified DT.
    Makes the effect of the sensitivity modifier immediately visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: throughput
    ax = axes[0]
    labels = ["Lean\nbaseline", "DT\nbaseline",
              f"Lean\n+{extra_sec}s", f"DT\n+{extra_sec}s"]
    vals   = [lean_base_r["throughput"], dt_base_r["throughput"],
              lean_mod_r["throughput"],  dt_mod_r["throughput"]]
    colors = [LEAN_COLOR, DT_COLOR, "#7B9BB5", "#6DC3A0"]
    bars   = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Throughput (units/hour)")
    ax.set_title("Throughput — baseline vs modified")

    # Right: queue at OP2
    ax = axes[1]
    qvals  = [lean_base_r["avg_queue"]["OP2"], dt_base_r["avg_queue"]["OP2"],
              lean_mod_r["avg_queue"]["OP2"],  dt_mod_r["avg_queue"]["OP2"]]
    bars2  = ax.bar(labels, qvals, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars2, qvals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg queue length at OP2")
    ax.set_title("Queue at OP2 — baseline vs modified")

    fig.suptitle(f"Sensitivity analysis — {target_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def chart_sweep(sweep_op2, sweep_lean_tp, sweep_dt_tp, target):
    """Continuous line: OP2 processing time vs achieved throughput."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sweep_op2, sweep_lean_tp,
            color=LEAN_COLOR, linewidth=2, marker="o", markersize=5,
            label="Lean baseline")
    ax.plot(sweep_op2, sweep_dt_tp,
            color=DT_COLOR,   linewidth=2, marker="s", markersize=5,
            linestyle="--",   label="DT-assisted")
    ax.axhline(target, color=HL_COLOR, linewidth=1.3, linestyle="--",
               label=f"Target ({target} u/h)")
    ax.axvline(197, color=LEAN_COLOR, linewidth=0.9, linestyle=":",
               alpha=0.7, label="Current Lean OP2 = 197 sec")
    ax.axvline(187, color=DT_COLOR,   linewidth=0.9, linestyle=":",
               alpha=0.7, label="Current DT OP2 = 187 sec")
    ax.set_xlabel("OP2 processing time (seconds)")
    ax.set_ylabel("Achieved throughput (units/hour)")
    ax.set_title(f"Sensitivity sweep — OP2 processing time vs throughput\n"
                 f"({target} u/h demand stress)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(
    page_title="Lean–Digital Model Stress Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Lean–Digital Model Stress Dashboard")
st.caption(
    "Interactive stress-testing dashboard for the production-line case study "
    "used in the thesis (Chapter 4 — Scenario-Based Analysis)."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Controls")

    target_demand    = st.selectbox("Target demand (units/hour)", [18, 19, 20], index=1)
    release_interval = 3600 / target_demand

    st.subheader("Lean configuration")
    lean_op1 = st.number_input("Lean OP1 (sec)", value=LEAN_BASE[0], step=1, min_value=1)
    lean_op2 = st.number_input("Lean OP2 (sec)", value=LEAN_BASE[1], step=1, min_value=1)
    lean_op3 = st.number_input("Lean OP3 (sec)", value=LEAN_BASE[2], step=1, min_value=1)
    lean_op4 = st.number_input("Lean OP4 (sec)", value=LEAN_BASE[3], step=1, min_value=1)

    st.subheader("Digital Model configuration")
    dt_op1 = st.number_input("DT OP1 (sec)", value=DT_BASE[0], step=1, min_value=1)
    dt_op2 = st.number_input("DT OP2 (sec)", value=DT_BASE[1], step=1, min_value=1)
    dt_op3 = st.number_input("DT OP3 (sec)", value=DT_BASE[2], step=1, min_value=1)
    dt_op4 = st.number_input("DT OP4 (sec)", value=DT_BASE[3], step=1, min_value=1)

    st.subheader("Sensitivity modifier")
    sens_target = st.selectbox(
        "Apply extra seconds to",
        ["None", "Lean OP2", "DT OP2", "Both OP2"],
        index=0
    )
    extra_sec = st.selectbox("Extra seconds", [0, 5, 10, 15], index=0)

    st.subheader("Sensitivity sweep")
    run_sweep = st.checkbox("Run OP2 sweep (185 → 210 sec)", value=False)
    st.caption("Adds ~30 s of computation. Enable only when needed.")

    run_btn = st.button("Run simulation", type="primary", use_container_width=True)

if not run_btn:
    st.info("Configure scenario parameters in the sidebar, then click **Run simulation**.")
    st.stop()

# ── Build time vectors ────────────────────────────────────────────────────────
lean_times = [lean_op1, lean_op2, lean_op3, lean_op4]
dt_times   = [dt_op1,   dt_op2,   dt_op3,   dt_op4]

# Baseline copies (before sensitivity modifier)
lean_base_times = lean_times[:]
dt_base_times   = dt_times[:]

# Apply sensitivity modifier
if sens_target == "Lean OP2":
    lean_times = lean_times[:1] + [lean_times[1] + extra_sec] + lean_times[2:]
elif sens_target == "DT OP2":
    dt_times   = dt_times[:1]   + [dt_times[1]   + extra_sec] + dt_times[2:]
elif sens_target == "Both OP2":
    lean_times = lean_times[:1] + [lean_times[1] + extra_sec] + lean_times[2:]
    dt_times   = dt_times[:1]   + [dt_times[1]   + extra_sec] + dt_times[2:]

# ── Run main simulations ──────────────────────────────────────────────────────
with st.spinner("Running simulations…"):
    lean_r = run_sim(lean_times, release_interval)
    dt_r   = run_sim(dt_times,   release_interval)

    # Baseline results (needed for 4-bar sensitivity chart)
    if sens_target != "None" and extra_sec > 0:
        lean_base_r = run_sim(lean_base_times, release_interval)
        dt_base_r   = run_sim(dt_base_times,   release_interval)
    else:
        lean_base_r = lean_r
        dt_base_r   = dt_r

    # OP2 sweep
    sweep_op2, sweep_lean_tp, sweep_dt_tp = [], [], []
    if run_sweep:
        for op2_val in range(185, 211, 5):
            lt = [lean_op1, op2_val,        lean_op3, lean_op4]
            dt = [dt_op1,   op2_val - 10,   dt_op3,   dt_op4]
            sweep_op2.append(op2_val)
            sweep_lean_tp.append(run_sim(lt, release_interval)["throughput"])
            sweep_dt_tp.append(run_sim(dt, release_interval)["throughput"])

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Lean configuration")
    ka, kb, kc, kd = st.columns(4)
    ka.metric("Throughput (u/h)",      f"{lean_r['throughput']:.2f}")
    kb.metric("Avg system time (sec)", f"{lean_r['system_time']:.0f}")
    kc.metric("Bottleneck",            lean_r["bottleneck"])
    kd.metric("Theoretical cap (u/h)", f"{theoretical_cap(lean_times):.2f}")

with c2:
    st.subheader("Digital Model configuration")
    ka, kb, kc, kd = st.columns(4)
    ka.metric("Throughput (u/h)",      f"{dt_r['throughput']:.2f}")
    kb.metric("Avg system time (sec)", f"{dt_r['system_time']:.0f}")
    kc.metric("Bottleneck",            dt_r["bottleneck"])
    kd.metric("Theoretical cap (u/h)", f"{theoretical_cap(dt_times):.2f}")

# ── Warnings ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Diagnostic warnings")
wl, wr = st.columns(2)

with wl:
    st.markdown("**Lean configuration**")
    flags = warn_flags(lean_r, target_demand)
    if not flags:
        st.success("No warnings detected.")
    for lvl, msg in flags:
        (st.error if lvl == "error" else st.warning)(msg)

with wr:
    st.markdown("**Digital Model configuration**")
    flags = warn_flags(dt_r, target_demand)
    if not flags:
        st.success("No warnings detected.")
    for lvl, msg in flags:
        (st.error if lvl == "error" else st.warning)(msg)

# ── Summary tables ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Summary table")
summary_df = pd.DataFrame([
    {
        "Scenario":               "Lean",
        "Target demand (u/h)":    target_demand,
        "Throughput (u/h)":       round(lean_r["throughput"],   2),
        "Avg system time (sec)":  round(lean_r["system_time"],  0),
        "Bottleneck":             lean_r["bottleneck"],
        "Bottleneck util.":       round(lean_r["utilization"][lean_r["bottleneck"]], 4),
        "Queue at OP2":           round(lean_r["avg_queue"]["OP2"], 2),
        "Avg wait OP2 (sec)":     round(lean_r["avg_wait"]["OP2"],  2),
    },
    {
        "Scenario":               "Digital Model",
        "Target demand (u/h)":    target_demand,
        "Throughput (u/h)":       round(dt_r["throughput"],   2),
        "Avg system time (sec)":  round(dt_r["system_time"],  0),
        "Bottleneck":             dt_r["bottleneck"],
        "Bottleneck util.":       round(dt_r["utilization"][dt_r["bottleneck"]], 4),
        "Queue at OP2":           round(dt_r["avg_queue"]["OP2"], 2),
        "Avg wait OP2 (sec)":     round(dt_r["avg_wait"]["OP2"],  2),
    },
])
st.dataframe(summary_df, use_container_width=True)

st.subheader("Resource utilization & queue detail")
util_df = pd.DataFrame([
    {
        "Resource":              r,
        "Lean util.":            round(lean_r["utilization"][r], 4),
        "DT util.":              round(dt_r["utilization"][r],   4),
        "Lean avg wait (sec)":   round(lean_r["avg_wait"][r],    2),
        "DT avg wait (sec)":     round(dt_r["avg_wait"][r],      2),
        "Lean avg queue":        round(lean_r["avg_queue"][r],   2),
        "DT avg queue":          round(dt_r["avg_queue"][r],     2),
        "Lean max queue":        lean_r["max_queue"][r],
        "DT max queue":          dt_r["max_queue"][r],
    }
    for r in STATIONS
])
st.dataframe(util_df, use_container_width=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Charts")

# Figure 4.1 & 4.2 side by side
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Figure 4.1** — Throughput comparison ({target_demand} u/h)")
    st.pyplot(chart_throughput(lean_r, dt_r, target_demand))
with col_b:
    st.markdown("**Figure 4.2** — Average system time comparison")
    st.pyplot(chart_system_time(lean_r, dt_r))

# Figure 4.3
st.markdown("**Figure 4.3** — Resource utilization by operator")
st.pyplot(chart_utilization(lean_r, dt_r))

# Figure 4.4 — rolling-average queue evolution
st.markdown("**Figure 4.4** — Queue evolution at OP2 (5-minute rolling average)")
st.pyplot(chart_queue_evolution(lean_r, dt_r))

# Figure 4.5 — sensitivity 4-bar chart (only when modifier is active)
if sens_target != "None" and extra_sec > 0:
    st.markdown(
        f"**Figure 4.5** — Sensitivity analysis: effect of +{extra_sec} sec "
        f"on {sens_target}"
    )
    st.pyplot(
        chart_sensitivity_bars(
            lean_base_r, dt_base_r,
            lean_r,      dt_r,
            extra_sec,
            f"{sens_target} under {target_demand} u/h demand"
        )
    )

# Figure 4.6 — continuous OP2 sweep (optional)
if run_sweep and sweep_op2:
    st.markdown(
        "**Figure 4.6** — OP2 processing time sensitivity sweep "
        f"({target_demand} u/h demand)"
    )
    st.pyplot(chart_sweep(sweep_op2, sweep_lean_tp, sweep_dt_tp, target_demand))

# ── Simulation parameters footer ──────────────────────────────────────────────
with st.expander("Simulation parameters"):
    st.write({
        "Warm-up period (sec)":       WARMUP_SEC,
        "Measurement period (sec)":   MEASURE_SEC,
        "Queue sample interval (sec)":QUEUE_INTERVAL,
        "Lean processing times (sec)":dict(zip(STATIONS, lean_times)),
        "DT processing times (sec)":  dict(zip(STATIONS, dt_times)),
        "Release interval (sec/unit)":round(release_interval, 2),
        "Target demand (u/h)":        target_demand,
    })
