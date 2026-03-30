import math import simpy import numpy as np import pandas as pd import streamlit as st import matplotlib.pyplot as plt


============================================================


Simulation core


============================================================


WARMUP_SEC = 3600 MEASURE_SEC = 8 * 3600 TOTAL_SEC = WARMUP_SEC + MEASURE_SEC


class PacedLineModel: def init(self, env, station_names, station_times, release_interval, warmup, measure): self.env = env self.station_names = station_names self.station_times = station_times self.release_interval = release_interval self.warmup = warmup self.measure = measure self.end_measure = warmup + measure


self.resources = {
        name: simpy.Resource(env, capacity=1)
        for name in station_names
    }


    self.completed = 0
    self.system_times = []
    self.station_busy = {name: 0.0 for name in station_names}
    self.station_waits = {name: [] for name in station_names}
    self.queue_snapshots = {name: [] for name in station_names}
    self.queue_time = {name: [] for name in station_names}
    self.queue_values = {name: [] for name in station_names}


def monitor_queues(self):
    while True:
        if self.warmup <= self.env.now <= self.end_measure:
            for name in self.station_names:
                q_len = len(self.resources[name].queue)
                self.queue_snapshots[name].append(q_len)
                self.queue_time[name].append(self.env.now - self.warmup)
                self.queue_values[name].append(q_len)
        yield self.env.timeout(10)


def process_station(self, station_name, proc_time):
    arrival = self.env.now
    with self.resources[station_name].request() as req:
        yield req
        wait = self.env.now - arrival


        if self.warmup <= self.env.now <= self.end_measure:
            self.station_waits[station_name].append(wait)


        start = self.env.now
        yield self.env.timeout(proc_time)
        end = self.env.now


        overlap = max(0.0, min(end, self.end_measure) - max(start, self.warmup))
        self.station_busy[station_name] += overlap


def part_flow(self, part_id):
    start_time = self.env.now
    for name, ptime in zip(self.station_names, self.station_times):
        yield self.env.process(self.process_station(name, ptime))
    finish_time = self.env.now


    if self.warmup <= finish_time <= self.end_measure:
        self.completed += 1
        self.system_times.append(finish_time - start_time)


def source(self):
    part_id = 0
    while True:
        part_id += 1
        self.env.process(self.part_flow(part_id))
        yield self.env.timeout(self.release_interval)


def run_paced_line(station_names, station_times, release_interval): env = simpy.Environment() model = PacedLineModel( env=env, station_names=station_names, station_times=station_times, release_interval=release_interval, warmup=WARMUP_SEC, measure=MEASURE_SEC, )


env.process(model.source())
env.process(model.monitor_queues())
env.run(until=TOTAL_SEC)


throughput = model.completed / (MEASURE_SEC / 3600)
avg_system_time = float(np.mean(model.system_times)) if model.system_times else 0.0


utilization = {
    name: model.station_busy[name] / MEASURE_SEC
    for name in station_names
}
avg_waits = {
    name: (float(np.mean(model.station_waits[name])) if model.station_waits[name] else 0.0)
    for name in station_names
}
avg_queue = {
    name: (float(np.mean(model.queue_snapshots[name])) if model.queue_snapshots[name] else 0.0)
    for name in station_names
}
max_queue = {
    name: (int(np.max(model.queue_snapshots[name])) if model.queue_snapshots[name] else 0)
    for name in station_names
}
bottleneck = max(utilization, key=utilization.get)


return {
    "throughput_per_hour": throughput,
    "avg_system_time_sec": avg_system_time,
    "utilization": utilization,
    "avg_waits": avg_waits,
    "avg_queue": avg_queue,
    "max_queue": max_queue,
    "bottleneck": bottleneck,
    "queue_time": model.queue_time,
    "queue_values": model.queue_values,
}


============================================================


Default case study inputs


Based on the production-line study used in the thesis.


============================================================


STATIONS = ["OP1", "OP2", "OP3", "OP4"] LEAN_BASE = [178, 197, 179, 172] DT_BASE = [183, 187, 184, 172]


def theoretical_capacity(times): return min(3600 / t for t in times)


def warning_flags(results, target_throughput, queue_focus="OP2"): flags = [] if results["throughput_per_hour"] < target_throughput - 0.1: flags.append(("error", f"Target throughput not achieved. Achieved {results['throughput_per_hour']:.2f} vs target {target_throughput:.2f}.")) if results["utilization"][results["bottleneck"]] >= 0.99: flags.append(("warning", f"Critical bottleneck loading at {results['bottleneck']} ({results['utilization'][results['bottleneck']]:.4f}).")) elif results["utilization"][results["bottleneck"]] >= 0.95: flags.append(("warning", f"High bottleneck loading at {results['bottleneck']} ({results['utilization'][results['bottleneck']]:.4f}).")) if results["avg_queue"].get(queue_focus, 0.0) > 1.0: flags.append(("error", f"Queue accumulation detected at {queue_focus}: avg queue = {results['avg_queue'][queue_focus]:.2f}.")) elif results["avg_queue"].get(queue_focus, 0.0) > 0.0: flags.append(("warning", f"Minor queue accumulation at {queue_focus}: avg queue = {results['avg_queue'][queue_focus]:.2f}.")) if results["avg_system_time_sec"] > 1200: flags.append(("warning", f"Average system time is very high ({results['avg_system_time_sec']:.2f} sec).")) return flags


def add_delta(times, op2_delta, op1_delta=0, op3_delta=0, op4_delta=0): mapping = {"OP1": op1_delta, "OP2": op2_delta, "OP3": op3_delta, "OP4": op4_delta} return [max(1, t + mapping[name]) for name, t in zip(STATIONS, times)]


============================================================


Streamlit app


============================================================


st.set_page_config(page_title="Lean–Digital Model Stress Dashboard", layout="wide") st.title("Lean–Digital Model Stress Dashboard") st.caption("Interactive stress-testing dashboard for the production-line case used in the thesis.")


with st.sidebar: st.header("Scenario Controls") target_demand = st.selectbox("Target demand (units/hour)", [18, 19, 20], index=1) release_interval = 3600 / target_demand


st.subheader("Lean configuration")
lean_op1 = st.number_input("Lean OP1 (sec)", value=LEAN_BASE[0], step=1)
lean_op2 = st.number_input("Lean OP2 (sec)", value=LEAN_BASE[1], step=1)
lean_op3 = st.number_input("Lean OP3 (sec)", value=LEAN_BASE[2], step=1)
lean_op4 = st.number_input("Lean OP4 (sec)", value=LEAN_BASE[3], step=1)


st.subheader("Digital Model configuration")
dt_op1 = st.number_input("DT OP1 (sec)", value=DT_BASE[0], step=1)
dt_op2 = st.number_input("DT OP2 (sec)", value=DT_BASE[1], step=1)
dt_op3 = st.number_input("DT OP3 (sec)", value=DT_BASE[2], step=1)
dt_op4 = st.number_input("DT OP4 (sec)", value=DT_BASE[3], step=1)


st.subheader("Quick sensitivity modifiers")
sensitivity_target = st.selectbox("Apply extra seconds to", ["None", "Lean OP2", "DT OP2", "Both OP2"], index=0)
extra_seconds = st.selectbox("Extra seconds", [0, 5, 10], index=0)


run_button = st.button("Run simulation", type="primary")


if not run_button: st.info("Set your scenario values, then click 'Run simulation'.") st.stop()


lean_times = [lean_op1, lean_op2, lean_op3, lean_op4] dt_times = [dt_op1, dt_op2, dt_op3, dt_op4]


if sensitivity_target == "Lean OP2": lean_times = add_delta(lean_times, extra_seconds) elif sensitivity_target == "DT OP2": dt_times = add_delta(dt_times, extra_seconds) elif sensitivity_target == "Both OP2": lean_times = add_delta(lean_times, extra_seconds) dt_times = add_delta(dt_times, extra_seconds)


lean_results = run_paced_line(STATIONS, lean_times, release_interval) dt_results = run_paced_line(STATIONS, dt_times, release_interval)


============================================================


Top summary


============================================================


col1, col2 = st.columns(2) with col1: st.subheader("Lean configuration") st.metric("Throughput (units/hour)", f"{lean_results['throughput_per_hour']:.2f}") st.metric("Average system time (sec)", f"{lean_results['avg_system_time_sec']:.2f}") st.metric("Bottleneck", lean_results["bottleneck"]) st.metric("Theoretical capacity (units/hour)", f"{theoretical_capacity(lean_times):.2f}") with col2: st.subheader("Digital Model configuration") st.metric("Throughput (units/hour)", f"{dt_results['throughput_per_hour']:.2f}") st.metric("Average system time (sec)", f"{dt_results['avg_system_time_sec']:.2f}") st.metric("Bottleneck", dt_results["bottleneck"]) st.metric("Theoretical capacity (units/hour)", f"{theoretical_capacity(dt_times):.2f}")


============================================================


Warning flags


============================================================


st.subheader("Diagnostic warnings") lean_flags = warning_flags(lean_results, target_demand) dt_flags = warning_flags(dt_results, target_demand)


left, right = st.columns(2) with left: st.markdown("Lean configuration warnings") if not lean_flags: st.success("No critical warning flags detected.") for level, msg in lean_flags: if level == "error": st.error(msg) else: st.warning(msg) with right: st.markdown("Digital Model configuration warnings") if not dt_flags: st.success("No critical warning flags detected.") for level, msg in dt_flags: if level == "error": st.error(msg) else: st.warning(msg)


============================================================


Summary tables


============================================================


summary_df = pd.DataFrame([ { "Scenario": "Lean", "Target demand": target_demand, "Release interval (sec/unit)": round(release_interval, 2), "Throughput (units/hour)": round(lean_results["throughput_per_hour"], 2), "Average system time (sec)": round(lean_results["avg_system_time_sec"], 2), "Bottleneck": lean_results["bottleneck"], "Queue at OP2": round(lean_results["avg_queue"]["OP2"], 2), }, { "Scenario": "Digital Model", "Target demand": target_demand, "Release interval (sec/unit)": round(release_interval, 2), "Throughput (units/hour)": round(dt_results["throughput_per_hour"], 2), "Average system time (sec)": round(dt_results["avg_system_time_sec"], 2), "Bottleneck": dt_results["bottleneck"], "Queue at OP2": round(dt_results["avg_queue"]["OP2"], 2), }, ])


util_df = pd.DataFrame([ { "Resource": r, "Lean utilization": round(lean_results["utilization"][r], 4), "Digital Model utilization": round(dt_results["utilization"][r], 4), "Lean avg wait (sec)": round(lean_results["avg_waits"][r], 2), "Digital Model avg wait (sec)": round(dt_results["avg_waits"][r], 2), "Lean avg queue": round(lean_results["avg_queue"][r], 2), "Digital Model avg queue": round(dt_results["avg_queue"][r], 2), } for r in STATIONS ])


st.subheader("Summary table") st.dataframe(summary_df, use_container_width=True)


st.subheader("Resource table") st.dataframe(util_df, use_container_width=True)


============================================================


Charts


============================================================


st.subheader("Charts")


1) Throughput comparison


fig1, ax1 = plt.subplots(figsize=(7, 4)) ax1.bar(["Lean", "Digital Model"], [lean_results["throughput_per_hour"], dt_results["throughput_per_hour"]]) ax1.set_ylabel("Throughput (units/hour)") ax1.set_title(f"Throughput Comparison at {target_demand} units/hour demand") st.pyplot(fig1)


2) System time comparison


fig2, ax2 = plt.subplots(figsize=(7, 4)) ax2.bar(["Lean", "Digital Model"], [lean_results["avg_system_time_sec"], dt_results["avg_system_time_sec"]]) ax2.set_ylabel("Average System Time (sec)") ax2.set_title("Average System Time Comparison") st.pyplot(fig2)


3) Utilization comparison


fig3, ax3 = plt.subplots(figsize=(8, 4)) x = np.arange(len(STATIONS)) width = 0.35 ax3.bar(x - width/2, [lean_results["utilization"][r] for r in STATIONS], width, label="Lean") ax3.bar(x + width/2, [dt_results["utilization"][r] for r in STATIONS], width, label="Digital Model") ax3.set_xticks(x) ax3.set_xticklabels(STATIONS) ax3.set_ylabel("Utilization") ax3.set_title("Resource Utilization Comparison") ax3.legend() st.pyplot(fig3)


4) Queue evolution at OP2


fig4, ax4 = plt.subplots(figsize=(9, 4)) ax4.plot([t/60 for t in lean_results["queue_time"]["OP2"]], lean_results["queue_values"]["OP2"], label="Lean", linestyle="--") ax4.plot([t/60 for t in dt_results["queue_time"]["OP2"]], dt_results["queue_values"]["OP2"], label="Digital Model") ax4.set_xlabel("Time (minutes)") ax4.set_ylabel("Queue Length at OP2") ax4.set_title("Queue Evolution at OP2") ax4.legend() st.pyplot(fig4)


5) Sensitivity mini-chart if extra seconds used


if extra_seconds > 0: fig5, ax5 = plt.subplots(figsize=(7, 4)) labels = ["Lean", "Digital Model"] vals = [lean_results["throughput_per_hour"], dt_results["throughput_per_hour"]] ax5.bar(labels, vals) ax5.set_ylabel("Throughput (units/hour)") ax5.set_title(f"Sensitivity Result with +{extra_seconds} sec on {sensitivity_target}") st.pyplot(fig5)


============================================================


Export block for thesis use


============================================================


st.subheader("Thesis-ready text block") st.code( f""" Target demand: {target_demand} units/hour Release interval: {release_interval:.2f} sec/unit


Lean configuration:


Throughput = {lean_results['throughput_per_hour']:.2f} units/hour


Average system time = {lean_results['avg_system_time_sec']:.2f} sec


Bottleneck = {lean_results['bottleneck']}


Queue at OP2 = {lean_results['avg_queue']['OP2']:.2f}




Digital Model configuration:


Throughput = {dt_results['throughput_per_hour']:.2f} units/hour


Average system time = {dt_results['avg_system_time_sec']:.2f} sec


Bottleneck = {dt_results['bottleneck']}


Queue at OP2 = {dt_results['avg_queue']['OP2']:.2f} """.strip(), language="text" )