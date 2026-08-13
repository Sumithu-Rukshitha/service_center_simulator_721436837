import random
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set base seed for statistical reproducibility across 30 runs
BASE_SEED = 42

class PriorityServiceCenterSim:
    def __init__(self, lambda_p, lambda_g, mean_sp, mean_sg, num_counters, sim_time, warmup_time, dedicated_p=0):
        # Arrival rates in arrivals per second
        self.lambda_p = lambda_p / 3600.0
        self.lambda_g = lambda_g / 3600.0
        self.mean_sp = mean_sp  # Service time in seconds
        self.mean_sg = mean_sg

        self.num_counters = num_counters
        self.dedicated_p = dedicated_p  # Dedicated counters for Priority (if Option A)
        self.sim_time = sim_time
        self.warmup_time = warmup_time

        self.clock = 0.0
        self.event_queue = []  # Heap priority queue: (time, event_type, priority_class, cust_id)

        # Customer queues
        self.waiting_p = []  # Priority Queue
        self.waiting_g = []  # General Queue

        # Server states
        self.busy_counters_shared = 0
        self.busy_counters_p = 0  # For dedicated counters

        self.cust_id_counter = 0
        self.records = []

    def schedule_event(self, time, event_type, priority_class, cust_id):
        heapq.heappush(self.event_queue, (time, event_type, priority_class, cust_id))

    def get_service_time(self, priority_class):
        mean = self.mean_sp if priority_class == 0 else self.mean_sg
        sigma = 0.4  # Lognormal variability as specified in Section 2.2
        mu = np.log(mean) - (sigma**2 / 2)
        return np.random.lognormal(mu, sigma)

    def run(self):
        # Schedule first arrivals
        self.schedule_event(np.random.exponential(1.0 / self.lambda_p), "ARRIVAL", 0, self._next_cust_id())
        self.schedule_event(np.random.exponential(1.0 / self.lambda_g), "ARRIVAL", 1, self._next_cust_id())

        while self.event_queue:
            time, event_type, priority_class, cust_id = heapq.heappop(self.event_queue)
            if time > self.sim_time:
                break
            self.clock = time

            if event_type == "ARRIVAL":
                self._handle_arrival(priority_class, cust_id)
            elif event_type == "DEPARTURE":
                self._handle_departure(priority_class, cust_id)

    def _next_cust_id(self):
        self.cust_id_counter += 1
        return self.cust_id_counter

    def _handle_arrival(self, priority_class, cust_id):
        # Schedule next arrival for the same class
        rate = self.lambda_p if priority_class == 0 else self.lambda_g
        self.schedule_event(self.clock + np.random.exponential(1.0 / rate), "ARRIVAL", priority_class, self._next_cust_id())

        arrival_time = self.clock

        # Dedicated Counter Routing Logic (Option A) vs Shared Counter Pool
        if priority_class == 0 and self.dedicated_p > 0:
            if self.busy_counters_p < self.dedicated_p:
                self.busy_counters_p += 1
                service_time = self.get_service_time(priority_class)
                self.schedule_event(self.clock + service_time, "DEPARTURE", priority_class, cust_id)
                self._log_record(cust_id, priority_class, arrival_time, 0.0, service_time)
                return

        # Shared counters check
        shared_capacity = self.num_counters - self.dedicated_p
        if self.busy_counters_shared < shared_capacity:
            self.busy_counters_shared += 1
            service_time = self.get_service_time(priority_class)
            self.schedule_event(self.clock + service_time, "DEPARTURE", priority_class, cust_id)
            self._log_record(cust_id, priority_class, arrival_time, 0.0, service_time)
        else:
            if priority_class == 0:
                self.waiting_p.append((cust_id, arrival_time))
            else:
                self.waiting_g.append((cust_id, arrival_time))

    def _handle_departure(self, priority_class, cust_id):
        # Handle dedicated departure
        if priority_class == 0 and self.dedicated_p > 0 and self.busy_counters_p > 0:
            if len(self.waiting_p) > 0:
                next_cust_id, arr_time = self.waiting_p.pop(0)
                wait_time = self.clock - arr_time
                service_time = self.get_service_time(0)
                self.schedule_event(self.clock + service_time, "DEPARTURE", 0, next_cust_id)
                self._log_record(next_cust_id, 0, arr_time, wait_time, service_time)
            else:
                self.busy_counters_p -= 1
            return

        # Handle shared pool departure
        self.busy_counters_shared -= 1

        # Non-preemptive priority dispatching: Check Priority Queue first, then General Queue
        if len(self.waiting_p) > 0:
            next_cust_id, arr_time = self.waiting_p.pop(0)
            next_class = 0
        elif len(self.waiting_g) > 0:
            next_cust_id, arr_time = self.waiting_g.pop(0)
            next_class = 1
        else:
            return

        self.busy_counters_shared += 1
        wait_time = self.clock - arr_time
        service_time = self.get_service_time(next_class)
        self.schedule_event(self.clock + service_time, "DEPARTURE", next_class, next_cust_id)
        self._log_record(next_cust_id, next_class, arr_time, wait_time, service_time)

    def _log_record(self, cust_id, priority_class, arrival_time, wait_time, service_time):
        # Purge transient data collected during warm-up period
        if self.clock >= self.warmup_time:
            self.records.append({
                'id': cust_id,
                'class': priority_class,
                'arrival': arrival_time,
                'wait_time': wait_time,
                'service_time': service_time
            })



# 1. RUN SECTION 4.1 BASELINE SIMULATION (30 REPLICATIONS)

def run_baseline_experiment():
    print("Executing 30 Simulation Replications for Section 4.1 Baseline...")
    all_runs = []

    for seed in range(1, 31):
        np.random.seed(BASE_SEED + seed)
        random.seed(BASE_SEED + seed)

        sim = PriorityServiceCenterSim(
            lambda_p=20, lambda_g=60,
            mean_sp=240, mean_sg=360,
            num_counters=5,
            sim_time=28800, warmup_time=7200
        )
        sim.run()
        df = pd.DataFrame(sim.records)
        df['run_id'] = seed
        all_runs.append(df)

    full_df = pd.concat(all_runs, ignore_index=True)

    # Calculate Summary Statistics (Table 4.1)
    print("\n=======================================================")
    print("TABLE 4.1: STEADY-STATE PERFORMANCE SUMMARY (30 RUNS)")
    print("=======================================================")
    for c_type, label in [(0, "Priority Class"), (1, "General Class")]:
        sub = full_df[full_df['class'] == c_type]
        mean_w = sub['wait_time'].mean() / 60.0
        p50_w = sub['wait_time'].median() / 60.0
        p95_w = np.percentile(sub['wait_time'], 95) / 60.0
        p99_w = np.percentile(sub['wait_time'], 99) / 60.0
        print(f"{label:15s} | Mean: {mean_w:5.2f}m | P50: {p50_w:5.2f}m | P95: {p95_w:5.2f}m | P99: {p99_w:5.2f}m")

    return full_df


# 2. RUN SECTION 6 CAPACITY PLANNING EXPERIMENTS

def run_capacity_planning_experiments():
    print("\n=======================================================")
    print("SECTION 6: CAPACITY PLANNING & STRATEGY EVALUATION")
    print("=======================================================")

    configs = [
        ("Unmodified (+35% Traffic)", 27, 81, 5, 0),
        ("Option A (1 Dedicated / 5 General)", 27, 81, 6, 1),
        ("Option B (7 Shared Counters)", 27, 81, 7, 0)
    ]

    for name, lp, lg, counters, ded_p in configs:
        p95_p_runs, p95_g_runs = [], []
        for seed in range(1, 31):
            np.random.seed(BASE_SEED + seed)
            random.seed(BASE_SEED + seed)

            sim = PriorityServiceCenterSim(
                lambda_p=lp, lambda_g=lg,
                mean_sp=240, mean_sg=360,
                num_counters=counters, dedicated_p=ded_p,
                sim_time=28800, warmup_time=7200
            )
            sim.run()
            df = pd.DataFrame(sim.records)

            p_sub = df[df['class'] == 0]
            g_sub = df[df['class'] == 1]

            p95_p_runs.append(np.percentile(p_sub['wait_time'], 95) / 60.0 if len(p_sub) > 0 else 0)
            p95_g_runs.append(np.percentile(g_sub['wait_time'], 95) / 60.0 if len(g_sub) > 0 else 0)

        print(f"Scenario: {name:35s} | Priority P95: {np.mean(p95_p_runs):5.1f} mins | General P95: {np.mean(p95_g_runs):5.1f} mins")


# 3. GENERATE SECTION 5 VISUALIZATIONS

def generate_visualizations(df):
    print("\nGenerating Report Figures (Saving as PNG images)...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Cumulative Distribution Function (CDF) Plot
    plt.figure(figsize=(8, 5))
    for c_type, label, color in [(0, "Priority Class (Class 0)", "blue"), (1, "General Class (Class 1)", "red")]:
        sub = df[df['class'] == c_type]['wait_time'] / 60.0
        sorted_data = np.sort(sub)
        yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
        plt.plot(sorted_data, yvals, label=label, color=color, linewidth=2)

    plt.axvline(x=5.0, color='green', linestyle='--', label='Priority SLA Limit (5 mins)')
    plt.axhline(y=0.95, color='gray', linestyle=':', label='95th Percentile Target')
    plt.title('Cumulative Distribution Function (CDF) of Wait Times', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Wait Time (minutes)')
    plt.ylabel('Cumulative Probability P(Wait <= x)')
    plt.xlim(0, 30)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('cdf_plot.png', dpi=300)
    plt.close()

    # 2. Box Plot of Wait Times
    plt.figure(figsize=(7, 5))
    p_wait = df[df['class'] == 0]['wait_time'] / 60.0
    g_wait = df[df['class'] == 1]['wait_time'] / 60.0

    plt.boxplot([p_wait, g_wait], tick_labels=['Priority Class', 'General Class'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
    plt.title('Wait Time Distribution & Tail Outliers by Class', fontsize=12, fontweight='bold')
    plt.ylabel('Wait Time (minutes)')
    plt.ylim(0, 35)
    plt.tight_layout()
    plt.savefig('boxplot_wait_times.png', dpi=300)
    plt.close()

    # 3. Load vs Response Curve (Knee Chart)
    plt.figure(figsize=(8, 5))
    arrival_multipliers = np.linspace(0.5, 1.3, 8)
    p95_p_curve, p95_g_curve = [], []

    for mult in arrival_multipliers:
        sim = PriorityServiceCenterSim(
            lambda_p=20 * mult, lambda_g=60 * mult,
            mean_sp=240, mean_sg=360,
            num_counters=5, sim_time=28800, warmup_time=7200
        )
        sim.run()
        res_df = pd.DataFrame(sim.records)
        p95_p_curve.append(np.percentile(res_df[res_df['class'] == 0]['wait_time'], 95) / 60.0)
        p95_g_curve.append(np.percentile(res_df[res_df['class'] == 1]['wait_time'], 95) / 60.0)

    rates = arrival_multipliers * 80  # Total arrival rate per hour
    plt.plot(rates, p95_p_curve, marker='o', color='blue', label='Priority P95 Wait')
    plt.plot(rates, p95_g_curve, marker='s', color='red', label='General P95 Wait')
    plt.axvline(x=90, color='orange', linestyle='--', label='System Knee / Saturation Threshold')
    plt.title('Load vs. Response Time Curve (P95 Latency)', fontsize=12, fontweight='bold')
    plt.xlabel('Total Customer Arrival Rate (requests/hour)')
    plt.ylabel('P95 Wait Time (minutes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('load_vs_response_curve.png', dpi=300)
    plt.close()

    print("Saved 'cdf_plot.png', 'boxplot_wait_times.png', and 'load_vs_response_curve.png'!")

# Execute complete workflow
if __name__ == "__main__":
    baseline_df = run_baseline_experiment()
    run_capacity_planning_experiments()
    generate_visualizations(baseline_df)
