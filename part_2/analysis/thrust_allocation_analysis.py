"""
Thrust allocation analysis: sweep of the rotation penalty weight q.

Open-loop study (no controller, no vessel). A desired wrench tau_d(t) is fed
to ThrustAllocator every time step. The commands go through ThrusterSet with
dynamics on (same rate limits as in the simulator), and the actual actuator
state is fed back as u_now / alpha_now. This isolates the effect of q on the
allocator itself.

Run from the repo root (TMR4240_LAB):
    python -m part_2.analysis.thrust_allocation_analysis
    python -m part_2.analysis.thrust_allocation_analysis --scenario noisy --show

Scenarios
---------
rotating   : constant force turning slowly through 360 deg (necessary rotation)
steps      : constant force jumping between directions (necessary rotation)
noisy      : constant mean force plus wave-like oscillations. After the warm-up
             the mean needs no rotation, so all measured rotation is unnecessary
             (azimuth hunting). This is what the rotation penalty is meant for.
noisy_low  : as noisy, but with a low mean force, where hunting is worst

Figures are written to figures/thrust_allocation_analysis/.

Metrics per run (computed after the scenario's warm-up time)
---------------
rotation   : total azimuth rotation, sum_k sum_i |alpha_i[k] - alpha_i[k-1]|  [deg]
effort     : thrust effort, sum_k sum_i u_i[k]^2 * dt                          [kN^2 s]
act_res    : RMS of tau_alloc - tau_thr (actuator part of the residual)       [kN], [kNm]
track_res  : RMS of tau_d - tau_thr (total delivered vs desired)              [kN], [kNm]

tau_alloc is the wrench the commands would give with ideal actuators, and
tau_thr is the wrench the rate-limited actuators actually deliver.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


DT = 0.1                                     # same as run_case_part_2.py
Q_VALUES = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
Q_TIMESERIES = [0.0, 3.0, 10.0]
OUT_DIR = Path("figures") / "thrust_allocation_analysis"


# -----------------------------------------------------------------------------
# Scenarios: desired BODY wrench tau_d(t), shape (6,)
# -----------------------------------------------------------------------------

F_DEMAND = 40e3      # [N], well inside capacity so regime 1 dominates
T_ROT = 120.0        # [s], time for one full turn in the rotating scenario
STEP_DIRS_DEG = [0.0, 90.0, 180.0, 45.0, -120.0, 0.0]
STEP_HOLD = 30.0     # [s] per direction in the step scenario


def scenario_rotating(t):
    """Constant force whose direction turns slowly through 360 deg."""
    phi = 2.0 * np.pi * t / T_ROT
    return np.array([F_DEMAND * np.cos(phi), F_DEMAND * np.sin(phi), 0.0, 0.0, 0.0, 0.0])


def scenario_steps(t):
    """Constant force that jumps to a new direction every STEP_HOLD seconds."""
    idx = min(int(t // STEP_HOLD), len(STEP_DIRS_DEG) - 1)
    phi = np.radians(STEP_DIRS_DEG[idx])
    return np.array([F_DEMAND * np.cos(phi), F_DEMAND * np.sin(phi), 0.0, 0.0, 0.0, 0.0])


# Wave-like content: two sinusoids per DOF with periods in the wave band (5-12 s),
# so the demand direction wobbles around a constant mean.
NOISY_DIR_DEG = 30.0  # direction of the mean force [deg]
NOISY_OSC = [         # (DOF index in tau_d, amplitude, period [s], phase [rad])
    (0, 4e3, 8.0, 0.0), (0, 2e3, 5.3, 1.1),
    (1, 4e3, 9.5, 0.7), (1, 2e3, 6.1, 2.3),
    (5, 40e3, 11.0, 0.4), (5, 20e3, 7.2, 1.9),
]
NOISY_WARMUP = 60.0   # [s], time to settle on the mean before measuring


def _noisy(t, F_mean):
    phi = np.radians(NOISY_DIR_DEG)
    tau = np.array([F_mean * np.cos(phi), F_mean * np.sin(phi), 0.0, 0.0, 0.0, 0.0])
    for dof, amp, period, phase in NOISY_OSC:
        tau[dof] += amp * np.sin(2.0 * np.pi * t / period + phase)
    return tau


def scenario_noisy(t):
    """Moderate mean force (20 kN) with wave-like oscillations."""
    return _noisy(t, 20e3)


def scenario_noisy_low(t):
    """Low mean force (6 kN) with the same oscillations."""
    return _noisy(t, 6e3)


# name: (tau_d(t), duration [s], warm-up excluded from the metrics [s])
SCENARIOS = {
    "rotating": (scenario_rotating, 180.0, 0.0),
    "steps": (scenario_steps, STEP_HOLD * len(STEP_DIRS_DEG), 0.0),
    "noisy": (scenario_noisy, 300.0, NOISY_WARMUP),
    "noisy_low": (scenario_noisy_low, 300.0, NOISY_WARMUP),
}


# -----------------------------------------------------------------------------
# One open-loop run
# -----------------------------------------------------------------------------

def run(thrusters, q, tau_fun, T):
    alloc = ThrustAllocator(thrusters, q=q)
    ts = ThrusterSet(thrusters, dynamics=True)

    n_steps = int(round(T / DT))
    n = len(thrusters)
    t = np.arange(n_steps) * DT

    log = {
        "t": t,
        "tau_d": np.zeros((n_steps, 3)),
        "tau_alloc": np.zeros((n_steps, 3)),
        "tau_thr": np.zeros((n_steps, 3)),
        "u_cmd": np.zeros((n_steps, n)),
        "alpha_cmd": np.zeros((n_steps, n)),
        "u": np.zeros((n_steps, n)),
        "alpha": np.zeros((n_steps, n)),
    }

    for k, tk in enumerate(t):
        tau_d = tau_fun(tk)
        u_now = ts.get_thrusts()
        alpha_now = ts.get_angles()

        u_cmd, alpha_cmd = alloc.allocate(tk, DT, tau_d, u_now, alpha_now)
        u, alpha, tau_thr = ts.step(u_cmd, alpha_cmd, DT)

        log["tau_d"][k] = tau_d[[0, 1, 5]]
        log["tau_alloc"][k] = ts._wrench_from(u_cmd, alpha_cmd)
        log["tau_thr"][k] = tau_thr
        log["u_cmd"][k] = u_cmd
        log["alpha_cmd"][k] = alpha_cmd
        log["u"][k] = u
        log["alpha"][k] = alpha

    log["n_calls"] = alloc.n_calls
    log["n_triggered"] = alloc.n_triggered
    log["n_fallback"] = alloc.n_fallback
    return log


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _rms_force(d):
    # RMS of the planar force error norm, in kN
    return np.sqrt(np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)) / 1e3


def _rms_moment(d):
    # RMS of the yaw moment error, in kNm
    return np.sqrt(np.mean(d[:, 2] ** 2)) / 1e3


def metrics(log, thrusters, t_skip=0.0):
    az = [i for i, th in enumerate(thrusters) if th.kind == "azimuth"]
    keep = log["t"] >= t_skip
    alpha = log["alpha"][keep][:, az]
    dalpha = _wrap(np.diff(alpha, axis=0))

    act = (log["tau_alloc"] - log["tau_thr"])[keep]
    trk = (log["tau_d"] - log["tau_thr"])[keep]

    return {
        "rotation": np.degrees(np.sum(np.abs(dalpha))),
        "effort": np.sum(log["u"][keep] ** 2) * DT / 1e6,
        "act_res_F": _rms_force(act),
        "act_res_M": _rms_moment(act),
        "track_res_F": _rms_force(trk),
        "track_res_M": _rms_moment(trk),
        "slsqp_share": log["n_triggered"] / max(log["n_calls"], 1),
        "fallbacks": log["n_fallback"],
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_metrics_vs_q(results, scenario):
    keys = [
        ("rotation", "Total azimuth rotation [deg]"),
        ("effort", r"Thrust effort $\sum u^2 \Delta t$ [kN$^2$ s]"),
        ("act_res_F", r"RMS $\|\tau_{alloc} - \tau_{thr}\|$ force [kN]"),
        ("track_res_F", r"RMS $\|\tau_d - \tau_{thr}\|$ force [kN]"),
    ]
    q_pos = [q for q in results if q > 0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, label) in zip(axes.ravel(), keys):
        ax.semilogx(q_pos, [results[q][key] for q in q_pos], "o-")
        if 0.0 in results:
            ax.axhline(results[0.0][key], ls="--", color="gray", label="q = 0")
            ax.legend()
        ax.set_xlabel("q")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Rotation penalty sweep, scenario: {scenario}")
    fig.tight_layout()
    return fig


def plot_tradeoff(results, scenario):
    fig, ax = plt.subplots(figsize=(6, 5))
    qs = list(results)
    x = [results[q]["rotation"] for q in qs]
    y = [results[q]["effort"] for q in qs]
    ax.plot(x, y, "o-")
    for q, xi, yi in zip(qs, x, y):
        ax.annotate(f"q={q:g}", (xi, yi), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Total azimuth rotation [deg]")
    ax.set_ylabel(r"Thrust effort [kN$^2$ s]")
    ax.set_title(f"Rotation vs effort, scenario: {scenario}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_timeseries(logs, thrusters, scenario):
    az = [i for i, th in enumerate(thrusters) if th.kind == "azimuth"]
    fig, axes = plt.subplots(len(az) + 3, 1, figsize=(10, 2.2 * (len(az) + 3)), sharex=True)

    for q, log in logs.items():
        t = log["t"]
        for row, i in enumerate(az):
            axes[row].plot(t, np.degrees(np.unwrap(log["alpha"][:, i])), label=f"q={q:g}")  # unwrap: no false 360 deg jumps at +-180
            axes[row].set_ylabel(f"alpha {thrusters[i].name} [deg]")
        axes[len(az)].plot(t, np.linalg.norm(log["u"], axis=1) / 1e3, label=f"q={q:g}")
        axes[len(az) + 1].plot(t, log["tau_thr"][:, 0] / 1e3, label=f"q={q:g}")
        axes[len(az) + 2].plot(t, log["tau_thr"][:, 1] / 1e3, label=f"q={q:g}")

    ref = next(iter(logs.values()))
    axes[len(az) + 1].plot(ref["t"], ref["tau_d"][:, 0] / 1e3, "k--", lw=1, label="desired")
    axes[len(az) + 2].plot(ref["t"], ref["tau_d"][:, 1] / 1e3, "k--", lw=1, label="desired")

    axes[len(az)].set_ylabel(r"$\|u\|$ [kN]")
    axes[len(az) + 1].set_ylabel(r"$F_x$ delivered [kN]")
    axes[len(az) + 2].set_ylabel(r"$F_y$ delivered [kN]")
    axes[-1].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=len(logs))
    axes[-1].legend(loc="upper right", ncol=len(logs) + 1)
    fig.suptitle(f"Time series, scenario: {scenario}")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def print_table(results, scenario, t_skip=0.0):
    print(f"\nScenario: {scenario}" + (f"  (metrics from t = {t_skip:g} s)" if t_skip else ""))
    header = (f"{'q':>8} {'rot [deg]':>10} {'effort':>10} {'act F':>8} {'act M':>8} "
              f"{'trk F':>8} {'trk M':>8} {'SLSQP %':>8} {'fallb.':>7}")
    print(header)
    print("-" * len(header))
    for q, m in results.items():
        print(f"{q:>8g} {m['rotation']:>10.1f} {m['effort']:>10.1f} "
              f"{m['act_res_F']:>8.2f} {m['act_res_M']:>8.1f} "
              f"{m['track_res_F']:>8.2f} {m['track_res_M']:>8.1f} "
              f"{100 * m['slsqp_share']:>8.1f} {m['fallbacks']:>7d}")


def analyse(scenario, show):
    tau_fun, T, t_skip = SCENARIOS[scenario]
    thrusters = default_thrusters_part_2()

    logs = {q: run(thrusters, q, tau_fun, T) for q in Q_VALUES}
    results = {q: metrics(log, thrusters, t_skip) for q, log in logs.items()}
    print_table(results, scenario, t_skip)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "metrics_vs_q": plot_metrics_vs_q(results, scenario),
        "tradeoff": plot_tradeoff(results, scenario),
        "timeseries": plot_timeseries({q: logs[q] for q in Q_TIMESERIES if q in logs},
                                      thrusters, scenario),
    }
    for name, fig in figs.items():
        path = OUT_DIR / f"{scenario}_{name}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
    if not show:
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--show", action="store_true", help="show figures after saving")
    args = parser.parse_args()

    for sc in (SCENARIOS if args.scenario == "all" else [args.scenario]):
        analyse(sc, args.show)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()