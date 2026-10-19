"""
Thrust allocation analysis: steady-state (Part 1) vs sequential allocator, open loop.

Open-loop study (no controller, no vessel). A commanded wrench tau_c(t) is fed
to ThrustAllocator every time step. The commands go through ThrusterSet with
dynamics on (same rate limits as in the simulator), and the actual actuator
state is fed back as u_now / alpha_now. This isolates the allocator from the
rest of the DP loop.

Both methods are run on the same scenario and drawn in the same plots, as
recommended in the task text and by the lecturer:
    - commanded vs actual wrench, and the residual r_tau = tau_c - tau_thr
    - commanded vs actual thrust per thruster
    - commanded vs actual azimuth angle per azimuth

Run from the repo root (TMR4240_LAB):
    python -m part_2.analysis.thrust_allocation_analysis
    python -m part_2.analysis.thrust_allocation_analysis --scenario steps --show

Scenarios
---------
steps       : 40 kN jumping between directions every 30 s, incl. a 180 deg turn.
              Feasible, so only the actuator transient shows up in r_tau.
saturation  : infeasible demands (200 kN sway, then 150 kN + 150 kN).
              Shows delivered capacity and whether the direction is kept.
rotating    : 40 kN turning slowly through 360 deg (necessary, smooth rotation)
noisy       : 20 kN mean plus wave-like oscillations (sinusoids with wave periods).
              After the warm-up all rotation is unnecessary (azimuth hunting).
noisy_low   : as noisy, but 6 kN mean, where hunting is worst
random      : 20 kN mean plus band-limited random noise on Fx, Fy and Mz
              (seeded first-order filtered Gaussian noise, time constant 1 s),
              like a controller reacting to noisy measurements

Residual split (memory of the task text: keep steady state and transient apart)
-------------------------------------------------------------------------------
r_tau     = tau_c - tau_thr                      total residual
r_alloc   = tau_c - tau_alloc                    steady-state part: demand above capacity
r_act     = tau_alloc - tau_thr                  actuator part: thrust ramps and slewing
tau_alloc = T(alpha_cmd) u_cmd is the wrench the commands would give with ideal actuators.

Metrics (after the scenario's warm-up)
-------
rms r_tau F / M   : RMS of the total residual, force [kN] and moment [kNm]
rms r_alloc F     : RMS of the steady-state part, force [kN]
rms r_act F       : RMS of the actuator part, force [kN]
rotation          : total azimuth rotation, sum_k sum_i |alpha_i[k] - alpha_i[k-1]|  [deg]
effort            : sum_k sum_i u_i[k]^2 * dt                                     [kN^2 s]
capacity          : delivered force along the commanded direction / commanded, at the end [-]
direction error   : delivered force across the commanded direction at the end     [kN]
                    (both only when the command is constant at the end, else "-")

Figures are written to figures/thrust_allocation_analysis/.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


DT = 0.1                                     # same as run_case_part_2.py
OUT_DIR = Path("figures") / "thrust_allocation_analysis"

# name: (keyword arguments to ThrustAllocator, colour in the plots)
METHODS = {
    "steady-state": (dict(method="steady_state"), "C0"),
    "sequential": (dict(method="sequential"), "C1"),
}


# -----------------------------------------------------------------------------
# Scenarios: commanded BODY wrench tau_c(t), shape (6,)
# -----------------------------------------------------------------------------

F_DEMAND = 40e3      # [N], well inside capacity
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


SAT_HOLD = 60.0      # [s] per demand in the saturation scenario


def scenario_saturation(t):
    """Infeasible demands: 200 kN sway, then 150 kN surge + 150 kN sway."""
    if t < SAT_HOLD:
        return np.array([0.0, 200e3, 0.0, 0.0, 0.0, 0.0])
    return np.array([150e3, 150e3, 0.0, 0.0, 0.0, 0.0])


# Wave-like content: two sinusoids per DOF with periods in the wave band (5-12 s),
# so the demand direction wobbles around a constant mean.
NOISY_DIR_DEG = 30.0  # direction of the mean force [deg]
NOISY_OSC = [         # (DOF index in tau_c, amplitude, period [s], phase [rad])
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


# Random noise: constant mean force plus band-limited Gaussian noise on Fx, Fy and Mz,
# like a controller reacting to noisy measurements. Seeded, so both methods (and every run)
# see exactly the same demand.
RANDOM_T = 300.0                       # [s] length of the scenario
RANDOM_MEAN = (17e3, 10e3, 0.0)        # [N, N, Nm] mean demand (about 20 kN at 30 deg)
RANDOM_STD = (4e3, 4e3, 40e3)          # [N, N, Nm] standard deviation of the noise
RANDOM_TC = 1.0                        # [s] time constant of the first-order filter
RANDOM_SEED = 1


def _make_random_noise():
    rng = np.random.default_rng(RANDOM_SEED)
    n = int(round(RANDOM_T / DT)) + 1
    a = np.exp(-DT / RANDOM_TC)
    w = np.zeros((n, 3))
    for k in range(1, n):
        # first-order low-pass of white noise, scaled so the output has the wanted std
        w[k] = a * w[k - 1] + np.sqrt(1 - a ** 2) * rng.standard_normal(3)
    return np.asarray(RANDOM_MEAN) + w * np.asarray(RANDOM_STD)


RANDOM_SERIES = _make_random_noise()


def scenario_random(t):
    """Mean force (about 20 kN) with band-limited random noise."""
    k = min(int(round(t / DT)), len(RANDOM_SERIES) - 1)
    Fx, Fy, Mz = RANDOM_SERIES[k]
    return np.array([Fx, Fy, 0.0, 0.0, 0.0, Mz])


# name: (tau_c(t), duration [s], warm-up excluded from the metrics [s])
# (also imported by allocator_comparison.py)
SCENARIOS = {
    "steps": (scenario_steps, STEP_HOLD * len(STEP_DIRS_DEG), 0.0),
    "saturation": (scenario_saturation, 2 * SAT_HOLD, 0.0),
    "rotating": (scenario_rotating, 180.0, 0.0),
    "noisy": (scenario_noisy, 300.0, NOISY_WARMUP),
    "noisy_low": (scenario_noisy_low, 300.0, NOISY_WARMUP),
    "random": (scenario_random, RANDOM_T, NOISY_WARMUP),
}
DEFAULT_SCENARIOS = ["steps", "saturation", "rotating", "noisy", "noisy_low", "random"]


# -----------------------------------------------------------------------------
# One open-loop run
# -----------------------------------------------------------------------------

def run(thrusters, alloc_kwargs, tau_fun, T):
    alloc = ThrustAllocator(thrusters, **alloc_kwargs)
    ts = ThrusterSet(thrusters, dynamics=True)

    n_steps = int(round(T / DT))
    n = len(thrusters)
    t = np.arange(n_steps) * DT

    log = {
        "t": t,
        "tau_c": np.zeros((n_steps, 3)),
        "tau_alloc": np.zeros((n_steps, 3)),
        "tau_thr": np.zeros((n_steps, 3)),
        "u_cmd": np.zeros((n_steps, n)),
        "alpha_cmd": np.zeros((n_steps, n)),
        "u": np.zeros((n_steps, n)),
        "alpha": np.zeros((n_steps, n)),
    }

    for k, tk in enumerate(t):
        tau_c = tau_fun(tk)
        u_now = ts.get_thrusts()
        alpha_now = ts.get_angles()

        u_cmd, alpha_cmd = alloc.allocate(tk, DT, tau_c, u_now, alpha_now)
        u, alpha, tau_thr = ts.step(u_cmd, alpha_cmd, DT)

        log["tau_c"][k] = tau_c[[0, 1, 5]]
        log["tau_alloc"][k] = ts._wrench_from(u_cmd, alpha_cmd)
        log["tau_thr"][k] = tau_thr
        log["u_cmd"][k] = u_cmd
        log["alpha_cmd"][k] = alpha_cmd
        log["u"][k] = u
        log["alpha"][k] = alpha

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
    dalpha = _wrap(np.diff(log["alpha"][keep][:, az], axis=0))

    r_tau = (log["tau_c"] - log["tau_thr"])[keep]
    r_alloc = (log["tau_c"] - log["tau_alloc"])[keep]
    r_act = (log["tau_alloc"] - log["tau_thr"])[keep]

    # Delivered vs commanded at the end, along and across the commanded direction.
    # Only meaningful when the command is constant at the end (steps, saturation),
    # otherwise NaN (printed as "-").
    tau_c_end, tau_thr_end = log["tau_c"][-1] / 1e3, log["tau_thr"][-1] / 1e3
    last = log["t"] >= log["t"][-1] - 5.0
    if np.ptp(log["tau_c"][last], axis=0).max() < 1.0:
        d = tau_c_end / np.linalg.norm(tau_c_end)
        along = tau_thr_end @ d
        capacity, dir_error = along / np.linalg.norm(tau_c_end), np.linalg.norm(tau_thr_end - along * d)
    else:
        capacity, dir_error = np.nan, np.nan

    return {
        "rms_r_tau_F": _rms_force(r_tau),
        "rms_r_tau_M": _rms_moment(r_tau),
        "rms_r_alloc_F": _rms_force(r_alloc),
        "rms_r_act_F": _rms_force(r_act),
        "rotation": np.degrees(np.sum(np.abs(dalpha))),
        "effort": np.sum(log["u"][keep] ** 2) * DT / 1e6,
        "capacity": capacity,
        "dir_error": dir_error,
    }


# -----------------------------------------------------------------------------
# Plots: both methods in the same axes
# -----------------------------------------------------------------------------

WRENCH_ROWS = (("Fx [kN]", 0), ("Fy [kN]", 1), ("Mz [kNm]", 2))


def plot_wrench(logs, scenario):
    """Left: commanded vs actual wrench. Right: residual r_tau = tau_c - tau_thr."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), sharex=True)
    t = next(iter(logs.values()))["t"]
    tau_c = next(iter(logs.values()))["tau_c"]      # the same command for both methods
    for row, (label, j) in enumerate(WRENCH_ROWS):
        ax = axes[row, 0]
        ax.plot(t, tau_c[:, j] / 1e3, "k--", lw=1.2, label=r"commanded $\tau_c$")
        for name, log in logs.items():
            ax.plot(t, log["tau_thr"][:, j] / 1e3, color=METHODS[name][1],
                    label=rf"actual $\tau_{{thr}}$, {name}")
        ax.set_ylabel(label)

        ax = axes[row, 1]
        for name, log in logs.items():
            ax.plot(t, (log["tau_c"][:, j] - log["tau_thr"][:, j]) / 1e3,
                    color=METHODS[name][1], label=name)
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
        ax.set_ylabel(f"residual {label}")

    axes[0, 0].set_title(r"Commanded vs actual wrench")
    axes[0, 1].set_title(r"Residual $r_\tau = \tau_c - \tau_{thr}$")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].legend(fontsize=8, loc="best")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle(f"Wrench, open loop, scenario: {scenario}")
    fig.tight_layout()
    return fig


def plot_thrust(logs, thrusters, scenario):
    """Commanded (dashed) vs actual (solid) thrust per thruster."""
    fig, axes = plt.subplots(len(thrusters), 1, figsize=(12, 7.5), sharex=True)
    for i, (ax, thr) in enumerate(zip(axes, thrusters)):
        for name, log in logs.items():
            c = METHODS[name][1]
            ax.plot(log["t"], log["u_cmd"][:, i] / 1e3, "--", color=c, lw=1.0, label=f"commanded, {name}")
            ax.plot(log["t"], log["u"][:, i] / 1e3, color=c, label=f"actual, {name}")
        ax.axhline(thr.u_max / 1e3, color="grey", lw=0.8, ls=":")
        ax.axhline(-thr.u_max / 1e3, color="grey", lw=0.8, ls=":")
        ax.set_ylabel(f"{thr.name}\nthrust [kN]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Thrust, commanded vs actual (dotted: $\\pm u_{{max}}$), scenario: {scenario}")
    fig.tight_layout()
    return fig


def plot_azimuth(logs, thrusters, scenario):
    """Commanded (dashed) vs actual (solid) azimuth angle, unwrapped."""
    az = [i for i, th in enumerate(thrusters) if th.kind == "azimuth"]
    fig, axes = plt.subplots(len(az), 1, figsize=(12, 6), sharex=True)
    for ax, i in zip(np.atleast_1d(axes), az):
        for name, log in logs.items():
            c = METHODS[name][1]
            # unwrap: no false 360 deg jumps at +-180
            ax.plot(log["t"], np.degrees(np.unwrap(log["alpha_cmd"][:, i])), "--", color=c, lw=1.0,
                    label=f"commanded, {name}")
            ax.plot(log["t"], np.degrees(np.unwrap(log["alpha"][:, i])), color=c, label=f"actual, {name}")
        ax.set_ylabel(f"{thrusters[i].name}\nangle [deg]")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="best", ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Azimuth angle, commanded vs actual (unwrapped), scenario: {scenario}")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def print_table(results, scenario, t_skip=0.0):
    print(f"\nScenario: {scenario}" + (f"  (metrics from t = {t_skip:g} s)" if t_skip else ""))
    header = (f"{'method':>14} {'r_tau F':>8} {'r_tau M':>8} {'r_alloc F':>9} {'r_act F':>8} "
              f"{'rot [deg]':>10} {'effort':>9} {'capacity':>9} {'dir err':>8}")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:>14} {m['rms_r_tau_F']:>8.2f} {m['rms_r_tau_M']:>8.1f} {m['rms_r_alloc_F']:>9.2f} "
              f"{m['rms_r_act_F']:>8.2f} {m['rotation']:>10.1f} {m['effort']:>9.1f} "
              + (f"{m['capacity']:>9.3f} {m['dir_error']:>8.2f}" if np.isfinite(m["capacity"])
                 else f"{'-':>9} {'-':>8}"))


def analyse(scenario, show):
    tau_fun, T, t_skip = SCENARIOS[scenario]
    thrusters = default_thrusters_part_2()

    logs = {name: run(thrusters, kwargs, tau_fun, T) for name, (kwargs, _) in METHODS.items()}
    results = {name: metrics(log, thrusters, t_skip) for name, log in logs.items()}
    print_table(results, scenario, t_skip)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "wrench": plot_wrench(logs, scenario),
        "thrust": plot_thrust(logs, thrusters, scenario),
        "azimuth": plot_azimuth(logs, thrusters, scenario),
    }
    for key, fig in figs.items():
        path = OUT_DIR / f"{scenario}_{key}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
    if not show:
        plt.close("all")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    parser.add_argument("--show", action="store_true", help="show the figures as well")
    args = parser.parse_args()

    scenarios = DEFAULT_SCENARIOS if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        analyse(scenario, args.show)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()