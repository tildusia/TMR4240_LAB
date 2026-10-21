"""
Thrust allocation analysis for the report: steady-state (Part 1) vs JFB allocator, open loop.

Open-loop study (no controller, no vessel). A commanded wrench tau_c(t) is fed
to ThrustAllocator every time step. The commands go through ThrusterSet with
dynamics on (same limits as in the simulator), and the actual actuator state
is fed back as u_now / alpha_now. This isolates the allocator from the rest of
the DP loop.

Only what the report needs to justify the choice of allocator:

Figures (figures/thrust_allocation_analysis/)
    saturation_residual.png : the residual r_tau = tau_c - tau_thr, and the same residual split
                              into an allocation part and an actuator part, r_tau = r_alloc + r_act,
                              r_alloc = tau_c - tau_alloc   (demand above capacity, persistent)
                              r_act   = tau_alloc - tau_thr (thrust ramps and rotation, transient)
                              with tau_alloc = B(alpha_cmd) u_cmd. Shows which limit is active.
    noisy_low_azimuth.png   : commanded and actual azimuth angle, and the force residual |r_tau|.
                              Shows azimuth hunting (steady-state) vs none (JFB).
    noisy_low_rotation.png  : direction of the demanded and delivered force, cumulative azimuth rotation
                              sum_k sum_i |dalpha_i[k]| and azimuth thrust over time. Shows that
                              JFB follows the changing demand with thrust instead of rotation.

    capacity_polar.png      : delivered force in every direction for a 400 kN demand, ideal actuators.

Tables (printed, and saved to tables.txt in the same folder)
    1. Rotation and effort for every scenario, both methods (what JFB preserves and sacrifices)
    2. Delivered wrench for infeasible demands with ideal actuators (capacity, direction kept)
    3. JFB singularity weight rho: final azimuth angle and delivered sway force

Run from the repo root (TMR4240_LAB):
    python -m part_2.analysis.thrust_allocation_analysis
    python -m part_2.analysis.thrust_allocation_analysis --show

Scenarios
---------
steps       : 40 kN jumping between directions every 30 s, incl. a 180 deg turn.
saturation  : infeasible demands (200 kN sway, then 150 kN + 150 kN).
rotating    : 40 kN turning slowly through 360 deg (necessary, smooth rotation)
noisy       : 20 kN mean plus wave-like oscillations (sinusoids with wave periods).
noisy_low   : as noisy, but 6 kN mean, where hunting is worst
random      : 20 kN mean plus band-limited random noise on Fx, Fy and Mz
              (seeded first-order filtered Gaussian noise, time constant 1 s)
Metrics are taken after each scenario's warm-up (60 s for the noisy ones).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


DT = 0.1                                     # same as run_case_part_2.py
# CHANGED: relative to the repo root (this file is part_2/analysis/), not to the terminal folder
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "thrust_allocation_analysis"

# name: (keyword arguments to ThrustAllocator, colour in the plots)
METHODS = {
    "steady-state": (dict(method="steady_state"), "C0"),
    "jfb": (dict(method="jfb"), "C1"),
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


# Random noise: constant mean force plus band-limited Gaussian noise on Fx, Fy and Mz.
# Seeded, so both methods (and every run) see exactly the same demand.
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


# name: (tau_c(t), duration [s], warm-up excluded from the RMS [s])
# (also imported by allocator_comparison.py)
SCENARIOS = {
    "steps": (scenario_steps, STEP_HOLD * len(STEP_DIRS_DEG), 0.0),
    "saturation": (scenario_saturation, 2 * SAT_HOLD, 0.0),
    "rotating": (scenario_rotating, 180.0, 0.0),
    "noisy": (scenario_noisy, 300.0, NOISY_WARMUP),
    "noisy_low": (scenario_noisy_low, 300.0, NOISY_WARMUP),
    "random": (scenario_random, RANDOM_T, NOISY_WARMUP),
}



# -----------------------------------------------------------------------------
# One open-loop run
# -----------------------------------------------------------------------------

def run(thrusters, alloc_kwargs, tau_fun, T, dynamics=True):
    alloc = ThrustAllocator(thrusters, **alloc_kwargs)
    ts = ThrusterSet(thrusters, dynamics=dynamics)

    n_steps = int(round(T / DT))
    n = len(thrusters)
    t = np.arange(n_steps) * DT
    log = {
        "t": t,
        "tau_c": np.zeros((n_steps, 3)),
        "tau_alloc": np.zeros((n_steps, 3)),
        "tau_thr": np.zeros((n_steps, 3)),
        "alpha_cmd": np.zeros((n_steps, n)),
        "u": np.zeros((n_steps, n)),
        "alpha": np.zeros((n_steps, n)),
    }

    for k, tk in enumerate(t):
        tau_c = tau_fun(tk)
        u_cmd, alpha_cmd = alloc.allocate(tk, DT, tau_c, ts.get_thrusts(), ts.get_angles())
        u, alpha, tau_thr = ts.step(u_cmd, alpha_cmd, DT)

        log["tau_c"][k] = tau_c[[0, 1, 5]]
        log["tau_alloc"][k] = ts._wrench_from(u_cmd, alpha_cmd)  # B(alpha_cmd) u_cmd
        log["tau_thr"][k] = tau_thr
        log["alpha_cmd"][k] = alpha_cmd
        log["u"][k] = u
        log["alpha"][k] = alpha

    return log


def _azimuths(thrusters):
    return [i for i, th in enumerate(thrusters) if th.kind == "azimuth"]


def cumulative_rotation(log, thrusters):
    # sum_i |dalpha_i[k]| accumulated over time [deg], wrapped so a +-pi crossing is not 2 pi
    az = _azimuths(thrusters)
    d = np.diff(log["alpha"][:, az], axis=0)
    d = np.arctan2(np.sin(d), np.cos(d))
    return np.degrees(np.concatenate([[0.0], np.cumsum(np.sum(np.abs(d), axis=1))]))


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

ROWS = (("Fx [kN]", 0), ("Fy [kN]", 1), ("Mz [kNm]", 2))


def plot_saturation_residual(logs):
    """r_tau (left), r_alloc (middle) and r_act (right) for both methods, saturation scenario.
    r_tau = r_alloc + r_act."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 7.5), sharex=True, sharey="row")
    for row, (label, j) in enumerate(ROWS):
        for name, log in logs.items():
            c = METHODS[name][1]
            axes[row, 0].plot(log["t"], (log["tau_c"][:, j] - log["tau_thr"][:, j]) / 1e3, color=c, label=name)
            axes[row, 1].plot(log["t"], (log["tau_c"][:, j] - log["tau_alloc"][:, j]) / 1e3, color=c, label=name)
            axes[row, 2].plot(log["t"], (log["tau_alloc"][:, j] - log["tau_thr"][:, j]) / 1e3, color=c, label=name)
        axes[row, 0].set_ylabel(label)
    axes[0, 0].set_title(r"Total $r_\tau = \tau_c - \tau_{thr}$")
    axes[0, 1].set_title(r"Allocation part $r_{alloc} = \tau_c - \tau_{alloc}$")
    axes[0, 2].set_title(r"Actuator part $r_{act} = \tau_{alloc} - \tau_{thr}$")
    axes[0, 0].legend(fontsize=9, loc="best")
    for ax in axes.flat:
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Residual, saturation scenario (200 kN sway, then 150 kN + 150 kN)")
    fig.tight_layout()
    return fig


def plot_noisy_low_azimuth(logs, thrusters, t_skip):
    """Commanded (dashed) and actual (solid) azimuth angle, unwrapped, and |r_tau| force."""
    az = _azimuths(thrusters)
    fig, axes = plt.subplots(len(az) + 1, 1, figsize=(11, 7), sharex=True)
    for ax, i in zip(axes, az):
        for name, log in logs.items():
            c = METHODS[name][1]
            ax.plot(log["t"], np.degrees(np.unwrap(log["alpha_cmd"][:, i])), "--", color=c, lw=1.0,
                    label=f"commanded, {name}")
            ax.plot(log["t"], np.degrees(np.unwrap(log["alpha"][:, i])), color=c, label=f"actual, {name}")
        ax.set_ylabel(f"{thrusters[i].name}\nangle [deg]")
    for name, log in logs.items():
        r = log["tau_c"][:, :2] - log["tau_thr"][:, :2]
        axes[-1].plot(log["t"], np.hypot(r[:, 0], r[:, 1]) / 1e3, color=METHODS[name][1], label=name)
    axes[-1].set_ylabel(r"$|r_\tau|$ force [kN]")
    axes[0].legend(fontsize=8, loc="best", ncol=2)
    for ax in axes:
        ax.axvline(t_skip, color="grey", lw=0.8, ls=":")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Azimuth angle and force residual, noisy_low (6 kN mean + wave-like oscillations)")
    fig.tight_layout()
    return fig


def plot_noisy_low_rotation(logs, thrusters, t_skip):
    """Direction of the demanded and delivered force, cumulative azimuth rotation and azimuth thrust.
    Shows that JFB follows the changing demand with thrust instead of rotation."""
    az = _azimuths(thrusters)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    first = next(iter(logs.values()))
    keep = first["t"] >= t_skip
    t = first["t"][keep]

    # Panel 1: direction of the demanded force, the same for both methods
    # Deviation from the mean direction, wrapped to +-180 deg. Not unwrapped: when the force
    # is close to zero the direction can flip by 180 deg, which unwrap would turn into false turns.
    def direction(tau):
        beta = np.arctan2(tau[:, 1], tau[:, 0]) - np.radians(NOISY_DIR_DEG)
        beta = np.degrees(np.arctan2(np.sin(beta), np.cos(beta)))
        beta[1:][np.abs(np.diff(beta)) > 180.0] = np.nan  # no vertical lines at the +-180 deg jumps
        return beta

    # Direction of the delivered force tau_thr for each method, and the demand dashed on top
    # (JFB delivers the demanded direction so closely that a solid demand line would be hidden)
    for name, log in logs.items():
        axes[0].plot(t, direction(log["tau_thr"][keep]), color=METHODS[name][1], lw=1.4,
                     label=rf"delivered $\tau_{{thr}}$, {name}")
    axes[0].plot(t, direction(first["tau_c"][keep]), color="black", lw=1.0, ls="--", zorder=5,
                 label=r"demanded $\tau_c$")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")
    axes[0].set_ylim(-185, 185)
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel("Force\ndirection [deg]")
    axes[0].set_title("Direction of the demanded and the delivered force, relative to the mean "
                      + f"({NOISY_DIR_DEG:g} deg)")

    for name, log in logs.items():
        c = METHODS[name][1]
        # Panel 2: cumulative rotation
        rot = cumulative_rotation({k: v[keep] for k, v in log.items()}, thrusters)
        axes[1].plot(t, rot, color=c, label=f"{name} ({rot[-1]:.0f} deg)")
        # Panel 3: actual thrust of each azimuth (solid Azimuth 1, dashed Azimuth 2)
        for i, ls in zip(az, ["-", "--"]):
            axes[2].plot(t, log["u"][keep, i] / 1e3, color=c, ls=ls, lw=1.0, label=f"{thrusters[i].name}, {name}")

    axes[1].set_ylabel("Cumulative\nrotation [deg]")
    axes[1].set_title("Total azimuth rotation")
    axes[1].legend(fontsize=9)
    axes[2].set_ylabel("Azimuth\nthrust [kN]")
    axes[2].set_title("Actual azimuth thrust (signed)")
    axes[2].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"noisy_low from t = {t_skip:g} s: changing demand followed by rotation or by thrust")
    fig.tight_layout()
    return fig


CAPACITY_DEMAND = 400e3            # [N], far above capacity in every direction
CAPACITY_DIRS_DEG = np.arange(0.0, 360.0, 10.0)
CAPACITY_T = 60.0                  # [s] JFB needs time to rotate (180 deg takes 15 s)


def capacity_polar(thrusters):
    """Delivered force along each direction for a 400 kN demand (Mz = 0), ideal actuators.
    Returns {method: delivered force [kN] per direction} and the largest error across the
    demanded direction [kN] (should be ~0 when the direction is kept)."""
    result, cross = {}, {}
    for name, (kwargs, _) in METHODS.items():
        along, across = [], []
        for deg in CAPACITY_DIRS_DEG:
            phi = np.radians(deg)
            e = np.array([np.cos(phi), np.sin(phi)])
            tau_fun = _constant(CAPACITY_DEMAND * e[0], CAPACITY_DEMAND * e[1], 0.0)
            # The steady-state method has no memory, so one call is its final answer
            T = DT if kwargs["method"] == "steady_state" else CAPACITY_T
            log = run(thrusters, kwargs, tau_fun, T, dynamics=False)
            F = log["tau_thr"][-1, :2] / 1e3
            along.append(F @ e)
            across.append(abs(F @ np.array([-e[1], e[0]])))
        result[name] = np.array(along)
        cross[name] = max(across)
    return result, cross


def plot_capacity_polar(result):
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="polar")
    th = np.radians(np.append(CAPACITY_DIRS_DEG, CAPACITY_DIRS_DEG[0]))
    for name, F in result.items():
        ax.plot(th, np.append(F, F[0]), color=METHODS[name][1], marker="o", ms=3, label=name)
    # Body frame: x forward (up in the plot), y starboard (right), angle clockwise from x
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Delivered force for a {CAPACITY_DEMAND / 1e3:.0f} kN demand, Mz = 0, ideal actuators\n"
                 "[kN], 0 deg = surge (bow), 90 deg = sway (starboard)", fontsize=10)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1), fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def table_rotation_effort(thrusters):
    """Rotation [deg] and effort sum u^2 dt [kN^2 s] after the warm-up, all scenarios."""
    names = list(METHODS)
    lines = ["Table 1: rotation [deg] and effort [kN^2 s] after the warm-up",
             f"{'scenario':>11} " + " ".join(f"{'rot ' + m:>18} {'effort ' + m:>20}" for m in names)
             + f" {'effort change':>14}"]
    for sc, (tau_fun, T, t_skip) in SCENARIOS.items():
        row, effort = [], []
        for name in names:
            log = run(thrusters, METHODS[name][0], tau_fun, T)
            keep = log["t"] >= t_skip
            rot = cumulative_rotation({k: v[keep] for k, v in log.items()}, thrusters)[-1]
            eff = np.sum(log["u"][keep] ** 2) * DT / 1e6
            row.append(f"{rot:>18.0f} {eff:>20.0f}")
            effort.append(eff)
        lines.append(f"{sc:>11} " + " ".join(row) + f" {100 * (effort[1] / effort[0] - 1):>+13.0f}%")
    return lines


CAPACITY_CASES = {      # infeasible demands [Fx N, Fy N, Mz Nm]
    "400 kN sway": (0.0, 400e3, 0.0),
    "200 + 200 kN": (200e3, 200e3, 0.0),
    "3000 kNm yaw": (0.0, 0.0, 3000e3),
}


def _constant(Fx, Fy, Mz):
    return lambda t: np.array([Fx, Fy, 0.0, 0.0, 0.0, Mz])


def table_capacity(thrusters, T=60.0):
    """Delivered wrench after T seconds with IDEAL actuators (as in the course check)."""
    lines = [f"Table 2: delivered [Fx kN, Fy kN, Mz kNm] after {T:g} s, ideal actuators"]
    for case, tc in CAPACITY_CASES.items():
        cells = []
        for name, (kwargs, _) in METHODS.items():
            log = run(thrusters, kwargs, _constant(*tc), T, dynamics=False)
            d = log["tau_thr"][-1] / 1e3
            cells.append(f"{name}: [{d[0]:6.1f}, {d[1]:6.1f}, {d[2]:7.1f}]")
        lines.append(f"{case:>13}   " + "   ".join(cells))
    return lines


RHO_VALUES = [0.0, 3e-4, 1e-3]
RHO_CASES = {"60 kN sway": (0.0, 60e3, 0.0), "400 kN sway": (0.0, 400e3, 0.0)}


def table_rho(thrusters, T=60.0):
    """JFB with different rho: final azimuth angles and delivered sway, ideal actuators."""
    az = _azimuths(thrusters)
    lines = [f"Table 3: JFB singularity weight rho, after {T:g} s, ideal actuators"]
    for rho in RHO_VALUES:
        cells = []
        for case, tc in RHO_CASES.items():
            log = run(thrusters, dict(method="jfb", rho=rho), _constant(*tc), T, dynamics=False)
            ang = np.degrees(log["alpha"][-1, az])
            cells.append(f"{case}: angles [{ang[0]:6.1f}, {ang[1]:6.1f}] deg, Fy {log['tau_thr'][-1, 1] / 1e3:5.1f} kN")
        lines.append(f"rho = {rho:<7g}  " + "   ".join(cells))
    return lines


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show the figures as well")
    args = parser.parse_args()

    thrusters = default_thrusters_part_2()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figures
    tau_fun, T, _ = SCENARIOS["saturation"]
    logs = {name: run(thrusters, kwargs, tau_fun, T) for name, (kwargs, _) in METHODS.items()}
    figs = {"saturation_residual": plot_saturation_residual(logs)}

    tau_fun, T, t_skip = SCENARIOS["noisy_low"]
    logs = {name: run(thrusters, kwargs, tau_fun, T) for name, (kwargs, _) in METHODS.items()}
    figs["noisy_low_azimuth"] = plot_noisy_low_azimuth(logs, thrusters, t_skip)
    figs["noisy_low_rotation"] = plot_noisy_low_rotation(logs, thrusters, t_skip)

    cap, cap_cross = capacity_polar(thrusters)
    figs["capacity_polar"] = plot_capacity_polar(cap)

    for key, fig in figs.items():
        path = OUT_DIR / f"{key}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")

    # Tables
    lines = table_rotation_effort(thrusters) + [""] + table_capacity(thrusters) + [""] + table_rho(thrusters)
    lines += ["", f"Capacity polar ({CAPACITY_DEMAND / 1e3:.0f} kN demand): delivered force [kN] min / max over directions, "
              "largest error across the demanded direction [kN]"]
    for name, F in cap.items():
        lines.append(f"{name:>13}: {F.min():6.1f} / {F.max():6.1f} kN, across {cap_cross[name]:.2f} kN")
    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "tables.txt").write_text(text + "\n")
    print(f"\nsaved {OUT_DIR / 'tables.txt'}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()