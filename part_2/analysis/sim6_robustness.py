"""Simulation 6 analysis: heavy-sea robustness of the selected observer.

Runs the design sea (Hs 1.5 m, Tp 8 s, 1000 s) and the heavy sea (Hs 2.5 m,
Tp 10 s, 1000 s) with identical seed, environment and observer, evaluates both
over the same-length window and saves the figures and metrics.

    python -m part_2.analysis.sim6_robustness
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_case_part_2 as rc
from part_2.config import Part2SimConfig, default_thrusters_part_2
from part_2.utilization import thrust_utilization
from simulation.capability import low_frequency_motion

T_RUN, WINDOW = 1000.0, 300.0     # evaluation window = last WINDOW s of the run
SEAS = {"design": (rc.WAVES_HS, rc.WAVES_TP), "heavy": (rc.ROBUST_HS, rc.ROBUST_TP)}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def run_sea(hs, tp):
    cfg = Part2SimConfig(dt=rc.DT, T=T_RUN, use_observer=True,
                         observer_type=rc.selected_observer())
    return rc.run(cfg, np.zeros(6), rc.environment(hs=hs, tp=tp))


def moving_avg(x, dt, w=30.0):
    k = int(round(w / dt))
    xp = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, "valid")


def metrics(logs, thrusters, t0, t1):
    dt = float(logs.t[1] - logs.t[0])
    sel = (logs.t >= t0) & (logs.t <= t1)
    lf = low_frequency_motion(logs)
    # LF values near the run edges are biased by the padding; trim half a filter window
    inner = sel & (logs.t >= 15.0) & (logs.t <= logs.t[-1] - 15.0)
    pos_tot = np.hypot(logs.eta[:, 0], logs.eta[:, 1])
    pos_lf = np.hypot(lf[:, 0], lf[:, 1])
    psi_tot = np.abs(wrap(logs.eta[:, 5]))
    psi_lf = np.abs(wrap(lf[:, 2]))
    m = {
        "pos_tot_max": pos_tot[sel].max(), "pos_tot_rms": np.sqrt(np.mean(pos_tot[sel] ** 2)),
        "pos_lf_max": pos_lf[inner].max(), "pos_lf_rms": np.sqrt(np.mean(pos_lf[inner] ** 2)),
        "psi_tot_max": np.degrees(psi_tot[sel].max()),
        "psi_tot_rms": np.degrees(np.sqrt(np.mean(psi_tot[sel] ** 2))),
        "psi_lf_max": np.degrees(psi_lf[inner].max()),
        "psi_lf_rms": np.degrees(np.sqrt(np.mean(psi_lf[inner] ** 2))),
    }
    # Estimation: estimate minus true low-frequency state, and minus the raw truth
    for name, idx, scale in (("N", 0, 1.0), ("E", 1, 1.0), ("psi", 5, 180 / np.pi)):
        true = np.unwrap(logs.eta[:, idx])
        est = np.unwrap(logs.eta_est[:, idx])
        e_lf = est - moving_avg(true, dt)
        e_tot = est - true
        m[f"est_{name}_lf_rms"] = scale * np.sqrt(np.mean(e_lf[inner] ** 2))
        m[f"est_{name}_lf_max"] = scale * np.abs(e_lf[inner]).max()
        m[f"est_{name}_tot_rms"] = scale * np.sqrt(np.mean(e_tot[sel] ** 2))
        meas = np.unwrap(logs.eta_measured[:, idx])
        m[f"meas_{name}_std"] = scale * np.std(meas[sel] - moving_avg(true, dt)[sel])
        m[f"estdev_{name}_std"] = scale * np.std(e_lf[inner])
    for name, idx, scale in (("u", 0, 1.0), ("v", 1, 1.0), ("r", 5, 180 / np.pi)):
        e = logs.nu_est[:, idx] - moving_avg(logs.nu[:, idx], dt)
        m[f"est_{name}_lf_rms"] = scale * np.sqrt(np.mean(e[inner] ** 2))
        m[f"est_{name}_lf_max"] = scale * np.abs(e[inner]).max()
    # Actuators
    u_max = np.array([th.u_max for th in thrusters])
    sat = np.abs(logs.u) >= 0.99 * u_max
    util = thrust_utilization(logs, thrusters)
    resid = logs.tau_d[:, [0, 1, 5]] - logs.tau_thr[:, [0, 1, 5]]
    for i, th in enumerate(thrusters):
        m[f"sat_{th.name}"] = 100 * np.mean(sat[sel, i])
        m[f"peak_{th.name}"] = np.abs(logs.u[sel, i]).max() * 1e-3
    m["util_mean"] = util[sel].mean()
    m["util_max"] = util[sel].max()
    for j, nm in enumerate(("Fx", "Fy", "Mz")):
        m[f"resid_{nm}_rms"] = np.sqrt(np.mean(resid[sel, j] ** 2)) * 1e-3
        m[f"resid_{nm}_max"] = np.abs(resid[sel, j]).max() * 1e-3
        m[f"tau_{nm}_peak"] = np.abs(logs.tau_d[sel, [0, 1, 5][j]]).max() * 1e-3
    da = np.abs(wrap(logs.alpha_cmd - logs.alpha))
    m["alpha_lag_max_deg"] = np.degrees(da[sel].max())
    m["alpha_lag_mean_deg"] = np.degrees(da[sel].mean())
    m["bias_N_mean"] = logs.bias_est[sel, 0].mean() * 1e-3
    m["bias_E_mean"] = logs.bias_est[sel, 1].mean() * 1e-3
    m["bias_psi_mean"] = logs.bias_est[sel, 5].mean() * 1e-3
    return m


def main():
    thrusters = default_thrusters_part_2()
    logs = {k: run_sea(*v) for k, v in SEAS.items()}
    t0, t1 = T_RUN - WINDOW, T_RUN
    results = {}
    for k, lg in logs.items():
        results[(k, "window")] = metrics(lg, thrusters, t0, t1)
        results[(k, "full")] = metrics(lg, thrusters, 0.0, T_RUN)
    keys = list(results[("heavy", "window")])
    print(f"observer={rc.selected_observer()} window=[{t0:.0f},{t1:.0f}] s")
    print(f"{'metric':24s}" + "".join(f"{k[0]+'/'+k[1]:>16s}" for k in results))
    for key in keys:
        print(f"{key:24s}" + "".join(f"{results[k][key]:16.3f}" for k in results))

    h = logs["heavy"]
    d = logs["design"]
    dt = rc.DT
    # Figure 1: position and heading
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for a, idx, lab, s in zip(ax, (0, 1, 5), ("North [m]", "East [m]", "Heading [deg]"),
                              (1, 1, 180 / np.pi)):
        a.plot(h.t, s * np.unwrap(h.eta[:, idx]), lw=0.6, label="true")
        a.plot(h.t, s * np.unwrap(h.eta_est[:, idx]), "--", lw=1.0, label="estimate")
        a.plot(h.t, s * moving_avg(np.unwrap(h.eta[:, idx]), dt), "k", lw=1.2, label="true, 30 s mean")
        a.axvspan(t0, t1, color="C2", alpha=0.12, label="evaluation window")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(ncol=4, fontsize=8)
    ax[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig("results/p2sim6_fig1.png", dpi=300, bbox_inches="tight")

    # Figure 2: observer errors, heavy vs design sea
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for row, (idx, lab, s) in enumerate(((0, "North error [m]", 1), (1, "East error [m]", 1),
                                         (5, "Heading error [deg]", 180 / np.pi))):
        for lg, name, c in ((d, "design sea", "C0"), (h, "heavy sea", "C3")):
            e = np.unwrap(lg.eta_est[:, idx]) - moving_avg(np.unwrap(lg.eta[:, idx]), dt)
            ax[row, 0].plot(lg.t, s * e, lw=0.7, color=c, label=name)
        ax[row, 0].set_ylabel(lab)
    for row, (idx, lab, s) in enumerate(((0, "u error [m/s]", 1), (1, "v error [m/s]", 1),
                                         (5, "r error [deg/s]", 180 / np.pi))):
        for lg, name, c in ((d, "design sea", "C0"), (h, "heavy sea", "C3")):
            e = lg.nu_est[:, idx] - moving_avg(lg.nu[:, idx], dt)
            ax[row, 1].plot(lg.t, s * e, lw=0.7, color=c, label=name)
        ax[row, 1].set_ylabel(lab)
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.axvspan(t0, t1, color="C2", alpha=0.08)
    ax[0, 0].legend(fontsize=8)
    ax[-1, 0].set_xlabel("Time [s]")
    ax[-1, 1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig("results/p2sim6_fig2.png", dpi=300, bbox_inches="tight")

    # Figure 3: actuators (thrust vs limit, utilisation, wrench residual)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, th in enumerate(thrusters):
        ax[0].plot(h.t, np.abs(h.u[:, i]) / th.u_max * 100, lw=0.7, label=th.name)
    ax[0].axhline(100, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("|thrust| / limit [%]")
    ax[0].legend(ncol=3, fontsize=8)
    ax[1].plot(d.t, thrust_utilization(d, thrusters), lw=0.7, label="design sea")
    ax[1].plot(h.t, thrust_utilization(h, thrusters), lw=0.7, color="C3", label="heavy sea")
    ax[1].set_ylabel("Utilisation [%]")
    ax[1].legend(fontsize=8)
    resid = (h.tau_d - h.tau_thr)[:, [0, 1, 5]] * 1e-3
    for j, nm in enumerate(("Fx [kN]", "Fy [kN]", "Mz [kNm]")):
        ax[2].plot(h.t, resid[:, j], lw=0.6, label=nm)
    ax[2].set_ylabel("Desired - applied wrench")
    ax[2].legend(ncol=3, fontsize=8)
    ax[2].set_xlabel("Time [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.axvspan(t0, t1, color="C2", alpha=0.1)
    fig.tight_layout()
    fig.savefig("results/p2sim6_fig3.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
