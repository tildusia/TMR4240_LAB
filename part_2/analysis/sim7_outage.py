"""Simulation 7: station keeping through a loss of the position reference.

Hypothesis: the nonlinear passive observer's model (vessel dynamics + estimated
bias) lets the DP system keep station through a GNSS/gyro outage by dead
reckoning, so the excursion stays inside the 5 m DP watch circle for outages up
to some tens of seconds, and then grows super-linearly because the slowly
varying wind and gusts are not modelled.  Naively freezing the last measurement
is much worse, because the observer then "sees" a vessel that is perfectly on
target.

    python -m part_2.analysis.sim7_outage
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_case_part_2 as rc
from part_2.config import Part2SimConfig
from part_2.analysis.sim6_robustness import moving_avg, wrap
from part_2.observer import NonlinearPassiveObserver, ObserverEstimate
from simulation.capability import low_frequency_motion
from simulation.utils import Rz, to_3dof, to_6dof, wrap_angle_pi

T_START = 300.0                         # outage begins after the filter has settled
T_AFTER = 200.0                         # time kept after the outage ends
DURATIONS = [0.0, 30.0, 60.0, 120.0, 180.0, 300.0]
SEEDS = [123, 124, 125]
MODES = ("dead_reckoning", "frozen")
POS_LIMIT = rc.CAP_POS_LIMIT            # DP watch circle [m]
PSI_LIMIT = rc.CAP_PSI_LIMIT_DEG        # [deg]


class OutageNPO(NonlinearPassiveObserver):
    """Nonlinear passive observer whose measurement is lost in [t0, t1).

    dead_reckoning: the correction terms are switched off (model prediction only).
    frozen: the last valid measurement is repeated and still trusted.
    """

    def __init__(self, t0, t1, mode, **kw):
        super().__init__(**kw)
        self.t0, self.t1, self.mode = t0, t1, mode
        self._held = None

    def step(self, t, dt, eta_measured, tau_est):
        lost = self.t0 <= t < self.t1
        z = np.asarray(eta_measured, float)
        if not lost:
            self._held = z.copy()
        if lost and self.mode == "frozen":
            z = self._held
        if not (lost and self.mode == "dead_reckoning"):
            return super().step(t, dt, z, tau_est)

        h = float(dt)
        tau = to_3dof(tau_est)
        R = Rz(self.eta[2])
        xi_dot = self.xi @ self.Aw.T
        nu_dot = self.Minv @ (-self.D @ self.nu + R.T @ self.b + tau)
        self.xi = self.xi + h * xi_dot
        self.eta = self.eta + h * (R @ self.nu)
        self.b = self.b - h * self.b / self.T_b
        self.nu = self.nu + h * nu_dot
        self.eta[2] = wrap_angle_pi(self.eta[2])
        return ObserverEstimate(eta=to_6dof(self.eta), nu=to_6dof(self.nu), bias=to_6dof(self.b))


def run_case(duration, mode, seed):
    t1 = T_START + duration
    cfg = Part2SimConfig(dt=rc.DT, T=t1 + T_AFTER, use_observer=True,
                         observer_type=rc.selected_observer())
    sim = rc.make_sim(cfg)
    sim.observer = OutageNPO(T_START, t1, mode)
    sim.reset_state()
    return sim.run(np.zeros(6), **dict(zip(("current", "wind", "waves"),
                                           rc.environment(seed=seed))))


def analyse(logs, duration):
    dt = rc.DT
    t0, t1 = T_START, T_START + duration
    lf = low_frequency_motion(logs)
    win = (logs.t >= t0) & (logs.t <= logs.t[-1] - 15.0)
    pos_lf = np.hypot(lf[:, 0], lf[:, 1])
    psi_lf = np.degrees(np.abs(wrap(lf[:, 2])))
    true_n, true_e = (moving_avg(np.unwrap(logs.eta[:, i]), dt) for i in (0, 1))
    err = np.hypot(np.unwrap(logs.eta_est[:, 0]) - true_n,
                   np.unwrap(logs.eta_est[:, 1]) - true_e)
    psi_err = np.degrees(np.abs(wrap(np.unwrap(logs.eta_est[:, 5])
                                    - moving_avg(np.unwrap(logs.eta[:, 5]), dt))))
    k1 = min(int(round(t1 / dt)), len(logs.t) - 1)
    thr_sat = np.mean(np.abs(logs.u[win]) >= 0.99 * np.array([32e3, 80e3, 80e3]))
    return {
        "pos_max": pos_lf[win].max(), "psi_max": psi_lf[win].max(),
        "pos_at_end": pos_lf[k1], "err_at_end": err[k1], "psi_err_at_end": psi_err[k1],
        "err_max": err[win].max(), "psi_err_max": psi_err[win].max(),
        "sat_pct": 100 * thr_sat,
    }, pos_lf, err


def main():
    res = {m: {d: [] for d in DURATIONS} for m in MODES}
    series = {}
    for m in MODES:
        for d in DURATIONS:
            for s in SEEDS:
                lg = run_case(d, m, s)
                r, pos, err = analyse(lg, d)
                res[m][d].append(r)
                if d == 120.0 and s == SEEDS[0]:
                    series[m] = (lg.t, pos, err)

    keys = list(res[MODES[0]][0.0][0])
    print(f"Outage starts at {T_START:.0f} s; mean over seeds {SEEDS}; "
          f"limits: {POS_LIMIT} m / {PSI_LIMIT} deg")
    mean = {(m, d): {k: float(np.mean([r[k] for r in res[m][d]])) for k in keys}
            for m in MODES for d in DURATIONS}
    worst = {(m, d): max(r["pos_max"] for r in res[m][d]) for m in MODES for d in DURATIONS}
    for m in MODES:
        print(f"\n{m}")
        print(f"{'dur [s]':>8s}" + "".join(f"{k:>15s}" for k in keys) + f"{'worst pos_max':>15s}")
        for d in DURATIONS:
            print(f"{d:8.0f}" + "".join(f"{mean[(m, d)][k]:15.2f}" for k in keys)
                  + f"{worst[(m, d)]:15.2f}")

    col = {"dead_reckoning": "C0", "frozen": "C3"}
    lab = {"dead_reckoning": "NPO dead reckoning", "frozen": "frozen measurement"}

    # Figure 1: time series for the 120 s outage
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m in MODES:
        t, pos, err = series[m]
        ax[0].plot(t, pos, color=col[m], lw=1.0, label=lab[m])
        ax[1].plot(t, err, color=col[m], lw=1.0, label=lab[m])
    for a in ax:
        a.axvspan(T_START, T_START + 120.0, color="0.85", label="outage")
        a.grid(alpha=0.3)
    ax[0].axhline(POS_LIMIT, color="k", ls="--", lw=0.8, label="watch circle")
    ax[0].set_ylabel("True LF position error [m]")
    ax[1].set_ylabel("Estimate - true LF position [m]")
    ax[1].set_xlabel("Time [s]")
    ax[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/p2sim7_fig1.png", dpi=300, bbox_inches="tight")

    # Figure 2: sweep over outage duration (mean with min-max over seeds)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for m in MODES:
        for a, key, yl in zip(ax, ("pos_max", "err_at_end", "psi_max"),
                              ("Max true LF position error [m]",
                               "Estimate error at outage end [m]",
                               "Max true LF heading error [deg]")):
            vals = np.array([[r[key] for r in res[m][d]] for d in DURATIONS])
            a.plot(DURATIONS, vals.mean(1), "o-", color=col[m], label=lab[m])
            a.fill_between(DURATIONS, vals.min(1), vals.max(1), color=col[m], alpha=0.2)
            a.set_xlabel("Outage duration [s]")
            a.set_ylabel(yl)
            a.grid(alpha=0.3)
    ax[0].axhline(POS_LIMIT, color="k", ls="--", lw=0.8, label="watch circle")
    ax[2].axhline(PSI_LIMIT, color="k", ls="--", lw=0.8)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/p2sim7_fig2.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
