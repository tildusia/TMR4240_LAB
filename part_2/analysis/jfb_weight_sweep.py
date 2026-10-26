"""
Weight sweep for the JFB allocator: rotation penalty mu_alpha and thrust cost mu_u.

Open loop, same set-up as thrust_allocation_analysis.py (allocator -> ThrusterSet with
dynamics on, actual state fed back as u_now / alpha_now). The scenarios, the run() loop
and the rotation metric are imported from that file, so the numbers are directly
comparable with Table 1 in tables.txt. Nothing in thrust_allocation_analysis.py is changed.

For every (mu_alpha, mu_u) pair:
    - rotation [deg] and effort [kN^2 s] after the warm-up, every scenario (as Table 1)
    - RMS of the force and moment residual |r_tau| after the warm-up, every scenario
    - capacity with ideal actuators after 60 s: 400 kN sway, 200 + 200 kN
    - chattering at the capacity limit: commanded azimuth rotation in the last 30 s
      of a constant infeasible demand (90 s, dynamics on). A settled allocator gives ~0.

Selection rule (agreed 2026-10-26), among the pairs that pass all three checks
    1. full sway capacity:  400 kN sway delivers >= CAP_MIN kN
    2. no chattering:       rotation in the last 30 s <= CHATTER_MAX deg, both demands
    3. little hunting:      noisy_low rotation <= HUNT_MAX deg
choose the one with the smallest total effort over the scenarios in EFFORT_SCENARIOS.

Run from the repo root (TMR4240_LAB):
    python -m part_2.analysis.jfb_weight_sweep               # full sweep, several minutes
    python -m part_2.analysis.jfb_weight_sweep --plot-only   # redraw the figure from sweep.json
Output in figures/jfb_weight_sweep/: sweep.txt (table), sweep.json (raw results), sweep_grid.png.
"""
import argparse
import inspect
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from part_2.analysis.thrust_allocation_analysis import (
    DT, REPO_ROOT, SCENARIOS, run, cumulative_rotation, _azimuths, _constant,
)

OUT_DIR = REPO_ROOT / "figures" / "jfb_weight_sweep"

MU_ALPHA_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0]   # [1/rad^2]
MU_U_VALUES = [1e-6, 1e-5, 1e-4]                # [1/kN^2]
CURRENT = (0.1, 1e-6)                           # values in the code today, marked in the output

# Selection rule
CAP_MIN = 95.0        # [kN] delivered for 400 kN sway (optimum is 95.4 kN)
CHATTER_MAX = 5.0     # [deg] commanded rotation in the last 30 s of a constant demand
HUNT_MAX = 100.0      # [deg] rotation in noisy_low after the warm-up
EFFORT_SCENARIOS = ["steps", "rotating", "noisy", "noisy_low", "random"]  # saturation left out:
# there the effort is set by the capacity, not by how the allocator distributes the thrust

CAPACITY_CASES = {"400 kN sway": (0.0, 400e3, 0.0), "200 + 200 kN": (200e3, 200e3, 0.0)}
CAPACITY_T = 60.0     # [s], ideal actuators (as Table 2 in tables.txt)
CHATTER_T = 90.0      # [s], dynamics on
CHATTER_WINDOW = 30.0 # [s] at the end of the run


def check_kwargs():
    """Stop early with a clear message if the allocator uses other argument names."""
    params = inspect.signature(ThrustAllocator.__init__).parameters
    missing = [k for k in ("mu_alpha", "mu_u") if k not in params]
    if missing:
        raise SystemExit(f"ThrustAllocator has no argument(s) {missing}. "
                         f"Available: {list(params)}. Change the names in kwargs() below.")


def kwargs(mu_alpha, mu_u):
    return dict(method="jfb", mu_alpha=mu_alpha, mu_u=mu_u)


def residual_rms(log, keep):
    r = log["tau_c"][keep] - log["tau_thr"][keep]
    force = np.sqrt(np.mean(r[:, 0] ** 2 + r[:, 1] ** 2)) / 1e3   # [kN]
    moment = np.sqrt(np.mean(r[:, 2] ** 2)) / 1e3                 # [kNm]
    return force, moment


def commanded_rotation(log, thrusters, t_from):
    """sum_i |dalpha_cmd_i| [deg] from t_from to the end, wrapped."""
    az = _azimuths(thrusters)
    keep = log["t"] >= t_from
    d = np.diff(log["alpha_cmd"][keep][:, az], axis=0)
    d = np.arctan2(np.sin(d), np.cos(d))
    return np.degrees(np.sum(np.abs(d)))


def evaluate(thrusters, alloc_kwargs):
    res = {}
    for sc, (tau_fun, T, t_skip) in SCENARIOS.items():
        log = run(thrusters, alloc_kwargs, tau_fun, T)
        keep = log["t"] >= t_skip
        res[f"rot {sc}"] = cumulative_rotation({k: v[keep] for k, v in log.items()}, thrusters)[-1]
        res[f"effort {sc}"] = np.sum(log["u"][keep] ** 2) * DT / 1e6
        res[f"rmsF {sc}"], res[f"rmsM {sc}"] = residual_rms(log, keep)

    for case, tc in CAPACITY_CASES.items():
        log = run(thrusters, alloc_kwargs, _constant(*tc), CAPACITY_T, dynamics=False)
        F = log["tau_thr"][-1] / 1e3
        e = np.array(tc[:2]) / np.hypot(*tc[:2])
        res[f"cap {case}"] = F[:2] @ e                         # along the demand [kN]
        res[f"cross {case}"] = abs(F[:2] @ np.array([-e[1], e[0]]))  # across [kN]

        log = run(thrusters, alloc_kwargs, _constant(*tc), CHATTER_T, dynamics=True)
        res[f"chatter {case}"] = commanded_rotation(log, thrusters, CHATTER_T - CHATTER_WINDOW)

    res["effort total"] = sum(res[f"effort {sc}"] for sc in EFFORT_SCENARIOS)
    return res


def passes(res):
    return (res["cap 400 kN sway"] >= CAP_MIN
            and all(res[f"chatter {c}"] <= CHATTER_MAX for c in CAPACITY_CASES)
            and res["rot noisy_low"] <= HUNT_MAX)


def plot_grid(ref, results):
    """One panel per check: rows mu_u, columns mu_alpha. Framed cells pass all three checks,
    the star marks the selected pair. Values are printed in every cell."""
    def grid(fun):
        return np.array([[fun(results[(a, u)]) for a in MU_ALPHA_VALUES] for u in MU_U_VALUES])

    cap = grid(lambda r: r["cap 400 kN sway"])
    chat = grid(lambda r: max(r[f"chatter {c}"] for c in CAPACITY_CASES))
    rot = grid(lambda r: r["rot noisy_low"])
    eff = grid(lambda r: 100 * (r["effort total"] / ref["effort total"] - 1))
    ok = grid(lambda r: passes(r)).astype(bool)
    sel = np.unravel_index(np.argmin(np.where(ok, eff, np.inf)), eff.shape) if ok.any() else None

    panels = [
        (cap, "(a) Sway capacity, 400 kN demand [kN]", "Blues", None, "{:.1f}",
         lambda v: v >= CAP_MIN, rf"$\geq$ {CAP_MIN:g} kN, no chatter"),
        (rot, "(b) Rotation in noisy_low [deg]", "Oranges", LogNorm(10, 6000), "{:.0f}",
         lambda v: v <= HUNT_MAX, rf"$\leq$ {HUNT_MAX:g} deg"),
        (eff, "(c) Effort above steady-state [%]", "Greys", None, "{:+.0f}",
         None, "lowest among accepted"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for ax, (Z, title, cmap, norm, fmt, crit, rule) in zip(axes, panels):
        im = ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(MU_U_VALUES)):
            for j in range(len(MA_LABELS)):
                txt = fmt.format(Z[i, j])
                if ax is axes[0] and chat[i, j] > CHATTER_MAX:
                    txt += "\nchatter"
                # bold = meets the requirement of this panel (panel (c) has none of its own)
                passed = crit is not None and crit(Z[i, j])
                if ax is axes[0]:
                    passed = passed and chat[i, j] <= CHATTER_MAX
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if im.norm(Z[i, j]) < 0.6 else "white",
                        fontweight="bold" if passed else "normal")
                if ok[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.2, ec="black"))
        if sel is not None:
            ax.plot(sel[1] + 0.38, sel[0] - 0.32, marker="*", ms=13, color="gold", mec="black", mew=0.8)
        ax.set_xticks(range(len(MA_LABELS)), MA_LABELS)
        ax.set_yticks(range(len(MU_U_VALUES)), [f"{u:g}" for u in MU_U_VALUES])
        ax.set_xlabel(r"$\mu_\alpha$ [rad$^{-2}$]")
        ax.set_title(f"{title}\nrequirement: {rule}", fontsize=10)
    axes[0].set_ylabel(r"$\mu_u$ [kN$^{-2}$]")
    fig.suptitle("JFB weight sweep, open loop. Framed: passes all checks (capacity, no chatter, "
                 "rotation). Star: selected.", fontsize=10)
    fig.savefig(OUT_DIR / "sweep_grid.png", dpi=150)
    plt.close(fig)


MA_LABELS = [f"{a:g}" for a in MU_ALPHA_VALUES]


def load_results():
    raw = json.loads((OUT_DIR / "sweep.json").read_text())
    results = {(p["mu_alpha"], p["mu_u"]): p for p in raw["pairs"]}
    return raw["ref"], results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from sweep.json")
    args = parser.parse_args()
    if args.plot_only:
        ref, results = load_results()
        plot_grid(ref, results)
        print(f"saved {OUT_DIR / 'sweep_grid.png'}")
        return

    check_kwargs()
    thrusters = default_thrusters_part_2()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("steady-state (reference) ...")
    ref = evaluate(thrusters, dict(method="steady_state"))

    results = {}
    pairs = [(a, u) for a in MU_ALPHA_VALUES for u in MU_U_VALUES]
    for n, (a, u) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] mu_alpha = {a:g}, mu_u = {u:g} ...")
        results[(a, u)] = evaluate(thrusters, kwargs(a, u))

    # --- Table ---
    head = (f"{'mu_alpha':>8} {'mu_u':>7} | {'cap sway':>8} {'cap 200/200':>11} | "
            f"{'chat sway':>9} {'chat 200':>8} | {'rot nlow':>8} {'rot noisy':>9} {'rot rand':>8} "
            f"{'rot steps':>9} {'rot rotat':>9} | {'effort tot':>10} {'vs ref':>7} | "
            f"{'rmsF nlow':>9} {'rmsM sat':>8} | ok")
    lines = ["JFB weight sweep, open loop (see docstring for the set-up and the selection rule)",
             f"rotation [deg], effort [kN^2 s] over {EFFORT_SCENARIOS}, capacity [kN] ideal actuators, "
             f"chatter = commanded rotation in the last {CHATTER_WINDOW:g} s [deg], rms residual [kN] / [kNm]",
             "", head, "-" * len(head)]

    def row(label_a, label_u, r, ok):
        return (f"{label_a:>8} {label_u:>7} | {r['cap 400 kN sway']:8.1f} {r['cap 200 + 200 kN']:11.1f} | "
                f"{r['chatter 400 kN sway']:9.1f} {r['chatter 200 + 200 kN']:8.1f} | "
                f"{r['rot noisy_low']:8.0f} {r['rot noisy']:9.0f} {r['rot random']:8.0f} "
                f"{r['rot steps']:9.0f} {r['rot rotating']:9.0f} | {r['effort total']:10.0f} "
                f"{100 * (r['effort total'] / ref['effort total'] - 1):+6.0f}% | "
                f"{r['rmsF noisy_low']:9.2f} {r['rmsM saturation']:8.1f} | {ok}")

    lines.append(row("steady", "state", ref, "ref"))
    for (a, u), r in results.items():
        mark = " <- current" if (a, u) == CURRENT else ""
        lines.append(row(f"{a:g}", f"{u:g}", r, ("yes" if passes(r) else "no") + mark))

    ok = {k: r for k, r in results.items() if passes(r)}
    lines.append("")
    if ok:
        best = min(ok, key=lambda k: ok[k]["effort total"])
        lines.append(f"Selected by the rule: mu_alpha = {best[0]:g}, mu_u = {best[1]:g} "
                     f"(effort total {ok[best]['effort total']:.0f}, "
                     f"noisy_low rotation {ok[best]['rot noisy_low']:.0f} deg)")
    else:
        lines.append("No pair passes all three checks. Look at the table and relax the rule.")

    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "sweep.txt").write_text(text + "\n")

    # --- Raw results, so the figure can be redrawn without rerunning the sweep ---
    raw = {"ref": ref, "pairs": [{"mu_alpha": a, "mu_u": u, **r} for (a, u), r in results.items()]}
    (OUT_DIR / "sweep.json").write_text(json.dumps(raw, indent=1))

    plot_grid(ref, results)
    print(f"\nsaved sweep.txt, sweep.json and sweep_grid.png in {OUT_DIR}")


if __name__ == "__main__":
    main()