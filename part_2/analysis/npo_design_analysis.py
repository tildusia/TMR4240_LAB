"""
NPO design analysis for the report: wave filtering and the passivity condition.

Linear analysis of the nonlinear passive observer in part_2/observer.py, with
the heading held constant (J(psi) = I), so no simulation is needed. All gains
are read from ``NonlinearPassiveObserver()`` with its default arguments, i.e.
exactly the observer the simulator and the checks use. Change the tuning in
observer.py and re-run this script to update the figure and the numbers.

Figure (figures/npo_design_analysis/)
    npo_design.png / .pdf : (a) wave filtering, |eta^_i / y_i|(jw), from measured
                            pose to LF pose estimate (notch at w0, low-pass at wc);
                            (b) phase of h_z,i(jw), the mapping that must be SPR
                            for the passivity argument (Sorensen 2018, Sec. 7.2.4).

Table (printed, and saved to tables.txt in the same folder)
    - the ordering condition 1/T_b << K3/K4 < w0 < wc, per DOF
    - the phase range of h_z,i(jw) per DOF (SPR requires it within +-90 deg)
    - the filter gain at w0 and at controller frequencies
    - the eigenvalues of the linearised error dynamics (full M, all 15 states)

Run from the repo root (TMR4240_LAB):
    python -m part_2.analysis.npo_design_analysis
    python -m part_2.analysis.npo_design_analysis --show
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import NonlinearPassiveObserver

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "npo_design_analysis"
DOF_NAMES = ("surge/North", "sway/East", "yaw")
W = np.logspace(-4, 1.5, 3000)          # frequency grid [rad/s]


def filter_gain(obs: NonlinearPassiveObserver, w: np.ndarray) -> np.ndarray:
    """|eta^_i / y_i|(jw): notch at w0 times low-pass at wc (velocity and bias
    parts neglected).  Same in every DOF."""
    s = 1j * w
    notch = (s**2 + 2 * obs.lam * obs.w0 * s + obs.w0**2) / (s**2 + 2 * obs.zeta_n * obs.w0 * s + obs.w0**2)
    return np.abs(notch * obs.wc / (s + obs.wc))


def h_z(obs: NonlinearPassiveObserver, i: int, w: np.ndarray) -> np.ndarray:
    """h_z,i(jw) = C_o (jwI - A_o)^-1 B_o for DOF i (subsystem H2 of the error
    dynamics, input nu~, output z~ = K4 y~ - b~, mooring G = 0)."""
    k1a, k1b = obs.K1
    k2, k3, k4 = obs.K2[i], obs.K3[i, i], obs.K4[i, i]
    w0, lam = obs.w0, obs.lam
    # states [xi1~, xi2~, eta~, b~],  y~ = eta~ + xi2~
    A = np.array([
        [0.0, 1.0 - k1a, -k1a, 0.0],
        [-w0**2, -2 * lam * w0 - k1b, -k1b, 0.0],
        [0.0, -k2, -k2, 0.0],
        [0.0, -k3, -k3, -1.0 / obs.T_b],
    ])
    B = np.array([0.0, 0.0, 1.0, 0.0])
    C = np.array([0.0, k4, k4, -1.0])
    return np.array([C @ np.linalg.solve(1j * wk * np.eye(4) - A, B) for wk in w])


def error_dynamics_eigenvalues(obs: NonlinearPassiveObserver) -> np.ndarray:
    """Eigenvalues of the full linearised error dynamics (J = I), 15 states
    [xi~ (6), eta~ (3), nu~ (3), b~ (3)], with the full M and D of the observer."""
    n = 15
    A = np.zeros((n, n))
    Minv = np.linalg.inv(obs.M)
    Cw = np.zeros((3, 6))
    for i in range(3):
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = obs.Aw
        Cw[i, 2 * i + 1] = 1.0
    K1 = np.zeros((6, 3))
    for i in range(3):
        K1[2 * i:2 * i + 2, i] = obs.K1
    H = np.hstack([Cw, np.eye(3), np.zeros((3, 6))])      # y~ = Cw xi~ + eta~
    A[6:9, 9:12] = np.eye(3)                               # eta~_dot = nu~ ...
    A[9:12, 9:12] = -Minv @ obs.D                          # M nu~_dot = -D nu~ + b~ ...
    A[9:12, 12:15] = Minv
    A[12:15, 12:15] = -np.eye(3) / obs.T_b                 # b~_dot = -b~/T_b ...
    K = np.vstack([K1, np.diag(obs.K2), Minv @ obs.K4, obs.K3])
    return np.linalg.eigvals(A - K @ H)


def plot_design(obs: NonlinearPassiveObserver, phase_deg: np.ndarray):
    blue, ink, ink2, grid, muted = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)
    a1.semilogx(W, filter_gain(obs, W), color=blue, lw=2)
    a1.set_ylabel(r"$|\hat\eta_i/y_i|$ [-]")
    a1.set_ylim(0, 1.1)
    a1.set_title(r"(a) Wave filtering: measurement $\to$ LF estimate", loc="left", fontsize=9, color=ink)
    a2.semilogx(W, phase_deg, color=blue, lw=2)
    a2.set_ylabel(r"$\angle h_{z,i}(j\omega)$ [deg]")
    a2.set_xlabel(r"$\omega$ [rad/s]")
    a2.axhspan(-120, -90, color=grid)
    a2.axhline(-90, color=ink2, lw=1, ls="--")
    a2.set_ylim(-110, 30)
    a2.text(2e-4, -100, r"SPR requires phase $> -90^\circ$", color=ink2, fontsize=8, va="center")
    a2.set_title(r"(b) Passivity: phase of $h_{z,i}(s)$ (identical in all DOFs)", loc="left", fontsize=9, color=ink)
    for ax in (a1, a2):
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    kappa = obs.K3[0, 0] / obs.K4[0, 0]
    marks = ((1 / obs.T_b, r"$1/T_b$", 1.03), (kappa, r"$K_3/K_4$", 1.03),
             (obs.w0, r"$\omega_0$", 1.03), (obs.wc, r"$\omega_c$", 0.90))
    for x, label, y in marks:
        for ax in (a1, a2):
            ax.axvline(x, color=muted, lw=0.8, ls=":")
        a1.text(x * 1.08, y, label, color=ink2, fontsize=8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show the figure as well")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obs = NonlinearPassiveObserver()                 # the tuned defaults
    phases = [np.degrees(np.angle(h_z(obs, i, W))) for i in range(3)]

    fig = plot_design(obs, phases[0])
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"npo_design.{ext}"
        fig.savefig(path, dpi=200)
        print(f"saved {path}")

    lines = ["NPO parameters (from NonlinearPassiveObserver defaults)",
             f"  w0 = {obs.w0:.4f} rad/s, wc = {obs.wc:.4f} rad/s (wc/w0 = {obs.wc / obs.w0:.2f}), "
             f"lambda = {obs.lam}, zeta_n = {obs.zeta_n}, T_b = {obs.T_b:.0f} s",
             f"  K1,i = [{obs.K1[0]:.4f}, {obs.K1[1]:.4f}], K2 = {np.round(obs.K2, 4)}",
             "  K3 = diag(" + ", ".join(f"{k:.1f}" for k in np.diag(obs.K3)) + "), "
             "K4 = diag(" + ", ".join(f"{k:.1f}" for k in np.diag(obs.K4)) + ")",
             "", "Passivity: ordering 1/T_b << K3/K4 < w0 < wc, and phase of h_z,i within +-90 deg"]
    for i, name in enumerate(DOF_NAMES):
        ratio = obs.K3[i, i] / obs.K4[i, i]
        order_ok = 1 / obs.T_b < ratio < obs.w0 < obs.wc
        ph = phases[i]
        spr_ok = ph.min() > -90.0 and ph.max() < 90.0
        lines.append(f"  {name:12s}: {1 / obs.T_b:.4f} < {ratio:.4f} < {obs.w0:.4f} < {obs.wc:.4f}  "
                     f"[{'ok' if order_ok else 'VIOLATED'}]   phase in [{ph.min():.2f}, {ph.max():.2f}] deg  "
                     f"[{'SPR ok' if spr_ok else 'NOT SPR'}]")
    lines += ["", "Wave filtering |eta^/y| (same in all DOFs)"]
    for w in (0.05, 0.1, 0.2, obs.w0, obs.wc):
        lines.append(f"  w = {w:.3f} rad/s: {filter_gain(obs, np.array([w]))[0]:.3f}")
    ev = sorted(error_dynamics_eigenvalues(obs), key=lambda z: (-z.real, abs(z.imag)))
    lines += ["", "Eigenvalues of the linearised error dynamics (full M, J = I), slowest first"]
    lines += [f"  {z.real:+.4f} {z.imag:+.4f}j   (time constant {-1 / z.real:6.1f} s)" for z in ev]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "tables.txt").write_text(text + "\n")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()