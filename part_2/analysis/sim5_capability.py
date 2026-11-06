"""Simulation 5: thrust-utilisation capability sweep with the selected observer.

Runs the 36-direction sweep (seed 123), prints the per-direction table and the
summary used in the report, and saves results/p2sim5_fig1.png.

    python -m part_2.analysis.sim5_capability
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

import run_case_part_2 as rc
from part_2.config import default_thrusters_part_2
from simulation.capability import plot_capability


def main():
    t_start = time.time()
    res = rc.sim5()
    fig = plot_capability(res, total_limits=(rc.CAP_TOTAL_POS_LIMIT, rc.CAP_TOTAL_PSI_LIMIT_DEG))
    fig.savefig("results/p2sim5_fig1.png", dpi=300, bbox_inches="tight")

    d = res.directions_deg
    u = res.utilization
    pos_lf, psi_lf = res.max_pos_dev_lf, np.degrees(res.max_psi_dev_lf)
    pos_t, psi_t = res.max_pos_dev_total, np.degrees(res.max_psi_dev_total)
    ok_lf = (pos_lf <= rc.CAP_POS_LIMIT) & (psi_lf <= rc.CAP_PSI_LIMIT_DEG)
    ok_t = (pos_t <= rc.CAP_TOTAL_POS_LIMIT) & (psi_t <= rc.CAP_TOTAL_PSI_LIMIT_DEG)
    print(f"\nobserver={rc.selected_observer()} seed={rc.SEED}")
    print("from  util   LFpos  LFpsi  Tpos  Tpsi")
    for i in range(len(d)):
        print(f"{d[i]:4.0f} {u[i]:5.1f} {pos_lf[i]:6.2f} {psi_lf[i]:6.2f} {pos_t[i]:5.2f} {psi_t[i]:5.2f}")
    print(f"util min {u.min():.1f} at {d[u.argmin()]:.0f}, max {u.max():.1f} at {d[u.argmax()]:.0f}, "
          f"mean {u.mean():.1f}")
    print(f"within LF limits {ok_lf.sum()}/{len(d)}; within total limits {ok_t.sum()}/{len(d)}: "
          f"{d[ok_t].astype(int).tolist()}")
    print(f"max LF pos {pos_lf.max():.2f} at {d[pos_lf.argmax()]:.0f}; "
          f"max LF psi {psi_lf.max():.2f} at {d[psi_lf.argmax()]:.0f}")
    print(f"max total pos {pos_t.max():.2f} at {d[pos_t.argmax()]:.0f}; "
          f"max total psi {psi_t.max():.2f} at {d[psi_t.argmax()]:.0f}")
    print(f"runtime {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
