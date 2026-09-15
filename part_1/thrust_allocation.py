"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Time invariant thrust allocation for a set of thrusters."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        self.Be = self._build_Be()

    """Finding the Be matrix"""

    def _build_Be(self) -> np.ndarray:
        """Constant 3 x n_z effectiveness matrix."""
        n_z = sum(2 if th.kind == "azimuth" else 1 for th in self.thrusters)
        Be = np.zeros((3, n_z))
        col = 0
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                Be[:, col] = [np.cos(thr.alpha0), np.sin(thr.alpha0), thr.x *np.sin(thr.alpha0) - thr.y *np.cos(thr.alpha0)]
                col += 1
            elif thr.kind == "azimuth":
                Be[:, col] = [1.0, 0.0, -thr.y]
                Be[:, col + 1] = [0.0, 1.0, thr.x]
                col += 2
        return Be

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        #Only interested in surge, sway and yaw moment (3-DOF)
        tau = tau_d[[0, 1, 5]]
        z = np.linalg.pinv(self.Be) @ tau

        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)

        #Magnitude of each thruster command, and scale down if any exceed max thrust
        #as recommended in discussion forum
        
        col = 0
        magnitudes = []
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                magnitudes.append(abs(z[col]))
                col += 1
            elif thr.kind == "azimuth":
                magnitudes.append(np.hypot(z[col], z[col + 1]))
                col += 2

        r = max(mag / thr.u_max for mag, thr in zip(magnitudes, self.thrusters))
        if r > 1.0:
            z = z / r

        col = 0
        for i, thr in enumerate(self.thrusters):
            if thr.kind == "tunnel":
                u_cmd[i] = z[col]
                alpha_cmd[i] = thr.alpha0
                col += 1
            elif thr.kind == "azimuth":
                Fx, Fy = z[col], z[col + 1]
                u_cmd[i] = np.hypot(Fx, Fy)
                alpha_cmd[i] = np.atan2(Fy, Fx)
                col += 2



        return u_cmd, alpha_cmd
