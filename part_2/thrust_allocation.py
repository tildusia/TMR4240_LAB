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

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi  # needed for the minimal-rotation logic

from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Helper functions used by ThrustAllocator
# -----------------------------------------------------------------------------

#Finding the Be matrix

def build_Be(thrusters: List[ThrusterConfig]) -> np.ndarray:

    #Constant 3 x n_z effectiveness matrix.

    n_z = sum(2 if th.kind == "azimuth" else 1 for th in thrusters)
    Be = np.zeros((3, n_z))
    col = 0
    for thr in thrusters:
        if thr.kind == "tunnel":
            Be[:, col] = [np.cos(thr.alpha0), np.sin(thr.alpha0), thr.x * np.sin(thr.alpha0) - thr.y * np.cos(thr.alpha0)]
            col += 1
        elif thr.kind == "azimuth":
            Be[:, col] = [1.0, 0.0, -thr.y]
            Be[:, col + 1] = [0.0, 1.0, thr.x]
            col += 2
       
    return Be

def build_rotation_penalty(thrusters, alpha_now, q) -> np.ndarray:

    n_z = sum(2 if th.kind == "azimuth" else 1 for th in thrusters)
    Rp = np.zeros((n_z, n_z))
    col = 0

    if alpha_now is None:
        return Rp
    for i, thr in enumerate(thrusters):
        alpha_now_i = alpha_now[i]
        if thr.kind == "tunnel":
            col += 1
        elif thr.kind == "azimuth":
            e = np.array([-np.sin(alpha_now_i), np.cos(alpha_now_i)])
            Rp[col:col + 2, col:col + 2] = q * np.outer(e, e)
            col += 2

    return Rp

def magnitudes(z: np.ndarray, thrusters: List[ThrusterConfig]) -> list: #Magnutude is needed multiple times, so it is extracted to a function to avoid code duplication
    mags = []
    col = 0
    for thr in thrusters:
        if thr.kind == "tunnel":
            mags.append(abs(z[col]))
            col += 1
        elif thr.kind == "azimuth":
            mags.append(np.hypot(z[col], z[col + 1]))
            col += 2
    return mags


def max_ratio(mags: list, thrusters: List[ThrusterConfig]) -> float:
    # Largest ratio of magnitude to u_max over all thrusters
    return max(mag / thr.u_max for mag, thr in zip(mags, thrusters))


def z_to_commands(
    z: np.ndarray,
    thrusters: List[ThrusterConfig],
    alpha_now: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Converts the z-vector back to thrust magnitude and angle for each thruster
    
    n = len(thrusters)
    u_cmd = np.zeros(n)
    alpha_cmd = np.zeros(n)

    rotation_tol = 1.0  # 1N, small tolerance for zero thrust, found through testing and simulation.

    col = 0
    for i, thr in enumerate(thrusters):
        if thr.kind == "tunnel":
            u_cmd[i] = z[col]
            alpha_cmd[i] = thr.alpha0
            col += 1
        elif thr.kind == "azimuth":
            Fx, Fy = z[col], z[col + 1]
            u_cmd[i] = np.hypot(Fx, Fy)
            alpha_cmd[i] = np.arctan2(Fy, Fx)

            # PART 2
            # Zero-thrust angle.
            # Since arctan2(0, 0) = 0, an azimuth with no thrust is commanded to turn towards 0 rad.
            # Instead, we want it to stay at its current angle (or the default angle if no current angle is known).

            if abs(u_cmd[i]) < rotation_tol: 
                if alpha_now is not None:
                    alpha_cmd[i] = alpha_now[i]
                else:
                    alpha_cmd[i] = thr.alpha0  

            # Minimal rotation.
            # (u, alpha) and (-u, alpha + pi) give the same force. Use alpha_now[i] and
            # wrap_angle_pi to choose the pair that needs the smaller rotation.
            
            if alpha_now is not None:
                alpha_now_i = wrap_angle_pi(alpha_now[i])
                delta_alpha = wrap_angle_pi(alpha_cmd[i] - alpha_now_i)
                if abs(delta_alpha) > np.pi / 2:
                    u_cmd[i] = -u_cmd[i]
                    alpha_cmd[i] = wrap_angle_pi(alpha_cmd[i] + np.pi)

            col += 2

    return u_cmd, alpha_cmd

def weighted_pinv_solve(Be, H, tau):
    """
    Weighted minimum-norm solution of Be z = tau.

    Solves  min_z  z^T H z   s.t.  Be z = tau,
    with H = I + P symmetric positive definite.

    Parameters
    ----------
    Be  : (3, n_z) extended thrust configuration matrix
    H   : (n_z, n_z) weighting matrix
    tau : (3,) desired generalized force [X, Y, N]

    Returns
    -------
    z : (n_z,) extended force vector [Fx1, Fy1, Fx2, Fy2, ...]
    """
    tau = np.asarray(tau, dtype=float)

    # TODO 1: Y = H^{-1} Be^T, without forming the inverse (np.linalg.solve)
    Y = np.linalg.solve(H, Be.T)
    # TODO 2: M = Be Y, a small 3x3 system
    M = Be @ Y
    # TODO 3: w = M^{-1} tau, again with np.linalg.solve
    w = np.linalg.solve(M, tau)
    # TODO 4: z = Y w
    z = Y @ w
    return z


# -----------------------------------------------------------------------------
# Allocator
# -----------------------------------------------------------------------------

class ThrustAllocator:
    """Two-regime thrust allocator with a rotation penalty.

    Regime 1 (feasible demand): weighted minimum-norm solution
        min z^T H z  s.t.  Be z = tau,   H = I + P(alpha_now),
    solved in closed form by weighted_pinv_solve. P penalises force across
    each azimuth's current direction, so the thrusters prefer to push along
    where they already point. Used whenever all thrusters stay within u_max.

    Regime 2 (infeasible demand): SLSQP on the augmented vector [z, lambda],
        min -lambda + mu z^T P z
        s.t. Be z = lambda tau,  |z_i| <= u_max,i,  0 <= lambda <= 1.
    This delivers the largest possible fraction lambda of the requested
    wrench in the requested direction. The small mu term only breaks ties
    between solutions with (almost) the same lambda, costing at most about
    1e-3 in lambda. Starts from the regime 1 solution scaled to the limit,
    and falls back to that scaled solution if SLSQP fails.

    q sets the strength of the rotation penalty (q = 0 gives the plain
    pseudo-inverse). mu is derived from q and the azimuth u_max values.
    """

    def __init__(self, thrusters: List[ThrusterConfig], q: float = 0.1):
        self.thrusters = thrusters
        self.Be = build_Be(thrusters)  #Constant 3 x n_z effectiveness matrix, built by build_Be()

        self.q = q
        if self.q == 0.0:
            self.mu = 0.0
        else:
            self.mu = 1e-3 / (self.q * sum((thr.u_max/1000)**2 for thr in self.thrusters if thr.kind == "azimuth"))  # weight for the rotation penalty, scaled to kN^2

        self.n_calls = 0  #counter
        self.n_triggered = 0  #counter
        self.n_optimized = 0  #counter
        self.n_fallback = 0  #counter
        self.last_used_optimizer = False  #counter


    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
        tau = np.asarray(tau_d, dtype=float)[[0, 1, 5]]
        n_z = self.Be.shape[1]
        Rp = build_rotation_penalty(self.thrusters, alpha_now, self.q)
        H = np.eye(n_z) + Rp

        z = weighted_pinv_solve(self.Be, H, tau)

        mags = magnitudes(z, self.thrusters)
        violates_limit = any(mag > thr.u_max for mag, thr in zip(mags, self.thrusters))

        self.n_calls += 1  #counter
        self.last_used_optimizer = violates_limit  #counter

        if violates_limit:

            self.n_triggered += 1  #counter


            # Needed the z-vector to be augmented with a lambda variable, so we can maximize it while minimizing the objective function.

            SCALE = 1000.0  # Working in kN instead of N to avoid numerical issues with the optimization, recommended by Enio
            tau_s = tau / SCALE

            def objective(z_bar):
                lam = z_bar[-1]  # Retrieves the lambda variable, can not use 1/lambda because of possible divison by 0
                return -lam + self.mu * z_bar[:-1].T @ Rp @ z_bar[:-1] # maximize lambda, small penalty on force across current azimuth direction

            def equality(z_bar):
                lam = z_bar[-1]
                return self.Be @ z_bar[:-1] - lam * tau_s

            def make_tunnel_constraint(col, u_max):
                def constraint(z_bar):
                    return u_max**2 - z_bar[col]**2
                return constraint

            def make_azimuth_constraint(col_x, col_y, u_max):
                def constraint(z_bar):
                    return u_max**2 - (z_bar[col_x]**2 + z_bar[col_y]**2)
                return constraint

            constraints = [{"type": "eq", "fun": equality}]
            col = 0
            for thr in self.thrusters:
                if thr.kind == "tunnel":
                    constraints.append({"type": "ineq", "fun": make_tunnel_constraint(col, thr.u_max / SCALE)})
                    col += 1
                elif thr.kind == "azimuth":
                    constraints.append({"type": "ineq", "fun": make_azimuth_constraint(col, col + 1, thr.u_max / SCALE)})
                    col += 2

            bounds = [(None, None)] * n_z + [(0, 1)]  # z-components are unbounded, lambda variable is bounded between 0 and 1

            # a good initial guess for the optimization is to scale down the z-vector by the maximum ratio of magnitude to u_max, and set lambda to 1/r
            r = max_ratio(mags, self.thrusters)
            lam0 = 1.0 / r
            z_bar0 = np.concatenate([(z / r) / SCALE, [lam0]])

            res = minimize(objective, z_bar0, method="SLSQP", bounds=bounds, constraints=constraints)
            z_bar = res.x

            #Check if the optimization was successful and if the solution satisfies the constraints. If not, fall back to the original z-vector scaled down by r.
            z_candidate = z_bar[:-1] * SCALE
            magnitudes_candidate = magnitudes(z_candidate, self.thrusters)

            residual = equality(z_bar)
            residual_ok = np.allclose(residual, 0, atol=1e-3)

            TOL = 1e-3  # kN-scale margin
            limits_ok = all(mag <= thr.u_max + TOL * 1000 for mag, thr in zip(magnitudes_candidate, self.thrusters))

            if residual_ok and limits_ok:
                self.n_optimized += 1  #counter
                z = z_candidate
            else:
                self.n_fallback += 1  #counter
                #Check if tolerance is needed
                if not limits_ok:
                    violations = [
                        (thr.name if hasattr(thr, "name") else i, mag - thr.u_max)
                        for i, (mag, thr) in enumerate(zip(magnitudes_candidate, self.thrusters))
                        if mag > thr.u_max
                    ]
                    print(f"t={t:.1f}: limit violation(s) = {[(idx, f'{v/1000:.4f} kN over') for idx, v in violations]}")
                if not residual_ok:
                    print(f"t={t:.1f}: equality residual = {np.max(np.abs(residual)):.6f} (kN scale)")
                z = z / r

        return z_to_commands(z, self.thrusters, alpha_now)