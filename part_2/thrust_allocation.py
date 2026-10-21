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
            # (u, alpha) and (-u, alpha + pi) give the same force. Alpha_now[i] and
            # wrap_angle_pi to choose the pair that needs the smaller rotation.
            
            if alpha_now is not None:
                alpha_now_i = wrap_angle_pi(alpha_now[i])
                delta_alpha = wrap_angle_pi(alpha_cmd[i] - alpha_now_i)
                if abs(delta_alpha) > np.pi / 2:
                    u_cmd[i] = -u_cmd[i]
                    alpha_cmd[i] = wrap_angle_pi(alpha_cmd[i] + np.pi)

            col += 2

    return u_cmd, alpha_cmd

# Part 2, JFB method: build_B, build_D and the singularity measure

def build_B(thrusters: List[ThrusterConfig], alpha: np.ndarray) -> np.ndarray:

    # 3 x n thrust configuration matrix T(alpha), one column per thruster.
    # Column i is the wrench [Fx, Fy, Mz] from thruster i per unit thrust.
    # Same moment arm as in build_Be: Mz = x * Fy - y * Fx.

    B = np.zeros((3, len(thrusters)))
    for i, thr in enumerate(thrusters):
        if thr.kind == "tunnel":
            a = thr.alpha0
        else:
            a = alpha[i]
        B[:, i] = [np.cos(a), np.sin(a), thr.x * np.sin(a) - thr.y * np.cos(a)]
    return B


def build_D(thrusters: List[ThrusterConfig], alpha: np.ndarray, u: np.ndarray) -> np.ndarray:

    # 3 x n_az matrix, d(T(alpha) u)/d alpha at the current state.
    # One column per azimuth: the change in wrench per radian of rotation.
    # Proportional to u, so an idle azimuth gives no change in wrench when it rotates.

    az = [i for i, thr in enumerate(thrusters) if thr.kind == "azimuth"]
    D = np.zeros((3, len(az)))
    for j, i in enumerate(az):
        thr = thrusters[i]
        a = alpha[i]
        # CHECK: derivative of [cos a, sin a, x sin a - y cos a] * u with respect to a
        D[:, j] = u[i] * np.array([-np.sin(a), np.cos(a), thr.x * np.cos(a) + thr.y * np.sin(a)])
    return D


def singularity_measure(thrusters: List[ThrusterConfig], alpha: np.ndarray) -> float:

    # det(B(alpha) B(alpha)^T) from JFB (2004), eq. (3).
    # Zero when B(alpha) loses rank, i.e. the thrusters cannot produce a wrench in every direction
    # (for this vessel: both azimuths at +-90 deg, then no thruster gives surge).
    # With 3 thrusters B is 3 x 3, so this equals det(B)^2.
    B = build_B(thrusters, alpha)
    return float(np.linalg.det(B @ B.T))


# -----------------------------------------------------------------------------
# Allocator
# -----------------------------------------------------------------------------

class ThrustAllocator:
    """Thrust allocator with two methods, chosen with ``method``.

    The checks build ThrustAllocator(thrusters), so the default method is the
    one the checks see.

    STEADY-STATE METHOD (method = "steady_state", the Part 1 allocator)
    -------------------------------------------------------------------
    Regime 1 (feasible demand): pseudo-inverse z = Be^+ tau.
    Regime 2 (infeasible demand): SLSQP on the augmented vector [z, lambda],
        max lambda
        s.t. Be z = lambda tau,  |z_i| <= u_max,i,  0 <= lambda <= 1.
    This delivers the largest possible fraction lambda of the requested
    wrench in the requested direction. Starts from the regime 1 solution
    scaled to the limit, and falls back to that scaled solution if SLSQP fails.
    z_to_commands picks the (u, alpha) or (-u, alpha + pi) with the smaller
    rotation from alpha_now.

    JFB METHOD (method = "jfb", Part 2 continuation of Part 1)
    -----------------------------------------------------------
    NEW: the quasi-dynamic allocation of Johansen, Fossen and Berge (2004),
    with the Part 1 rule for saturation (largest lambda, direction kept)
    instead of the slack s. Nonlinear program, paper eq. (3)-(7):

        min   -lambda + mu_u sum u_i^2 + mu_alpha sum (alpha_i - alpha0_i)^2
                      + rho / (eps + det(B(alpha) B(alpha)^T))
        s.t.  B(alpha) u = lambda tau
              |u_i| <= u_max,i
              |alpha_i - alpha0_i| <= rot_speed * dt
              0 <= lambda <= 1

    alpha0, u0 = the commands from the previous step (paper: "the previous
    sample optimal solution"). Linearised around (u0, alpha0) and solved for
    the change du, dalpha in each step, paper eq. (8)-(12):

        min   -lambda + mu_u sum (u0 + du)^2 + mu_alpha sum dalpha^2
                      + rho * dg/dalpha(alpha0) dalpha,     g = 1 / (eps + det(B B^T))
        s.t.  B(alpha0)(u0 + du) + D dalpha = lambda tau
              |u0 + du| <= u_max
              |dalpha| <= rot_speed * dt
              0 <= lambda <= 1

    D = d(B(alpha) u)/d alpha at (u0, alpha0). The problem is convex (a QP)
    and solved with SLSQP. It is always feasible: du = -u0, dalpha = 0,
    lambda = 0 satisfies every constraint.

    Differences from the paper:
      - lambda (Part 1 rule) instead of the slack s^T Q s, so the direction of
        tau is kept exactly when the demand is infeasible
      - power W_i(u_i) = k |u_i|^(3/2) replaced by mu_u u_i^2
      - no sector limits alpha_min, alpha_max (eq. 6), the azimuths turn freely
      - no thrust-rate bound (also not in the paper), the thrust ramp is left
        to the actuator model

    mu_alpha [1/rad^2] costs rotation per step, mu_u [1/kN^2] costs thrust,
    rho and eps weight the singularity term.
    """

    def __init__(self, thrusters: List[ThrusterConfig], method: str = "steady_state",
                 mu_alpha: float = 0.1, mu_u: float = 1e-6, rho: float = 3e-4, eps: float = 1.0):
        self.thrusters = thrusters
        self.Be = build_Be(thrusters)  #Constant 3 x n_z effectiveness matrix, built by build_Be()
        self.Be_pinv = np.linalg.pinv(self.Be)

        # Choice of method
        if method not in ("steady_state", "jfb"):  # CHANGED: "sequential" and "nonlinear" removed, "jfb" added
            raise ValueError(f"unknown method {method!r}, use 'steady_state' or 'jfb'")
        self.method = method

        # Weights for the JFB method
        # given in the constructor, with default values. This way we could span over possible values in the analysis script.

        self.mu_alpha = mu_alpha  # cost on the change in azimuth angle per step [1/rad^2]
        self.mu_u = mu_u  # small cost on thrust [1/kN^2]

        # Singularity weight (JFB eq. 3). rho = 3e-4 keeps 60 kN sway away from +-90 deg
        # (det 30 instead of 0) without losing capacity. rho >= 1e-3 blocks the azimuths
        # at +-88 deg, and 400 kN sway then gives 61 kN instead of 95.4 kN.
        self.rho = rho
        self.eps = eps

        # Linearisation point, the commands from the previous step (paper: previous optimal solution).
        # None until the first call, then u_now / alpha_now are used.
        self.u_prev = None
        self.alpha_prev = None


    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Switch for the method
        if self.method == "jfb":
            return self.allocate_jfb(t, dt, tau_d, u_now, alpha_now)
        return self.allocate_steady_state(t, dt, tau_d, u_now, alpha_now)


    # Part 2: allocate_jfb method
    def allocate_jfb(self, t, dt, tau_d, u_now, alpha_now):

        SCALE = 1000.0  # kN and kNm, same as in the steady-state SLSQP
        thrusters = self.thrusters
        n = len(thrusters)
        az = [i for i, thr in enumerate(thrusters) if thr.kind == "azimuth"]
        n_az = len(az)

        tau = np.asarray(tau_d, dtype=float)[[0, 1, 5]] / SCALE

        # Step 1: linearisation point (u0, alpha0) = previous commands.
        # First call: start from the actual actuator state.
       
        if self.u_prev is None:
            u0 = np.asarray(u_now, dtype=float) / SCALE
            alpha0 = np.asarray(alpha_now, dtype=float).copy()
        else:
            u0 = self.u_prev
            alpha0 = self.alpha_prev

        # Step 2: linearised wrench around (u0, alpha0), paper eq. (9).
        # B(alpha0 + dalpha)(u0 + du) ~ B(alpha0)(u0 + du) + D dalpha
        B = build_B(thrusters, alpha0)
        D = build_D(thrusters, alpha0, u0)

        # Step 3: linearised singularity term, paper eq. (8): rho * dg/dalpha(alpha0) dalpha.
        # The gradient is found numerically (central differences), as in the paper.
        def g(alpha):
            return 1.0 / (self.eps + singularity_measure(thrusters, alpha))

        h = 1e-4  # [rad]
        grad_g = np.zeros(n_az)
        for j, i in enumerate(az):
            a_plus, a_minus = alpha0.copy(), alpha0.copy()
            a_plus[i] += h
            a_minus[i] -= h
            grad_g[j] = (g(a_plus) - g(a_minus)) / (2 * h)
        c_sing = self.rho * grad_g  # linear cost on dalpha

        # Step 4: the optimisation variable x = [du (n), dalpha (n_az), lambda].
        def objective(x):
            du, dalpha, lam = x[:n], x[n:n + n_az], x[-1]
            return (-lam + self.mu_u * np.sum((u0 + du) ** 2) + self.mu_alpha * np.sum(dalpha ** 2)
                    + c_sing @ dalpha)

        def objective_grad(x):
            du, dalpha = x[:n], x[n:n + n_az]
            return np.concatenate([2 * self.mu_u * (u0 + du), 2 * self.mu_alpha * dalpha + c_sing, [-1.0]])

        def equality(x):
            du, dalpha, lam = x[:n], x[n:n + n_az], x[-1]
            return B @ (u0 + du) + D @ dalpha - lam * tau  # same direction as tau, exactly

        def equality_jac(x):
            return np.hstack([B, D, -tau[:, None]])

        # Step 5: bounds, paper eq. (10) and (12).
        u_max = np.array([thr.u_max for thr in thrusters]) / SCALE
        dalpha_max = np.array([thrusters[i].rot_speed * dt for i in az])

        bounds = [(-u_max[i] - u0[i], u_max[i] - u0[i]) for i in range(n)]
        bounds += [(-dalpha_max[j], dalpha_max[j]) for j in range(n_az)]
        bounds += [(0.0, 1.0)]

        # Step 6: solve, starting from "no change" (x = 0).
        x0 = np.zeros(n + n_az + 1)
        res = minimize(objective, x0, jac=objective_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "eq", "fun": equality, "jac": equality_jac}],
                       options={"ftol": 1e-10, "maxiter": 200})
        x = res.x

        # Step 7: check the solution, fall back to the steady-state method if it is not usable.
        if not res.success or np.max(np.abs(equality(x))) > 1e-3:
            print(f"t={t:.1f}: jfb allocation failed ({res.message}), using steady-state")
            u_cmd, alpha_cmd = self.allocate_steady_state(t, dt, tau_d, u_now, alpha_now)
            self.u_prev = u_cmd / SCALE
            self.alpha_prev = np.asarray(alpha_cmd, dtype=float).copy()
            return u_cmd, alpha_cmd

        # Step 8: back to commands, and store them as the next linearisation point.
        # No minimal-rotation flip needed: u is signed and dalpha is at most one step.
        u_new = u0 + x[:n]
        alpha_new = alpha0.copy()
        for j, i in enumerate(az):
            alpha_new[i] = alpha0[i] + x[n + j]
        for i, thr in enumerate(thrusters):
            if thr.kind == "tunnel":
                alpha_new[i] = thr.alpha0

        self.u_prev = u_new
        self.alpha_prev = alpha_new  # kept unwrapped, only dalpha is used

        u_cmd = u_new * SCALE
        alpha_cmd = np.array([wrap_angle_pi(a) for a in alpha_new])  # wrap_angle_pi takes one angle
        return u_cmd, alpha_cmd


    def allocate_steady_state(self, t, dt, tau_d, u_now, alpha_now):

        tau = np.asarray(tau_d, dtype=float)[[0, 1, 5]]
        n_z = self.Be.shape[1]

        z = self.Be_pinv @ tau

        mags = magnitudes(z, self.thrusters)
        violates_limit = any(mag > thr.u_max for mag, thr in zip(mags, self.thrusters))

        if violates_limit:

            # Needed the z-vector to be augmented with a lambda variable, so we can maximize it while minimizing the objective function.

            SCALE = 1000.0  # Working in kN instead of N to avoid numerical issues with the optimization, recommended by Enio
            tau_s = tau / SCALE

            def objective(z_bar):
                lam = z_bar[-1]  # Retrieves the lambda variable, can not use 1/lambda because of possible divison by 0
                return -lam  # CHANGED: maximize lambda only (the mu * z^T P z tie-breaker is removed)

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
                z = z_candidate
            else:
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