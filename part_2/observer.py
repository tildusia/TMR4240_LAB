"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from simulation.utils import Rz, to_3dof, to_6dof, wrap_angle_pi
from importlib.resources import files
import pickle

@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )

  
def gunnerus_control_plant() -> tuple[np.ndarray, np.ndarray]:
    """Control plant matrices M = M_RB + M_A and D = D_L (3 DOF), read from the
    same mcsimpy data file as the process plant (models/gunnerus_3dof.py)."""
    path = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    M = np.asarray(data["Mrb"], dtype=float) + np.asarray(data["Ma"], dtype=float)
    D = np.asarray(data["Dl"], dtype=float)
    return M, D

class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(
        self,
        w0: float = 2.0 * np.pi / 8.0,     # wave peak frequency, T_p = 8 s [rad/s]
        lam: float = 0.1,                  # relative damping of the wave model [-]
        zeta_n: float = 1.0,               # notch damping [-]
        wc_ratio: float = 1.5,             # cut-off frequency wc = wc_ratio * w0
        k4: tuple = (0.1, 0.1, 0.1),       # K4 = diag(M) diag(k4) [1/s^2].
        kappa: float = 0.05,               # K3/K4 [rad/s]
        T_b: float = 1000.0,               # bias time constant [s]
        M: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ):
        M_plant, D_plant = gunnerus_control_plant()
        M_plant = 0.5 * (M_plant + M_plant.T)    # symmetric part, M = M^T > 0
        self.M = np.asarray(M_plant if M is None else M, dtype=float)
        self.D = np.asarray(D_plant if D is None else D, dtype=float)
        self.Minv = np.linalg.inv(self.M)
        self.w0, self.lam, self.zeta_n = float(w0), float(lam), float(zeta_n)
        self.wc = float(wc_ratio) * self.w0
        self.T_b = float(T_b)
 
        # Wave model, one 2x2 block per DOF (states [xi_1, xi_2] per DOF)
        self.Aw = np.array([[0.0, 1.0], [-self.w0**2, -2.0 * self.lam * self.w0]])
 
        # Correction gains, identical structure for N, E and psi
        self.K1 = np.array([
            -2.0 * (self.zeta_n - self.lam) * self.wc / self.w0,
            2.0 * self.w0 * (self.zeta_n - self.lam),
        ])
        self.K2 = np.full(3, self.wc)
        # Diagonal K4, scaled with the diagonal of M so that k4 is the same
        # acceleration per metre of innovation in every DOF.
        self.K4 = np.diag(np.diag(self.M) * np.asarray(k4, dtype=float))
        self.K3 = float(kappa) * self.K4
 
        self.check_passivity_condition()
        self.reset(np.zeros(6))

    def check_passivity_condition(self) -> None:
        """Raise if the gains violate 1/T_b < K3/K4 < w0 < wc (per DOF)."""
        ratio = np.diag(self.K3) / np.diag(self.K4)
        ok = np.all(1.0 / self.T_b < ratio) and np.all(ratio < self.w0) and self.w0 < self.wc
        if not ok:
            raise ValueError(
                f"NPO gains violate 1/T_b < K3/K4 < w0 < wc: 1/T_b = {1/self.T_b:.4f}, "
                f"K3/K4 = {ratio}, w0 = {self.w0:.3f}, wc = {self.wc:.3f}")

    def reset(self, eta0: np.ndarray | None = None):
        eta0 = np.zeros(6) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6)
        self.xi = np.zeros((3, 2))                 # WF states per DOF (N, E, psi)
        self.eta = to_3dof(eta0)         # LF pose [N, E, psi]
        self.eta[2] = wrap_angle_pi(self.eta[2])
        self.nu = np.zeros(3)                      # BODY velocity [u, v, r]
        self.b = np.zeros(3)                       # NED bias [b_N, b_E, b_psi]


    def step(self, t, dt, eta_measured, tau_est):
        h = float(dt)
        y = to_3dof(eta_measured)
        tau = to_3dof(tau_est)
        R = Rz(y[2])
 
        # 1. Prediction with the model (forward Euler), tau = tau_c[k-1]
        xi_dot = self.xi @ self.Aw.T
        eta_dot = R @ self.nu
        b_dot = -self.b / self.T_b
        nu_dot = self.Minv @ (-self.D @ self.nu + R.T @ self.b + tau)
        self.xi = self.xi + h * xi_dot
        self.eta = self.eta + h * eta_dot
        self.b = self.b + h * b_dot
        self.nu = self.nu + h * nu_dot
 
        # 2. Innovation y~ = y - eta^ - C_w xi^, heading component wrapped
        y_tilde = y - self.eta - self.xi[:, 1]
        y_tilde[2] = wrap_angle_pi(y_tilde[2])
 
        # 3. Correction
        R = Rz(self.eta[2])
        self.xi = self.xi + h * np.outer(y_tilde, self.K1)
        self.eta = self.eta + h * self.K2 * y_tilde
        self.b = self.b + h * self.K3 @ y_tilde
        self.nu = self.nu + h * self.Minv @ (R.T @ (self.K4 @ y_tilde))
        self.eta[2] = wrap_angle_pi(self.eta[2])
 
        # 4. Output, padded to the 6-DOF layout
        return ObserverEstimate(eta=to_6dof(self.eta), nu=to_6dof(self.nu),
                        bias=to_6dof(self.b))


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the Kalman filter or extended Kalman filter.
        return self._placeholder(eta_measured)


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

