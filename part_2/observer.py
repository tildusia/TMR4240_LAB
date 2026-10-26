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

    # state x (15): [xi_w(6) = wave pos/vel (N,E,psi), eta(3), bias(3), nu(3)]
    # bias is a NED force/moment in kN (kNm) for good conditioning.
    _W, _E, _B, _V = slice(0, 6), slice(6, 9), slice(9, 12), slice(12, 15)
    _KN = 1e3

    def __init__(
        self,
        omega0: float = 2 * np.pi / 8.0,    # wave peak frequency [rad/s]
        lam: float = 0.1,                   # wave relative damping
        T_bias: float = 1000.0,             # bias time constant [s]
        q_wave=(0.05, 0.05, 0.002),         # wave process noise std (N, E, psi)
        q_nu=(0.005, 0.005, 0.0005),        # velocity process noise std
        q_bias=(0.3, 0.3, 20.0),            # bias process noise std [kN, kN, kNm]/sqrt(s)
        r_meas=(0.3, 0.3, np.deg2rad(0.3)),  # measurement noise std (N, E, psi)
        p0_bias=(100.0, 100.0, 1000.0),
        vessel_model=None,
    ):
        self.w0, self.lam, self.Tb = float(omega0), float(lam), float(T_bias)
        self.q_wave = np.asarray(q_wave, float)
        self.q_nu = np.asarray(q_nu, float)
        self.q_bias = np.asarray(q_bias, float)
        self.R = np.diag(np.asarray(r_meas, float) ** 2)
        self.p0_bias = np.asarray(p0_bias, float)

        if vessel_model is None:
            from models.gunnerus_3dof import Gunnerus3DOF
            vessel_model = Gunnerus3DOF()._v
        self.M = np.asarray(vessel_model._Mrb + vessel_model._Ma, float)
        self.Minv = np.linalg.inv(self.M)
        self.D = np.asarray(vessel_model._Dl, float)

        self._last_meas = np.zeros(6)
        self.reset()

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _drz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])

    def reset(self, eta0=None):
        self.x = np.zeros(15)
        self._last_meas = np.zeros(6)
        if eta0 is not None:
            eta0 = np.asarray(eta0, float).reshape(6)
            self.x[self._E] = [eta0[0], eta0[1], eta0[5]]
            self._last_meas = eta0.copy()
        self.P = np.diag(np.concatenate([
            np.full(6, 0.5), [1.0, 1.0, 0.01], self.p0_bias, np.full(3, 0.1),
        ]) ** 2)
        self._initialised = eta0 is not None

    def _f(self, x, tau3):
        psi = x[8]
        R = self._rz(psi)
        xw, nu, b = x[self._W], x[self._V], x[self._B] * self._KN
        dxw = np.concatenate([
            xw[3:], -self.w0**2 * xw[:3] - 2 * self.lam * self.w0 * xw[3:]
        ])
        deta = R @ nu
        db = -x[self._B] / self.Tb
        dnu = self.Minv @ (-self.D @ nu + R.T @ b + tau3)
        return np.concatenate([dxw, deta, db, dnu])

    def _jac(self, x, tau3):
        psi = x[8]
        R, dR = self._rz(psi), self._drz(psi)
        nu, b = x[self._V], x[self._B] * self._KN
        A = np.zeros((15, 15))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = -self.w0**2 * np.eye(3)
        A[3:6, 3:6] = -2 * self.lam * self.w0 * np.eye(3)
        A[6:9, 8] = dR @ nu
        A[6:9, 12:15] = R
        A[9:12, 9:12] = -np.eye(3) / self.Tb
        A[12:15, 8] = self.Minv @ (dR.T @ b)
        A[12:15, 9:12] = self.Minv @ R.T * self._KN
        A[12:15, 12:15] = -self.Minv @ self.D
        return A

    def step(self, t, dt, eta_measured, tau_est):
        z_full = np.asarray(eta_measured, float).reshape(6)
        z = np.array([z_full[0], z_full[1], z_full[5]])
        tau6 = np.asarray(tau_est, float).reshape(6)
        tau3 = tau6[[0, 1, 5]]

        if not self._initialised:
            self.x[self._E] = z
            self._initialised = True

        # prediction (Euler)
        A = self._jac(self.x, tau3)
        self.x = self.x + dt * self._f(self.x, tau3)
        self.x[8] = self._wrap(self.x[8])
        F = np.eye(15) + dt * A
        Qc = np.zeros((15, 15))
        Qc[3:6, 3:6] = np.diag(self.q_wave**2)
        Qc[9:12, 9:12] = np.diag(self.q_bias**2)
        Qc[12:15, 12:15] = np.diag(self.q_nu**2)
        self.P = F @ self.P @ F.T + Qc * dt

        # correction: y = eta + wave position
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        H[:, 6:9] = np.eye(3)
        innov = z - (self.x[0:3] + self.x[6:9])
        innov[2] = self._wrap(innov[2])
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innov
        self.x[8] = self._wrap(self.x[8])
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

        eta = z_full.copy()
        eta[0], eta[1], eta[5] = self.x[6], self.x[7], self.x[8]
        nu = np.zeros(6)
        nu[[0, 1, 5]] = self.x[self._V]
        bias = np.zeros(6)
        bias[[0, 1, 5]] = self.x[self._B] * self._KN
        return ObserverEstimate(eta=eta, nu=nu, bias=bias)


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

