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


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the nonlinear passive observer.
        return self._placeholder(eta_measured)


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

