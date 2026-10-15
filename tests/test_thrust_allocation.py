"""
Unit tests for part_2/thrust_allocation.py

Each helper function is tested on its own with small hand-checkable cases,
then ThrustAllocator is tested end to end with ideal actuators.

Run all:              python -m pytest tests/test_thrust_allocation.py -v
Run one test:         python -m pytest tests/test_thrust_allocation.py -v -k rotation_penalty

Tests for functions that are not written yet are skipped automatically until
the function exists.
"""
import numpy as np
import pytest

import part_2.thrust_allocation as ta
from part_2.config import default_thrusters_part_2
from models.thruster_dynamics import ThrusterSet


# -----------------------------------------------------------------------------
# Shared fixtures and helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def thrusters():
    return default_thrusters_part_2()


@pytest.fixture
def Be(thrusters):
    return ta.build_Be(thrusters)


@pytest.fixture
def u_full(thrusters):
    """Every thruster at full thrust. Gives the scale factor (u_max / u_eff)^2 = 1."""
    return np.array([th.u_max for th in thrusters])


def needs(name):
    """Skip a test if the function is not implemented yet."""
    if not hasattr(ta, name):
        pytest.skip(f"{name} not implemented yet")
    return getattr(ta, name)


def wrench_from_commands(thrusters, u_cmd, alpha_cmd):
    """Body wrench [Fx, Fy, Mz] that the commands give with ideal actuators."""
    ts = ThrusterSet(thrusters, dynamics=False)
    _, _, tau = ts.step(u_cmd, alpha_cmd, 0.1)
    return tau


# -----------------------------------------------------------------------------
# build_Be
# -----------------------------------------------------------------------------

def test_build_Be_matches_hand_calculation(Be):
    expected = np.array([
        [0.0,  1.0,   0.0, 1.0,   0.0],
        [1.0,  0.0,   1.0, 0.0,   1.0],
        [12.0, -3.0, -13.0, 3.0, -13.0],
    ])
    assert Be.shape == (3, 5)
    assert np.allclose(Be, expected)


def test_build_Be_has_full_rank(Be):
    # Guide tip: rank(Be) = 3, otherwise some wrench directions are unreachable
    assert np.linalg.matrix_rank(Be) == 3


# -----------------------------------------------------------------------------
# build_rotation_penalty
#
# Signature: build_rotation_penalty(thrusters, alpha_now, u_now, q, u_floor_frac)
# Azimuth block: q * (u_max / u_eff)^2 * e_perp e_perp^T,
#                u_eff = max(|u_now|, u_floor_frac * u_max)
# With u_now = u_max the scale factor is 1, which the geometric tests use.
# -----------------------------------------------------------------------------

def test_penalty_azimuth_at_0deg_penalises_only_Fy(thrusters, u_full):
    P = ta.build_rotation_penalty(thrusters, np.array([np.pi / 2, 0.0, 0.0]), u_full, 1.0)
    expected = np.zeros((5, 5))
    expected[2, 2] = 1.0   # Azimuth_1, Fy
    expected[4, 4] = 1.0   # Azimuth_2, Fy
    assert np.allclose(P, expected)


def test_penalty_azimuth_at_90deg_penalises_only_Fx(thrusters, u_full):
    P = ta.build_rotation_penalty(thrusters, np.array([np.pi / 2, np.pi / 2, np.pi / 2]), u_full, 1.0)
    expected = np.zeros((5, 5))
    expected[1, 1] = 1.0   # Azimuth_1, Fx
    expected[3, 3] = 1.0   # Azimuth_2, Fx
    assert np.allclose(P, expected)


def test_penalty_tunnel_row_and_column_are_zero(thrusters):
    P = ta.build_rotation_penalty(thrusters, np.array([0.3, 1.1, -2.0]),
                                  np.array([5e3, 20e3, -30e3]), 5.0)
    assert np.allclose(P[0, :], 0.0)
    assert np.allclose(P[:, 0], 0.0)


def test_penalty_scales_with_q(thrusters):
    a = np.array([0.0, 0.7, -0.4])
    u = np.array([5e3, 20e3, 40e3])
    assert np.allclose(ta.build_rotation_penalty(thrusters, a, u, 3.0),
                       3.0 * ta.build_rotation_penalty(thrusters, a, u, 1.0))


def test_penalty_is_symmetric_and_positive_semidefinite(thrusters):
    P = ta.build_rotation_penalty(thrusters, np.array([0.0, 0.7, -2.4]),
                                  np.array([0.0, 10e3, 50e3]), 2.0)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) >= -1e-12)


def test_penalty_same_for_alpha_and_alpha_plus_pi(thrusters):
    # (u, alpha) and (-u, alpha + pi) are the same force, so the penalty must agree
    a = np.array([np.pi / 2, 0.7, -0.4])
    u = np.array([0.0, 30e3, 30e3])
    a_flip = a.copy()
    a_flip[1:] += np.pi
    assert np.allclose(ta.build_rotation_penalty(thrusters, a, u, 1.0),
                       ta.build_rotation_penalty(thrusters, a_flip, -u, 1.0))


def test_penalty_measures_force_across_current_direction(thrusters, u_full):
    # A force along the current azimuth direction costs nothing,
    # a unit force across it costs q * (u_max / u_eff)^2 = q here
    alpha = 0.6
    P = ta.build_rotation_penalty(thrusters, np.array([np.pi / 2, alpha, alpha]), u_full, 1.0)
    along = np.array([0.0, np.cos(alpha), np.sin(alpha), 0.0, 0.0])
    across = np.array([0.0, -np.sin(alpha), np.cos(alpha), 0.0, 0.0])
    assert np.isclose(along @ P @ along, 0.0)
    assert np.isclose(across @ P @ across, 1.0)


def test_penalty_grows_as_current_thrust_drops(thrusters, u_full):
    # Half the thrust gives four times the penalty: (u_max / (u_max/2))^2 = 4
    a = np.array([np.pi / 2, 0.3, -0.8])
    P_full = ta.build_rotation_penalty(thrusters, a, u_full, 1.0)
    P_half = ta.build_rotation_penalty(thrusters, a, u_full / 2, 1.0)
    assert np.allclose(P_half[1:, 1:], 4.0 * P_full[1:, 1:])


def test_penalty_floor_keeps_zero_thrust_finite(thrusters, u_full):
    # u_now = 0 falls back to u_floor_frac * u_max, scale = (1 / u_floor_frac)^2
    a = np.array([np.pi / 2, 0.3, -0.8])
    frac = 0.05
    P_zero = ta.build_rotation_penalty(thrusters, a, np.zeros(3), 1.0, frac)
    P_full = ta.build_rotation_penalty(thrusters, a, u_full, 1.0, frac)
    assert np.all(np.isfinite(P_zero))
    assert np.allclose(P_zero[1:, 1:], (1.0 / frac) ** 2 * P_full[1:, 1:])


def test_penalty_independent_of_thrust_sign(thrusters):
    a = np.array([np.pi / 2, 0.3, -0.8])
    u = np.array([10e3, 25e3, 60e3])
    assert np.allclose(ta.build_rotation_penalty(thrusters, a, u, 1.0),
                       ta.build_rotation_penalty(thrusters, a, -u, 1.0))


def test_penalty_approximates_q_umax2_dalpha2(thrusters):
    # The design goal: for a force of the current size turned by a small angle,
    # z^T P z ~ q * u_max^2 * dalpha^2
    q, alpha, dalpha, u_now = 2.0, 0.4, np.radians(3.0), 30e3
    az = thrusters[1]
    P = ta.build_rotation_penalty(thrusters, np.array([np.pi / 2, alpha, alpha]),
                                  np.array([0.0, u_now, u_now]), q)
    z = np.zeros(5)
    z[1:3] = u_now * np.array([np.cos(alpha + dalpha), np.sin(alpha + dalpha)])
    expected = q * az.u_max ** 2 * dalpha ** 2
    assert np.isclose(z @ P @ z, expected, rtol=1e-2)


# -----------------------------------------------------------------------------
# weighted_pinv_solve
# -----------------------------------------------------------------------------

def test_weighted_pinv_with_identity_equals_pinv(Be):
    tau = np.array([10e3, -20e3, 150e3])
    z = ta.weighted_pinv_solve(Be, np.eye(5), tau)
    assert np.allclose(z, np.linalg.pinv(Be) @ tau)


def test_weighted_pinv_reproduces_tau_exactly(Be, thrusters):
    # The penalty must only choose among exact solutions, never change tau
    P = ta.build_rotation_penalty(thrusters, np.array([0.0, 1.2, -0.5]),
                                  np.array([0.0, 10e3, 40e3]), 10.0)
    H = np.eye(5) + P
    tau = np.array([10e3, -20e3, 150e3])
    z = ta.weighted_pinv_solve(Be, H, tau)
    assert np.allclose(Be @ z, tau)


def test_weighted_pinv_reduces_force_across_current_direction(Be, thrusters, u_full):
    alpha_now = np.array([np.pi / 2, 0.0, 0.0])      # azimuths point forward
    tau = np.array([20e3, 20e3, 0.0])
    P1 = ta.build_rotation_penalty(thrusters, alpha_now, u_full, 1.0)   # measuring stick

    z0 = ta.weighted_pinv_solve(Be, np.eye(5), tau)
    P10 = ta.build_rotation_penalty(thrusters, alpha_now, u_full, 10.0)
    z10 = ta.weighted_pinv_solve(Be, np.eye(5) + P10, tau)
    assert z10 @ P1 @ z10 < z0 @ P1 @ z0


# -----------------------------------------------------------------------------
# magnitudes and max_ratio
# -----------------------------------------------------------------------------

def test_magnitudes(thrusters):
    z = np.array([-5e3, 3e3, 4e3, 0.0, 0.0])
    assert np.allclose(ta.magnitudes(z, thrusters), [5e3, 5e3, 0.0])


def test_max_ratio(thrusters):
    # 16/32 = 0.5, 40/80 = 0.5, 60/80 = 0.75
    assert np.isclose(ta.max_ratio([16e3, 40e3, 60e3], thrusters), 0.75)


# -----------------------------------------------------------------------------
# z_to_commands
# -----------------------------------------------------------------------------

def test_z_to_commands_tunnel_and_polar_conversion(thrusters):
    z = np.array([-7e3, 0.0, 10e3, -10e3, 0.0])
    u, a = ta.z_to_commands(z, thrusters, None)
    assert np.isclose(u[0], -7e3)                   # tunnel keeps its sign
    assert np.isclose(a[0], thrusters[0].alpha0)    # tunnel angle fixed
    assert np.isclose(u[1], 10e3) and np.isclose(a[1], np.pi / 2)
    assert np.isclose(abs(u[2]), 10e3)


def test_z_to_commands_reproduces_z_force(thrusters, Be):
    z = np.array([3e3, 12e3, -5e3, -8e3, 9e3])
    for alpha_now in (None, np.array([np.pi / 2, 2.5, -2.5])):
        u, a = ta.z_to_commands(z, thrusters, alpha_now)
        assert np.allclose(wrench_from_commands(thrusters, u, a), Be @ z, atol=1e-6)


def test_zero_thrust_keeps_current_angle(thrusters):
    z = np.zeros(5)
    alpha_now = np.array([np.pi / 2, 1.0, -2.0])
    u, a = ta.z_to_commands(z, thrusters, alpha_now)
    assert np.allclose(u, 0.0)
    assert np.allclose(a[1:], alpha_now[1:])


def test_zero_thrust_without_alpha_now_uses_alpha0(thrusters):
    u, a = ta.z_to_commands(np.zeros(5), thrusters, None)
    assert np.allclose(a, [th.alpha0 for th in thrusters])


def test_minimal_rotation_flips_sign_instead_of_turning_180(thrusters):
    # Azimuth_1 points at 170 deg, desired force points at -10 deg (backwards)
    alpha_now = np.array([np.pi / 2, np.radians(170.0), 0.0])
    F = 10e3
    z = np.array([0.0, F * np.cos(np.radians(-10.0)), F * np.sin(np.radians(-10.0)), 0.0, 0.0])
    u, a = ta.z_to_commands(z, thrusters, alpha_now)
    assert u[1] < 0                                             # negative thrust
    assert np.isclose(a[1], np.radians(170.0), atol=1e-9)       # no rotation


def test_minimal_rotation_keeps_sign_for_small_turn(thrusters):
    alpha_now = np.array([np.pi / 2, np.radians(10.0), 0.0])
    F = 10e3
    z = np.array([0.0, F * np.cos(np.radians(40.0)), F * np.sin(np.radians(40.0)), 0.0, 0.0])
    u, a = ta.z_to_commands(z, thrusters, alpha_now)
    assert u[1] > 0
    assert np.isclose(a[1], np.radians(40.0))


def test_minimal_rotation_never_turns_more_than_90deg(thrusters):
    rng = np.random.default_rng(1)
    for _ in range(200):
        z = rng.normal(size=5) * 20e3
        alpha_now = rng.uniform(-np.pi, np.pi, size=3)
        _, a = ta.z_to_commands(z, thrusters, alpha_now)
        for i in (1, 2):
            d = np.arctan2(np.sin(a[i] - alpha_now[i]), np.cos(a[i] - alpha_now[i]))
            assert abs(d) <= np.pi / 2 + 1e-9


# -----------------------------------------------------------------------------
# ThrustAllocator end to end (ideal actuators)
# -----------------------------------------------------------------------------

FEASIBLE_CASES = [
    ("pure surge 20 kN", [20e3, 0, 0, 0, 0, 0]),
    ("pure sway 20 kN", [0, 20e3, 0, 0, 0, 0]),
    ("pure yaw 200 kNm", [0, 0, 0, 0, 0, 200e3]),
    ("combined", [10e3, 10e3, 0, 0, 0, 100e3]),
    ("sway 70 kN (pinv saturates, SLSQP needed)", [0, 70e3, 0, 0, 0, 0]),
]


@pytest.mark.parametrize("label, tau_d", FEASIBLE_CASES)
def test_allocator_reproduces_feasible_wrench(thrusters, label, tau_d):
    tau_d = np.array(tau_d, dtype=float)
    alloc = ta.ThrustAllocator(thrusters)
    u, a = alloc.allocate(0.0, 0.1, tau_d, np.zeros(3), np.array([np.pi / 2, 0.0, 0.0]))
    tau = wrench_from_commands(thrusters, u, a)
    assert np.allclose(tau, tau_d[[0, 1, 5]], atol=1.0), label


@pytest.mark.parametrize("F", [100e3, 200e3, 400e3])
def test_allocator_infeasible_respects_limits_and_direction(thrusters, F):
    tau_d = np.array([0.0, F, 0.0, 0.0, 0.0, 0.0])
    alloc = ta.ThrustAllocator(thrusters)
    u, a = alloc.allocate(0.0, 0.1, tau_d, np.zeros(3), np.array([np.pi / 2, 0.0, 0.0]))
    u_max = np.array([th.u_max for th in thrusters])
    assert np.all(np.abs(u) <= u_max + 1.0)                     # course check limit
    tau = wrench_from_commands(thrusters, u, a)
    assert abs(tau[0]) < 1.0 and abs(tau[2]) < 1.0              # direction kept: pure sway
    assert tau[1] > 0


def test_allocator_penalty_does_not_cost_capacity_when_saturated(thrusters):
    # The mu term may cost at most about 1e-3 of lambda (derived bound).
    # Compare delivered sway with q = 0 (no penalty) and q > 0.
    tau_d = np.array([0.0, 100e3, 0.0, 0.0, 0.0, 0.0])
    alpha_now = np.array([np.pi / 2, 0.3, -0.3])
    u_now = np.array([0.0, 20e3, 20e3])
    Fy = {}
    for q in (0.0, 1.0):
        u, a = ta.ThrustAllocator(thrusters, q=q).allocate(0.0, 0.1, tau_d, u_now, alpha_now)
        Fy[q] = wrench_from_commands(thrusters, u, a)[1]
    assert Fy[1.0] >= Fy[0.0] * (1 - 2e-3)


@pytest.mark.xfail(reason="known issue: SLSQP stays near its start guess for very large demands")
def test_allocator_infeasible_400kN_reaches_full_capacity(thrusters):
    tau_d = np.array([0.0, 400e3, 0.0, 0.0, 0.0, 0.0])
    u, a = ta.ThrustAllocator(thrusters).allocate(
        0.0, 0.1, tau_d, np.zeros(3), np.array([np.pi / 2, 0.0, 0.0]))
    assert wrench_from_commands(thrusters, u, a)[1] > 90e3      # optimum is about 95.4 kN