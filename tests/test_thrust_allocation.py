"""
Unit tests for part_2/thrust_allocation.py

Each helper function is tested on its own with small hand-checkable cases,
then ThrustAllocator is tested end to end with ideal actuators.

Run all:              python -m pytest tests/test_thrust_allocation_part2.py -v
Run one test:         python -m pytest tests/test_thrust_allocation_part2.py -v -k rotation_penalty

Tests for functions that are not written yet (e.g. weighted_pinv_solve) are
skipped automatically until the function exists.
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
# -----------------------------------------------------------------------------

def test_penalty_is_zero_without_alpha_now(thrusters):
    f = needs("build_rotation_penalty")
    P = f(thrusters, None, 1.0)
    assert P.shape == (5, 5)
    assert np.allclose(P, 0.0)


def test_penalty_azimuth_at_0deg_penalises_only_Fy(thrusters):
    f = needs("build_rotation_penalty")
    P = f(thrusters, np.array([np.pi / 2, 0.0, 0.0]), 1.0)
    expected = np.zeros((5, 5))
    expected[2, 2] = 1.0   # Azimuth_1, Fy
    expected[4, 4] = 1.0   # Azimuth_2, Fy
    assert np.allclose(P, expected)


def test_penalty_azimuth_at_90deg_penalises_only_Fx(thrusters):
    f = needs("build_rotation_penalty")
    P = f(thrusters, np.array([np.pi / 2, np.pi / 2, np.pi / 2]), 1.0)
    expected = np.zeros((5, 5))
    expected[1, 1] = 1.0   # Azimuth_1, Fx
    expected[3, 3] = 1.0   # Azimuth_2, Fx
    assert np.allclose(P, expected)


def test_penalty_tunnel_row_and_column_are_zero(thrusters):
    f = needs("build_rotation_penalty")
    P = f(thrusters, np.array([0.3, 1.1, -2.0]), 5.0)
    assert np.allclose(P[0, :], 0.0)
    assert np.allclose(P[:, 0], 0.0)


def test_penalty_scales_with_q(thrusters):
    f = needs("build_rotation_penalty")
    a = np.array([0.0, 0.7, -0.4])
    assert np.allclose(f(thrusters, a, 3.0), 3.0 * f(thrusters, a, 1.0))


def test_penalty_is_symmetric_and_positive_semidefinite(thrusters):
    f = needs("build_rotation_penalty")
    P = f(thrusters, np.array([0.0, 0.7, -2.4]), 2.0)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) >= -1e-12)


def test_penalty_same_for_alpha_and_alpha_plus_pi(thrusters):
    # (u, alpha) and (-u, alpha + pi) are the same force, so the penalty must agree
    f = needs("build_rotation_penalty")
    a = np.array([np.pi / 2, 0.7, -0.4])
    a_flip = a.copy()
    a_flip[1:] += np.pi
    assert np.allclose(f(thrusters, a, 1.0), f(thrusters, a_flip, 1.0))


def test_penalty_measures_force_across_current_direction(thrusters):
    # A force along the current azimuth direction costs nothing,
    # a force across it costs q * |F|^2
    f = needs("build_rotation_penalty")
    alpha = 0.6
    P = f(thrusters, np.array([np.pi / 2, alpha, alpha]), 1.0)
    along = np.array([0.0, np.cos(alpha), np.sin(alpha), 0.0, 0.0])
    across = np.array([0.0, -np.sin(alpha), np.cos(alpha), 0.0, 0.0])
    assert np.isclose(along @ P @ along, 0.0)
    assert np.isclose(across @ P @ across, 1.0)


# -----------------------------------------------------------------------------
# weighted_pinv_solve
# -----------------------------------------------------------------------------

def test_weighted_pinv_with_identity_equals_pinv(Be):
    f = needs("weighted_pinv_solve")
    tau = np.array([10e3, -20e3, 150e3])
    z = f(Be, np.eye(5), tau)
    assert np.allclose(z, np.linalg.pinv(Be) @ tau)


def test_weighted_pinv_reproduces_tau_exactly(Be, thrusters):
    # The penalty must only choose among exact solutions, never change tau
    f = needs("weighted_pinv_solve")
    P = needs("build_rotation_penalty")(thrusters, np.array([0.0, 1.2, -0.5]), 10.0)
    H = np.eye(5) + P
    tau = np.array([10e3, -20e3, 150e3])
    z = f(Be, H, tau)
    assert np.allclose(Be @ z, tau)


def test_weighted_pinv_reduces_force_across_current_direction(Be, thrusters):
    f = needs("weighted_pinv_solve")
    build_P = needs("build_rotation_penalty")
    alpha_now = np.array([np.pi / 2, 0.0, 0.0])      # azimuths point forward
    tau = np.array([20e3, 20e3, 0.0])
    P1 = build_P(thrusters, alpha_now, 1.0)          # q = 1 as the measuring stick

    z0 = f(Be, np.eye(5), tau)
    z10 = f(Be, np.eye(5) + build_P(thrusters, alpha_now, 10.0), tau)
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


@pytest.mark.xfail(reason="known issue: SLSQP stays near its start guess for very large demands")
def test_allocator_infeasible_400kN_reaches_full_capacity(thrusters):
    tau_d = np.array([0.0, 400e3, 0.0, 0.0, 0.0, 0.0])
    u, a = ta.ThrustAllocator(thrusters).allocate(
        0.0, 0.1, tau_d, np.zeros(3), np.array([np.pi / 2, 0.0, 0.0]))
    assert wrench_from_commands(thrusters, u, a)[1] > 90e3      # optimum is about 95.4 kN