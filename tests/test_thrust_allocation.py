"""
Unit tests for part_2/thrust_allocation.py

Each helper function is tested on its own with small hand-checkable cases,
then both allocation methods are tested end to end with ideal actuators:

    method="jfb"           (default) Johansen, Fossen and Berge (2004),
                           tested over repeated calls, as the course check does
    method="steady_state"  Part 1 allocator, tested on a single call

Run all:              python -m pytest tests/test_thrust_allocation.py -v
Run one group:        python -m pytest tests/test_thrust_allocation.py -v -k jfb
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


def wrench_from_commands(thrusters, u_cmd, alpha_cmd):
    """Body wrench [Fx, Fy, Mz] that the commands give with ideal actuators."""
    ts = ThrusterSet(thrusters, dynamics=False)
    _, _, tau = ts.step(u_cmd, alpha_cmd, 0.1)
    return tau


# Start state used in the single-call tests: tunnel at its fixed angle, azimuths at 0 rad
ALPHA_START = np.array([np.pi / 2, 0.0, 0.0])


# -----------------------------------------------------------------------------
# Method choice
# -----------------------------------------------------------------------------

def test_default_method_is_jfb(thrusters):
    # The course check builds ThrustAllocator(thrusters), so this is the method it sees
    assert ta.ThrustAllocator(thrusters).method == "jfb"


def test_steady_state_can_be_chosen(thrusters):
    assert ta.ThrustAllocator(thrusters, method="steady_state").method == "steady_state"


def test_unknown_method_raises(thrusters):
    with pytest.raises(ValueError):
        ta.ThrustAllocator(thrusters, method="sequential")


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
# magnitudes and max_ratio
# -----------------------------------------------------------------------------

def test_magnitudes(thrusters):
    z = np.array([-5e3, 3e3, 4e3, 0.0, 0.0])
    assert np.allclose(ta.magnitudes(z, thrusters), [5e3, 5e3, 0.0])


def test_max_ratio(thrusters):
    # 16/32 = 0.5, 40/80 = 0.5, 60/80 = 0.75
    assert np.isclose(ta.max_ratio([16e3, 40e3, 60e3], thrusters), 0.75)


# -----------------------------------------------------------------------------
# z_to_commands (used by the steady-state method)
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
# build_B, build_D and singularity_measure (used by the JFB method)
# -----------------------------------------------------------------------------

def test_build_B_matches_build_Be_columns(thrusters, Be):
    # With all azimuths at 0 rad, the azimuth columns of T(alpha) equal the Fx columns of Be,
    # and the tunnel column is the same in both.
    B = ta.build_B(thrusters, ALPHA_START)
    assert np.allclose(B[:, 0], Be[:, 0])
    assert np.allclose(B[:, 1], Be[:, 1])
    assert np.allclose(B[:, 2], Be[:, 3])


def test_build_D_matches_finite_difference(thrusters):
    alpha = np.array([np.pi / 2, 0.4, -1.1])
    u = np.array([5.0, 30.0, 50.0])
    D = ta.build_D(thrusters, alpha, u)
    h = 1e-6
    for j, i in enumerate([1, 2]):
        a_plus = alpha.copy()
        a_plus[i] += h
        fd = (ta.build_B(thrusters, a_plus) @ u - ta.build_B(thrusters, alpha) @ u) / h
        assert np.allclose(D[:, j], fd, atol=1e-4)


@pytest.mark.parametrize("deg1, deg2, expected", [(0, 0, 36.0), (90, 90, 0.0), (90, -90, 0.0), (90, 0, 625.0)])
def test_singularity_measure_hand_values(thrusters, deg1, deg2, expected):
    # det(B B^T) = det(B)^2 for 3 thrusters. Both azimuths at +-90 deg: no thruster gives surge.
    alpha = np.array([np.pi / 2, np.radians(deg1), np.radians(deg2)])
    assert np.isclose(ta.singularity_measure(thrusters, alpha), expected, atol=1e-6)


# -----------------------------------------------------------------------------
# Shared test cases
# -----------------------------------------------------------------------------

FEASIBLE_CASES = [
    ("pure surge 20 kN", [20e3, 0, 0, 0, 0, 0]),
    ("pure sway 20 kN", [0, 20e3, 0, 0, 0, 0]),
    ("pure yaw 200 kNm", [0, 0, 0, 0, 0, 200e3]),
    ("combined", [10e3, 10e3, 0, 0, 0, 100e3]),
    ("sway 70 kN (pinv saturates, SLSQP needed)", [0, 70e3, 0, 0, 0, 0]),
]


# -----------------------------------------------------------------------------
# JFB method (default), repeated calls with the actuator state fed back
# -----------------------------------------------------------------------------

def run_jfb(thrusters, tau_d, T=30.0, dynamics=False, **kwargs):
    """Call the JFB allocator repeatedly and feed the actuator state back.
    Returns the last wrench, whether every command was within u_max, the largest change
    in commanded azimuth angle per step [rad] and the final actual angles."""
    alloc = ta.ThrustAllocator(thrusters, method="jfb", **kwargs)
    ts = ThrusterSet(thrusters, dynamics=dynamics)
    u_max = np.array([th.u_max for th in thrusters])
    within = True
    max_dalpha = 0.0
    a_prev = None
    for k in range(int(round(T / 0.1))):
        u, a = alloc.allocate(k * 0.1, 0.1, tau_d, ts.get_thrusts(), ts.get_angles())
        within = within and bool(np.all(np.abs(u) <= u_max + 1.0))
        if a_prev is not None:
            d = np.arctan2(np.sin(a[1:] - a_prev[1:]), np.cos(a[1:] - a_prev[1:]))
            max_dalpha = max(max_dalpha, float(np.max(np.abs(d))))
        a_prev = a
        _, _, tau = ts.step(u, a, 0.1)
    return tau, within, max_dalpha, ts.get_angles()


@pytest.mark.parametrize("label, tau_d", FEASIBLE_CASES)
def test_jfb_reproduces_feasible_wrench(thrusters, label, tau_d):
    # Same as the course check: repeated calls from the initial state, ideal actuators, 30 s
    tau_d = np.array(tau_d, dtype=float)
    tau, within, _, _ = run_jfb(thrusters, tau_d)
    assert within, label
    assert np.allclose(tau, tau_d[[0, 1, 5]], atol=500.0), label


def test_jfb_rotation_per_step_within_rate_limit(thrusters):
    # Paper eq. (12): |dalpha| <= rot_speed * dt in every step
    rot_step = max(th.rot_speed for th in thrusters) * 0.1
    _, _, max_dalpha, _ = run_jfb(thrusters, np.array([0.0, 60e3, 0, 0, 0, 0]), T=30.0)
    assert max_dalpha <= rot_step + 1e-9


@pytest.mark.parametrize("tau_d", [[0, 400e3, 0, 0, 0, 0], [200e3, 200e3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3000e3]])
def test_jfb_infeasible_keeps_direction_and_limits(thrusters, tau_d):
    tau_d = np.array(tau_d, dtype=float)
    tau, within, _, _ = run_jfb(thrusters, tau_d, T=60.0, dynamics=True)
    assert within
    d = tau_d[[0, 1, 5]] / np.linalg.norm(tau_d[[0, 1, 5]])
    along = tau @ d
    assert along > 0
    assert np.linalg.norm(tau - along * d) < 0.01 * along     # direction kept within 1 %


def test_jfb_400kN_reaches_full_capacity(thrusters):
    tau, _, _, _ = run_jfb(thrusters, np.array([0.0, 400e3, 0, 0, 0, 0]), T=60.0, dynamics=True)
    assert tau[1] > 90e3                                       # optimum is about 95.4 kN


def test_jfb_singularity_term_keeps_azimuths_away_from_90deg(thrusters):
    # Steady 60 kN sway: without the term the azimuths end at +-91 deg (det ~ 0),
    # with the default rho = 3e-4 at about +-84 deg
    tau_d = np.array([0.0, 60e3, 0, 0, 0, 0])
    _, _, _, a0 = run_jfb(thrusters, tau_d, T=60.0, rho=0.0)
    _, _, _, a1 = run_jfb(thrusters, tau_d, T=60.0)
    assert ta.singularity_measure(thrusters, a0) < 5.0
    assert ta.singularity_measure(thrusters, a1) > 10.0


def test_jfb_new_instance_has_no_memory(thrusters):
    # The linearisation point is stored in the allocator. A new instance starts from u_now / alpha_now.
    alloc = ta.ThrustAllocator(thrusters, method="jfb")
    assert alloc.u_prev is None and alloc.alpha_prev is None
    alloc.allocate(0.0, 0.1, np.array([20e3, 0, 0, 0, 0, 0]), np.zeros(3), ALPHA_START)
    assert alloc.u_prev is not None
    assert ta.ThrustAllocator(thrusters, method="jfb").u_prev is None


def test_jfb_first_call_without_actuator_state(thrusters):
    # Without u_now / alpha_now the first call starts from zero thrust and the default angles alpha0
    alloc = ta.ThrustAllocator(thrusters, method="jfb")
    u, a = alloc.allocate(0.0, 0.1, np.array([20e3, 0, 0, 0, 0, 0]))
    u_max = np.array([th.u_max for th in thrusters])
    assert np.all(np.abs(u) <= u_max + 1.0)
    rot_step = max(th.rot_speed for th in thrusters) * 0.1
    alpha0 = np.array([th.alpha0 for th in thrusters])
    assert np.all(np.abs(a[1:] - alpha0[1:]) <= rot_step + 1e-9)   # one step away from alpha0 at most


# -----------------------------------------------------------------------------
# Steady-state method (method="steady_state"), single call
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("label, tau_d", FEASIBLE_CASES)
def test_steady_state_reproduces_feasible_wrench(thrusters, label, tau_d):
    tau_d = np.array(tau_d, dtype=float)
    alloc = ta.ThrustAllocator(thrusters, method="steady_state")
    u, a = alloc.allocate(0.0, 0.1, tau_d, np.zeros(3), ALPHA_START)
    tau = wrench_from_commands(thrusters, u, a)
    assert np.allclose(tau, tau_d[[0, 1, 5]], atol=1.0), label


@pytest.mark.parametrize("F", [100e3, 200e3, 400e3])
def test_steady_state_infeasible_respects_limits_and_direction(thrusters, F):
    tau_d = np.array([0.0, F, 0.0, 0.0, 0.0, 0.0])
    alloc = ta.ThrustAllocator(thrusters, method="steady_state")
    u, a = alloc.allocate(0.0, 0.1, tau_d, np.zeros(3), ALPHA_START)
    u_max = np.array([th.u_max for th in thrusters])
    assert np.all(np.abs(u) <= u_max + 1.0)                     # course check limit
    tau = wrench_from_commands(thrusters, u, a)
    assert abs(tau[0]) < 1.0 and abs(tau[2]) < 1.0              # direction kept: pure sway
    assert tau[1] > 0


@pytest.mark.xfail(reason="known issue: steady-state SLSQP stays near its start guess for very large demands")
def test_steady_state_400kN_reaches_full_capacity(thrusters):
    tau_d = np.array([0.0, 400e3, 0.0, 0.0, 0.0, 0.0])
    u, a = ta.ThrustAllocator(thrusters, method="steady_state").allocate(
        0.0, 0.1, tau_d, np.zeros(3), ALPHA_START)
    assert wrench_from_commands(thrusters, u, a)[1] > 90e3      # optimum is about 95.4 kN