import numpy as np
from part_1.config import default_thrusters_gunnerus3
from part_1.thrust_allocation import ThrustAllocator


def test_build_Be_matches_hand_calculation():
    ta = ThrustAllocator(default_thrusters_gunnerus3())

    expected = np.array([
        [0,1,0,1,0],
        [1,0,1,0,1],
        [12.0,-3.0,-13.0,3.0,-13.0]
    ])

    assert ta.Be.shape == expected.shape
    assert np.allclose(ta.Be, expected)