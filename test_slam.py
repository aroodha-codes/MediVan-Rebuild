"""
test_slam.py — Pytest tests for Visual SLAM engine.
"""
import math, os, sys
import cv2, numpy as np, pytest

sys.path.insert(0, os.path.dirname(__file__))
from config import MotorCommand, MotorDirection
from modules.slam_engine import SLAMEngine, _GW, _GH


class TestSLAMEngine:
    def test_grid_dimensions(self, slam):
        assert slam.grid.shape == (_GH, _GW)

    def test_particles_initialized(self, slam):
        assert slam._particles_initialized is True
        assert slam.particles.shape == (30, 4)

    def test_coverage_increases(self, slam, dummy_frame):
        initial = slam.coverage
        for i in range(50):
            slam.update(dummy_frame, 2.0, 0, 0.01, 125+i*2, 465, -math.pi/2+i*0.01)
        assert slam.coverage > initial

    def test_explore_command_valid(self, slam):
        def mock_free(x, y): return 50 < x < 750 and 50 < y < 550
        cmd = slam.get_explore_command(125, 465, -math.pi/2, mock_free)
        assert isinstance(cmd, MotorCommand)
        assert 0 <= cmd.pwm_a <= 255

    def test_display_shape(self, slam):
        d = slam.get_slam_display()
        assert len(d.shape) == 3 and d.shape[2] == 3

    def test_stats_dict(self, slam):
        s = slam.get_stats()
        assert "coverage" in s and "complete" in s

    def test_frame_count(self, slam, dummy_frame):
        slam.update(dummy_frame, 0, 0, 0, 125, 465, -math.pi/2)
        assert slam.frame_count == 1

    def test_update_returns_float(self, slam, dummy_frame):
        r = slam.update(dummy_frame, 0, 0, 0, 125, 465, -math.pi/2)
        assert isinstance(r, float) and 0.0 <= r <= 1.0
