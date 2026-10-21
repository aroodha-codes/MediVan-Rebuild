"""
test_ai.py — Pytest tests for AI modules (YOLOv8 detector + Q-Learning agent).

Tests cover:
  - AI detector initialization and mode selection
  - Q-table shape and training convergence
  - Reward function correctness
  - Heuristic classification labels
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from config import JunctionAction, ObstacleClass, ObstacleAction
from modules.ai_obstacle_detector import AIObstacleDetector
from modules.q_learning_agent import QLearningAgent, N_STATES, N_ACTIONS


# ── AI Obstacle Detector ─────────────────────────


class TestAIDetector:
    """Tests for the YOLOv8-Nano / heuristic obstacle detector."""

    def test_detector_initializes(self):
        det = AIObstacleDetector()
        assert det is not None

    def test_detector_mode_valid(self):
        """Detector should be in YOLO mode (if ONNX exists) or HEURISTIC."""
        det = AIObstacleDetector()
        assert det.mode in ("YOLO", "HEURISTIC")

    def test_detect_returns_list(self):
        det = AIObstacleDetector()
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)

    def test_skip_frame_caching(self, monkeypatch):
        """YOLO runs at configured stride and uses cached results in between."""
        det = AIObstacleDetector()
        if not det.model_loaded:
            pytest.skip("shipped ONNX model unavailable in this environment")
        frame = np.full((240, 320, 3), 180, dtype=np.uint8)
        calls = []
        monkeypatch.setattr(det, "_run_yolo", lambda image: (calls.append(1) or []))
        monkeypatch.setattr(det, "_convert_detections", lambda raw, image: [])
        from config import YOLO_SKIP_FRAMES
        outputs = [det.detect(frame) for _ in range(YOLO_SKIP_FRAMES * 2 + 1)]
        assert all(isinstance(output, list) for output in outputs)
        assert len(calls) == 3

    def test_heuristic_classification_labels(self):
        """Heuristic classifier should produce valid ObstacleClass values."""
        import cv2
        det = AIObstacleDetector()
        # Create frame with a tall obstacle (should classify as PERSON)
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        # Train background model with clean frames
        for _ in range(130):
            det.detect(frame)
        # Add obstacle
        cv2.rectangle(frame, (250, 100), (290, 350), (80, 120, 160), -1)
        results = det.detect(frame)
        for r in results:
            assert r.classification in list(ObstacleClass)
            assert r.action in list(ObstacleAction)
            assert 0.0 <= r.confidence <= 1.0

    def test_run_yolo_detections_parsing(self):
        """Verify _run_yolo decodes output tensors and returns detections list properly."""
        det = AIObstacleDetector()
        det.model_loaded = True
        det.input_size = 320
        # Shape (1, 84, 100) where rows < cols, transposing to (100, 84) like real YOLOv8
        output = np.zeros((1, 84, 100), dtype=np.float32)
        output[0, 0, 0] = 160.0
        output[0, 1, 0] = 160.0
        output[0, 2, 0] = 50.0
        output[0, 3, 0] = 50.0
        output[0, 4, 0] = 0.9  # class 0 confidence

        class MockNet:
            def setInput(self, blob):
                pass
            def forward(self):
                return output

        det.net = MockNet()

        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        results = det._run_yolo(frame)
        assert isinstance(results, list)
        assert len(results) == 1
        class_id, conf, bbox = results[0]
        assert class_id == 0
        assert conf >= 0.85
        assert len(bbox) == 4


# ── Q-Learning Agent ─────────────────────────────


class TestQLearning:
    """Tests for the tabular Q-Learning junction decision agent."""

    def test_q_table_shape(self, q_agent):
        assert q_agent.q_table.shape == (N_STATES, N_ACTIONS)
        assert q_agent.q_table.shape == (54, 4)

    def test_initial_q_table_zeros(self, q_agent):
        """Fresh agent should have all-zero Q-table."""
        assert np.all(q_agent.q_table == 0.0)

    def test_choose_action_returns_valid(self, q_agent):
        action = q_agent.choose_action(
            junction_dist_px=10.0, obstacle_nearby=False,
            speed_ms=0.2, battery_pct=80.0,
        )
        assert isinstance(action, JunctionAction)
        assert action in list(JunctionAction)

    def test_epsilon_decays_after_training(self, q_agent):
        """Epsilon should decrease after episodes."""
        initial_eps = q_agent.epsilon
        for _ in range(50):
            action = q_agent.choose_action(
                junction_dist_px=np.random.uniform(0, 30),
                obstacle_nearby=np.random.random() > 0.6,
                speed_ms=np.random.uniform(0, 0.4),
                battery_pct=np.random.uniform(20, 100),
            )
            reward = q_agent.compute_reward(
                action, collision=False, obstacle_present=False,
                time_spent_frames=10, safely_passed=True,
            )
            q_agent.learn(reward, 15.0, False, 0.2, 80.0, done=True)
        assert q_agent.epsilon < initial_eps

    def test_training_updates_q_table(self, q_agent):
        """After training, Q-table should have non-zero entries."""
        for _ in range(20):
            action = q_agent.choose_action(5.0, True, 0.2, 80.0)
            reward = q_agent.compute_reward(
                action, collision=False, obstacle_present=True,
                time_spent_frames=8, safely_passed=True,
            )
            q_agent.learn(reward, 15.0, False, 0.2, 80.0, done=True)
        assert np.count_nonzero(q_agent.q_table) > 0

    def test_collision_gives_negative_reward(self, q_agent):
        reward = q_agent.compute_reward(
            JunctionAction.PROCEED, collision=True,
            obstacle_present=True, time_spent_frames=5, safely_passed=False,
        )
        assert reward < 0

    def test_safe_passage_gives_positive_reward(self, q_agent):
        reward = q_agent.compute_reward(
            JunctionAction.PROCEED, collision=False,
            obstacle_present=False, time_spent_frames=5, safely_passed=True,
        )
        assert reward > 0

    def test_unnecessary_wait_penalized(self, q_agent):
        """Waiting when no obstacle should be penalized."""
        reward = q_agent.compute_reward(
            JunctionAction.WAIT, collision=False,
            obstacle_present=False, time_spent_frames=30, safely_passed=True,
        )
        # Should have penalty from unnecessary waiting
        reward_proceed = q_agent.compute_reward(
            JunctionAction.PROCEED, collision=False,
            obstacle_present=False, time_spent_frames=5, safely_passed=True,
        )
        assert reward < reward_proceed

    def test_state_discretization_bounds(self, q_agent):
        """All state indices should be within valid range."""
        test_cases = [
            (0, False, 0.0, 0.0),     # minimums
            (50, True, 0.5, 100.0),    # maximums
            (10, False, 0.2, 50.0),    # mid range
        ]
        for dist, obs, speed, bat in test_cases:
            state = q_agent._discretize(dist, obs, speed, bat)
            assert 0 <= state < N_STATES, f"State {state} out of range for input ({dist},{obs},{speed},{bat})"

    def test_episode_count_increments(self, q_agent):
        assert q_agent.episode_count == 0
        action = q_agent.choose_action(5.0, False, 0.2, 80.0)
        q_agent.learn(10.0, 15.0, False, 0.2, 80.0, done=True)
        assert q_agent.episode_count == 1

    def test_get_stats_returns_dict(self, q_agent):
        stats = q_agent.get_stats()
        assert isinstance(stats, dict)
        assert "episodes" in stats
        assert "epsilon" in stats
        assert "total_reward" in stats
