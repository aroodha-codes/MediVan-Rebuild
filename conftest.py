"""
conftest.py — Shared pytest fixtures for MediVan test suite.
"""

import math
import os
import sys

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def map_loader(project_root):
    """Pre-loaded MapLoader with the default hospital map."""
    from modules.map_loader import MapLoader
    loader = MapLoader()
    map_path = os.path.join(project_root, "assets", "hospital_map.png")
    if not os.path.exists(map_path):
        from map_editor import generate_default_map
        generate_default_map(map_path)
    loader.load_map(map_path)
    return loader


@pytest.fixture
def encoder():
    """Fresh EncoderSim instance."""
    from modules.encoder_sim import EncoderSim
    return EncoderSim()


@pytest.fixture
def planner():
    """Fresh PathPlanner instance."""
    from modules.path_planner import PathPlanner
    return PathPlanner()


@pytest.fixture
def q_agent():
    """Fresh Q-Learning agent (no saved table)."""
    from modules.q_learning_agent import QLearningAgent
    agent = QLearningAgent()
    # Reset to fresh state regardless of saved table
    agent.q_table[:] = 0.0
    agent.epsilon = 1.0
    agent.episode_count = 0
    agent.total_decisions = 0
    agent.total_reward = 0.0
    return agent


@pytest.fixture
def slam():
    """Initialized SLAMEngine."""
    from modules.slam_engine import SLAMEngine
    engine = SLAMEngine()
    engine.initialize(125, 465, -math.pi / 2)
    return engine


@pytest.fixture
def dummy_frame():
    """Synthetic camera frame with corridor + walls."""
    import cv2
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (0, 250), (640, 480), (160, 155, 145), -1)
    cv2.line(frame, (100, 150), (100, 450), (40, 40, 40), 5)
    cv2.line(frame, (540, 150), (540, 450), (40, 40, 40), 5)
    return frame
