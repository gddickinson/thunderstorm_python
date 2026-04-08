"""
Tests for simulation and evaluation modules.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.simulation import SMLMSimulator
from thunderstorm_python.evaluation import PerformanceEvaluator, create_test_pattern


class TestSMLMSimulator:
    def test_create_simulator(self):
        sim = SMLMSimulator(image_size=(64, 64))
        assert sim.image_size == (64, 64)

    def test_generate_molecule_positions(self):
        sim = SMLMSimulator(image_size=(64, 64), pixel_size=100.0)
        positions = sim.generate_molecule_positions(n_molecules=50)
        assert positions.shape == (50, 2)
        # Positions should be within image bounds
        assert np.all(positions[:, 0] >= 0)
        assert np.all(positions[:, 1] >= 0)

    def test_render_frame(self):
        sim = SMLMSimulator(image_size=(64, 64), pixel_size=100.0)
        positions = sim.generate_molecule_positions(n_molecules=10)
        frame, gt = sim.render_frame(positions)
        assert frame.shape == (64, 64)
        assert 'x' in gt
        assert 'y' in gt

    def test_simulate_blinking(self):
        sim = SMLMSimulator()
        states = sim.simulate_blinking(n_frames=10, n_molecules=5)
        assert states.shape == (10, 5)
        assert states.dtype == bool

    def test_generate_movie(self):
        sim = SMLMSimulator(image_size=(32, 32), pixel_size=100.0)
        movie, gt = sim.generate_movie(n_frames=3, n_molecules=5)
        assert movie.shape == (3, 32, 32)
        assert len(gt) == 3

    def test_generate_positions_with_mask(self):
        sim = SMLMSimulator(image_size=(64, 64), pixel_size=100.0)
        mask = np.ones((64, 64))
        positions = sim.generate_molecule_positions(n_molecules=20, mask=mask)
        assert positions.shape == (20, 2)


class TestPerformanceEvaluator:
    def test_perfect_detection(self):
        ev = PerformanceEvaluator(tolerance=10.0)
        positions = {'x': np.array([100, 200, 300.0]),
                     'y': np.array([100, 200, 300.0])}
        metrics = ev.evaluate(positions, positions)
        assert metrics['recall'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_no_detections(self):
        ev = PerformanceEvaluator(tolerance=10.0)
        detected = {'x': np.array([]), 'y': np.array([])}
        gt = {'x': np.array([100.0]), 'y': np.array([100.0])}
        metrics = ev.evaluate(detected, gt)
        assert metrics['recall'] == 0.0
        assert metrics['n_false_negative'] == 1

    def test_false_positives(self):
        ev = PerformanceEvaluator(tolerance=10.0)
        detected = {'x': np.array([100, 500.0]),
                     'y': np.array([100, 500.0])}
        gt = {'x': np.array([100.0]), 'y': np.array([100.0])}
        metrics = ev.evaluate(detected, gt)
        assert metrics['n_true_positive'] == 1
        assert metrics['n_false_positive'] == 1


class TestCreateTestPattern:
    def test_siemens_star(self):
        mask = create_test_pattern('siemens_star', size=64)
        assert mask.shape == (64, 64)

    def test_grid(self):
        mask = create_test_pattern('grid', size=64)
        assert mask.shape == (64, 64)

    def test_circle(self):
        mask = create_test_pattern('circle', size=64)
        assert mask.shape == (64, 64)

    def test_random(self):
        mask = create_test_pattern('random', size=64)
        assert mask.shape == (64, 64)
        assert np.all(mask == 1.0)
