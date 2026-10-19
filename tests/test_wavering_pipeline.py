"""
Project WAV-DET - Verification & Automated Integration Test Suite.
Verifies:
1. Trajectory kinematic feature extraction (drift, oscillations, jerk, variance)
2. Wavering & swerving detection (moderate weaving, severe acute swerve)
3. Road context spatial zoning (exempts off-road tractors, fields, and shoulders)
4. Bidirectional dual-carriageway corridor flow discovery (no false wrong-way alarms)
5. SQLite persistent storage and active learning feedback loop
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.calibration import HomographyCalibrator
from core.geometry import RoadGeometryManager, Lane
from core.road_context import RoadContextManager
from core.wavering_detector import WaveringDetector, WaveringAssessment
from core.detector_tracker import TrackedVehicle
from core.behavior_engine import BehaviorEngine
from backend.database import init_db, insert_event, list_events, update_event_feedback, get_event_stats


class TestWaveringPipeline(unittest.TestCase):

    def setUp(self):
        # 4-point metric ground calibration setup
        image_quad = [[50, 450], [600, 450], [400, 150], [200, 150]]
        ground_quad = [[0.0, 0.0], [15.0, 0.0], [15.0, 100.0], [0.0, 100.0]]
        self.calibrator = HomographyCalibrator(image_quad, ground_quad)
        
        self.lanes_cfg = [
            {"id": 1, "name": "Lane 1", "left_boundary_x": 0.0, "right_boundary_x": 5.0, "centerline_x": 2.5, "direction_y": 1.0, "speed_limit_kmh": 80.0},
            {"id": 2, "name": "Lane 2", "left_boundary_x": 5.0, "right_boundary_x": 10.0, "centerline_x": 7.5, "direction_y": 1.0, "speed_limit_kmh": 80.0},
            {"id": 3, "name": "Lane 3", "left_boundary_x": 10.0, "right_boundary_x": 15.0, "centerline_x": 12.5, "direction_y": 1.0, "speed_limit_kmh": 80.0}
        ]
        self.geometry_mgr = RoadGeometryManager(self.lanes_cfg, self.calibrator)
        self.road_context = RoadContextManager(self.calibrator, road_width_m=15.0, road_depth_m=100.0)
        self.wavering_detector = WaveringDetector(window_seconds=3.5, min_samples=8)
        self.behavior_engine = BehaviorEngine(self.geometry_mgr, {}, self.road_context)

    def test_01_normal_driving_not_flagged(self):
        """Stable, straight-line trajectory must NOT trigger wavering."""
        history = []
        for i in range(25):
            t = i * 0.1
            gx = 2.5 + np.random.normal(0, 0.02)  # Centered in Lane 1
            gy = 10.0 + i * 2.0  # Moving forward at ~72 km/h
            history.append((gx, gy, t))

        assessment = self.wavering_detector.analyze_trajectory(
            history_ground=history,
            class_name="Car",
            current_speed_kmh=72.0,
            zone="CARRIAGEWAY",
            is_agricultural=False
        )
        self.assertFalse(assessment.is_wavering, "Normal straight driving should not be flagged as wavering")
        self.assertEqual(assessment.severity, "NORMAL")

    def test_02_repeated_lane_weaving_detected(self):
        """Zig-zagging lane drift with >= 3 sign flips must trigger Wavering with MODERATE/SEVERE."""
        history = []
        for i in range(35):
            t = i * 0.1
            # 3 sinusoids across 3.5 seconds with 1.2m amplitude
            gx = 5.0 + 1.2 * np.sin(2.0 * np.pi * t / 1.1)
            gy = 15.0 + i * 1.5
            history.append((gx, gy, t))

        assessment = self.wavering_detector.analyze_trajectory(
            history_ground=history,
            class_name="Car",
            current_speed_kmh=54.0,
            zone="CARRIAGEWAY",
            is_agricultural=False
        )
        self.assertTrue(assessment.is_wavering, "Repeated lane weaving must be detected")
        self.assertIn(assessment.severity, ["MODERATE", "SEVERE"])
        self.assertGreaterEqual(assessment.oscillation_count, 3)
        self.assertIn("drift", assessment.reason.lower())

    def test_03_high_g_acute_swerve_detected(self):
        """Sudden high-G lateral swerve (peak lateral jerk > 3.2 m/s²) must trigger SEVERE."""
        history = []
        for i in range(25):
            t = i * 0.1
            # Sudden lateral displacement of 1.8m in 0.3 seconds
            if i < 10:
                gx = 2.5
            elif 10 <= i < 14:
                gx = 2.5 + (i - 10) * 0.6  # Acute swerve
            else:
                gx = 4.9
            gy = 20.0 + i * 2.2
            history.append((gx, gy, t))

        assessment = self.wavering_detector.analyze_trajectory(
            history_ground=history,
            class_name="Car",
            current_speed_kmh=75.0,
            zone="CARRIAGEWAY",
            is_agricultural=False
        )
        self.assertTrue(assessment.is_wavering, "Acute emergency/reckless swerve must be detected")
        self.assertEqual(assessment.severity, "SEVERE")
        self.assertIn("swerve", assessment.reason.lower())

    def test_04_offroad_agricultural_vehicle_suppressed(self):
        """Tractor in off-road field or shoulder must NEVER trigger wavering false alarm."""
        history = []
        for i in range(25):
            t = i * 0.1
            # Off-road field (X = -8.0m, outside road boundaries)
            gx = -8.0 + 0.5 * np.sin(t)
            gy = 5.0 + i * 0.4  # Plowing at 14 km/h
            history.append((gx, gy, t))

        # Check road context classification
        is_on_road, zone = self.road_context.is_on_drivable_road(-8.0, 10.0)
        self.assertFalse(is_on_road, "Field position should not be classified as drivable road")
        self.assertEqual(zone, "OFF_ROAD")

        assessment = self.wavering_detector.analyze_trajectory(
            history_ground=history,
            class_name="Tractor",
            current_speed_kmh=14.0,
            zone=zone,
            is_agricultural=True
        )
        self.assertFalse(assessment.is_wavering, "Tractor in field must be exempt from wavering alerts")
        self.assertEqual(assessment.severity, "NORMAL")
        self.assertIn("off-road", assessment.reason.lower())

    def test_05_bidirectional_corridor_flow_discovery(self):
        """Dual-carriageway traffic (oncoming vs receding) must not be flagged as false Wrong-Way."""
        # Register 10 vehicles moving towards camera on left carriageway (X=3.0, vy < 0)
        for tid in range(1, 10):
            for step in range(8):
                self.road_context.register_vehicle_velocity(tid, ground_x=3.0, ground_y=50.0, vy_m_s=-15.0)

        # Expected flow in left carriageway should automatically discover -1.0
        expected_dir = self.road_context.get_expected_flow_direction(ground_x=3.0)
        self.assertEqual(expected_dir, -1.0, "Discovered corridor flow for left carriageway should be towards camera (-1.0)")

        # Verify a normal oncoming vehicle in the left carriageway is NOT flagged as wrong way
        vehicle = TrackedVehicle(
            track_id=9,
            class_id=2,
            class_name="Car",
            entity_type="vehicle",
            bbox=[100, 200, 160, 260],
            contact_img=(130, 260),
            contact_ground=(3.0, 50.0),
            timestamp=1.0
        )
        vehicle.vy_m_s = -15.0  # Normal oncoming flow
        vehicle.speed_kmh = 54.0

        alerts = self.behavior_engine.analyze_vehicle(vehicle, timestamp=1.0)
        wrong_way_alerts = [a for a in alerts if a.event_type == "WRONG_WAY_TRAVEL"]
        self.assertEqual(len(wrong_way_alerts), 0, "Car traveling with corridor flow must NOT be flagged as wrong way")

    def test_06_database_incident_and_feedback(self):
        """Tests SQLite incident insertion, querying, and human active feedback."""
        init_db()
        test_id = f"INC-TEST-{int(np.random.randint(1000, 9999))}"
        insert_event({
            "id": test_id,
            "video_id": "test_clip.mp4",
            "track_id": 42,
            "class_name": "Car",
            "event_type": "WAVERING_VEHICLE",
            "severity": "MODERATE",
            "reason": "Repeated lane drift & weaving (3 direction changes, peak deviation 1.1m across 3.5s)",
            "confidence": 0.91,
            "timestamp_s": 12.4,
            "speed_kmh": 65.0,
            "lane_id": 2
        })

        # Query events
        records = list_events(limit=10)
        found = [r for r in records if r["id"] == test_id]
        self.assertEqual(len(found), 1, "Inserted test event must be queryable")
        self.assertEqual(found[0]["user_feedback"], "unreviewed")

        # Update feedback to 'confirmed_tp'
        ok = update_event_feedback(test_id, "confirmed_tp", notes="Verified by human auditor")
        self.assertTrue(ok)

        # Check stats
        stats = get_event_stats()
        self.assertGreaterEqual(stats["total_events"], 1)
        self.assertGreaterEqual(stats["feedback_breakdown"]["confirmed_tp"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
