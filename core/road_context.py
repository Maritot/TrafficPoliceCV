"""
Project WAV-DET - Road Context & Spatial Drivability Engine.
Understands road boundaries, drivable surface vs. shoulder/off-road fields,
and automatically discovers bidirectional traffic flow to eliminate false alarms.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from core.calibration import HomographyCalibrator


class RoadContextManager:
    """
    Manages spatial drivability zones, road corridor boundaries,
    and adaptive traffic flow directions.
    """

    def __init__(self, calibrator: HomographyCalibrator, road_width_m: float = 16.0, road_depth_m: float = 120.0):
        self.calibrator = calibrator
        self.road_width_m = road_width_m
        self.road_depth_m = road_depth_m

        # Dynamic bidirectional corridor flow trackers
        # Clusters flow vectors into primary travel directions
        # Negative vy (moving towards camera, decreasing Y) vs Positive vy (moving away, increasing Y)
        self.observed_headings: List[float] = []
        self.corridor_flows: Dict[str, float] = {
            "left_carriageway": -1.0,   # Traffic flowing towards camera by default in left carriageway
            "right_carriageway": 1.0    # Traffic flowing away from camera in right carriageway
        }
        
        # Heading observations for auto-calibration of dual carriageways
        self.flow_history: Dict[int, List[float]] = {}  # track_id -> [vy values]

    def register_vehicle_velocity(self, track_id: int, ground_x: float, ground_y: float, vy_m_s: float):
        """
        Learns corridor traffic flow dynamically from observed vehicle kinematics.
        Adapts to dual carriageways and bidirectional roads.
        """
        if abs(vy_m_s) < 1.0:
            return  # Disregard stationary or near-stopped vehicles

        if track_id not in self.flow_history:
            self.flow_history[track_id] = []
        
        self.flow_history[track_id].append(vy_m_s)
        if len(self.flow_history[track_id]) > 10:
            self.flow_history[track_id].pop(0)

        # Update corridor flows based on spatial position (median divider at x ~ half width)
        median_x = self.road_width_m / 2.0
        if ground_x < median_x:
            # Left corridor
            if len(self.flow_history[track_id]) >= 5:
                median_v = float(np.median(self.flow_history[track_id]))
                self.corridor_flows["left_carriageway"] = 1.0 if median_v > 0 else -1.0
        else:
            # Right corridor
            if len(self.flow_history[track_id]) >= 5:
                median_v = float(np.median(self.flow_history[track_id]))
                self.corridor_flows["right_carriageway"] = 1.0 if median_v > 0 else -1.0

    def get_expected_flow_direction(self, ground_x: float) -> float:
        """
        Returns expected travel direction (+1.0 for moving away, -1.0 for moving towards).
        """
        median_x = self.road_width_m / 2.0
        if ground_x < median_x:
            return self.corridor_flows.get("left_carriageway", -1.0)
        else:
            return self.corridor_flows.get("right_carriageway", 1.0)

    def is_on_drivable_road(self, ground_x: float, ground_y: float) -> Tuple[bool, str]:
        """
        Determines whether a vehicle's ground contact position is on the active carriageway,
        on the paved shoulder, or completely off-road in a roadside field/ditch.

        Returns:
            (is_on_road: bool, zone: "CARRIAGEWAY" | "SHOULDER" | "OFF_ROAD")
        """
        # Road boundaries in meters
        carriageway_left = 0.5
        carriageway_right = self.road_width_m - 0.5
        shoulder_left = -2.5
        shoulder_right = self.road_width_m + 2.5

        if carriageway_left <= ground_x <= carriageway_right and -5.0 <= ground_y <= self.road_depth_m + 15.0:
            return True, "CARRIAGEWAY"
        elif shoulder_left <= ground_x <= shoulder_right and -10.0 <= ground_y <= self.road_depth_m + 25.0:
            return True, "SHOULDER"
        else:
            return False, "OFF_ROAD"

    def is_agricultural_or_special_vehicle(self, class_name: str, speed_kmh: float) -> bool:
        """
        Identifies agricultural tractors, harvesters, slow-moving construction equipment,
        or roadside service machinery that legitimately travel differently.
        """
        c_lower = class_name.lower()
        if any(term in c_lower for term in ["tractor", "harvester", "machinery", "cart", "pushcart"]):
            return True
        # Slow-moving utility or heavy vehicle on the shoulder
        if ("truck" in c_lower or "van" in c_lower) and speed_kmh < 15.0:
            return True
        return False
