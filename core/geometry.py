"""
Stage 1: Road & Lane Geometry Engine.
Manages metric lane boundaries, centerline calculation, spline smoothing,
and Frenet frame coordinate transformations (s, d, d_normalized, heading_deviation).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional
from core.calibration import HomographyCalibrator


class Lane:
    def __init__(self, lane_id: int, name: str, left_x: float, right_x: float, 
                 center_x: float, y_min: float, y_max: float, direction_y: float = 1.0,
                 speed_limit_kmh: float = 60.0):
        """
        Represents a structured highway/arterial lane in metric ground coordinates.
        :param lane_id: Unique integer identifier
        :param name: Human-readable label (e.g. "Lane 1")
        :param left_x: X coordinate of left boundary (meters)
        :param right_x: X coordinate of right boundary (meters)
        :param center_x: X coordinate of centerline (meters)
        :param y_min: Start Y coordinate of lane (meters)
        :param y_max: End Y coordinate of lane (meters)
        :param direction_y: Normal flow direction (+1.0 for +Y, -1.0 for -Y)
        :param speed_limit_kmh: Posted speed limit
        """
        self.lane_id = lane_id
        self.name = name
        self.left_x = left_x
        self.right_x = right_x
        self.center_x = center_x
        self.y_min = y_min
        self.y_max = y_max
        self.direction_y = np.sign(direction_y)
        self.speed_limit_kmh = speed_limit_kmh
        
        self.width_m = abs(right_x - left_x)
        self.half_width_m = self.width_m / 2.0

        # Discretized spline points along Y (arc-length s proxy for straight/curved segments)
        num_pts = 50
        y_samples = np.linspace(y_min, y_max, num_pts)
        
        self.left_pts = np.column_stack([np.full(num_pts, left_x), y_samples])
        self.right_pts = np.column_stack([np.full(num_pts, right_x), y_samples])
        self.center_pts = np.column_stack([np.full(num_pts, center_x), y_samples])

        # Tangent vector: along travel direction
        self.tangent = np.array([0.0, self.direction_y], dtype=np.float32)
        # Unit normal vector pointing to the right of the centerline
        self.normal = np.array([1.0, 0.0], dtype=np.float32)

    def contains_ground_point(self, x: float, y: float, margin_m: float = 0.5) -> bool:
        """Check if a ground point falls within this lane polygon (with tolerance)."""
        x_min = min(self.left_x, self.right_x) - margin_m
        x_max = max(self.left_x, self.right_x) + margin_m
        y_min = min(self.y_min, self.y_max) - margin_m
        y_max = max(self.y_min, self.y_max) + margin_m
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def compute_frenet_state(self, x: float, y: float, vx: float, vy: float) -> Dict[str, float]:
        """
        Compute road-relative Frenet coordinates:
        - s: longitudinal progress along road
        - d: signed lateral offset from centerline (+ is right, - is left)
        - d_norm: normalized lateral offset (d / half_width)
        - heading_dev_deg: angle between vehicle velocity vector and road direction
        """
        s = float(y - self.y_min)
        # Signed lateral offset from centerline
        d = float(x - self.center_x)
        d_norm = d / max(self.half_width_m, 1e-4)

        # Heading deviation
        v_norm = np.hypot(vx, vy)
        if v_norm > 0.3:  # Only compute heading if moving noticeably
            v_dir = np.array([vx, vy]) / v_norm
            # Dot product with tangent vector
            dot_val = np.clip(np.dot(v_dir, self.tangent), -1.0, 1.0)
            heading_dev_deg = float(np.degrees(np.arccos(dot_val)))
        else:
            heading_dev_deg = 0.0

        return {
            "s": s,
            "d": d,
            "d_norm": d_norm,
            "heading_dev_deg": heading_dev_deg,
            "lane_id": self.lane_id,
            "lane_name": self.name
        }


class RoadGeometryManager:
    """Manages all lanes on a monitored road corridor."""
    def __init__(self, lanes_config: List[Dict], calibrator: HomographyCalibrator):
        self.calibrator = calibrator
        self.lanes: List[Lane] = []

        min_x, max_x, min_y, max_y = calibrator.get_ground_bounds()

        for cfg in lanes_config:
            lane = Lane(
                lane_id=cfg["id"],
                name=cfg.get("name", f"Lane {cfg['id']}"),
                left_x=float(cfg["left_boundary_x"]),
                right_x=float(cfg["right_boundary_x"]),
                center_x=float(cfg["centerline_x"]),
                y_min=min_y,
                y_max=max_y,
                direction_y=float(cfg.get("direction_y", 1.0)),
                speed_limit_kmh=float(cfg.get("speed_limit_kmh", 60.0))
            )
            self.lanes.append(lane)

    def find_assigned_lane(self, ground_x: float, ground_y: float, max_margin_m: float = 0.8) -> Optional[Lane]:
        """Assigns a vehicle ground point to the nearest or enclosing lane within road boundaries."""
        for lane in self.lanes:
            if lane.contains_ground_point(ground_x, ground_y, margin_m=0.2):
                return lane
        # If outside boundaries, only assign if within reasonable proximity of road shoulder
        best_lane = None
        min_dist = float("inf")
        for lane in self.lanes:
            if lane.contains_ground_point(ground_x, ground_y, margin_m=max_margin_m):
                dist = abs(ground_x - lane.center_x)
                if dist < min_dist:
                    min_dist = dist
                    best_lane = lane
        return best_lane

    def get_image_render_polylines(self) -> List[Dict]:
        """
        Returns pixel polyline coordinates for all boundaries and centerlines
        to be rendered onto the camera video stream.
        """
        renders = []
        for lane in self.lanes:
            left_img = self.calibrator.project_ground_to_image(lane.left_pts)
            right_img = self.calibrator.project_ground_to_image(lane.right_pts)
            center_img = self.calibrator.project_ground_to_image(lane.center_pts)

            renders.append({
                "lane_id": lane.lane_id,
                "name": lane.name,
                "left_img": left_img.astype(np.int32),
                "right_img": right_img.astype(np.int32),
                "center_img": center_img.astype(np.int32)
            })
        return renders
