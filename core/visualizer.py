"""
Stage 9: Real-Time Computer Vision Visualizer & HUD Engine.
Renders lane boundaries, centerlines, vehicle ground-contact points,
motion trajectory tails, alert banners, and a live top-down Bird's-Eye View (BEV) radar.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.detector_tracker import TrackedVehicle
from core.behavior_engine import AlertEvent
from core.geometry import RoadGeometryManager


class LATISVisualizer:
    def __init__(self, geometry_mgr: RoadGeometryManager, config: Dict):
        self.geometry_mgr = geometry_mgr
        self.vis_cfg = config.get("visualization", {})
        
        self.show_bev_radar = self.vis_cfg.get("show_bev_radar", True)
        self.show_lanes = self.vis_cfg.get("show_lanes", True)
        self.show_trails = self.vis_cfg.get("show_trails", True)
        self.trail_len = int(self.vis_cfg.get("trail_length", 20))

        # Cached polylines in image coordinates
        self.lane_renders = self.geometry_mgr.get_image_render_polylines()

        # Alert banner state
        self.active_alert_banner: Optional[AlertEvent] = None
        self.alert_banner_expire_time = 0.0

    def draw(self, frame: np.ndarray, vehicles: List[TrackedVehicle], 
             alerts: List[AlertEvent], timestamp: float, fps: float) -> np.ndarray:
        """
        Overlays all perception, tracking, lane geometry, and alert data onto the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Update Alert Banner State
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        if critical_alerts:
            self.active_alert_banner = critical_alerts[0]
            self.alert_banner_expire_time = timestamp + 1.5
        elif alerts:
            self.active_alert_banner = alerts[0]
            self.alert_banner_expire_time = timestamp + 1.0

        # 2. Draw Lane Boundaries and Centerlines (if enabled in config)
        if self.show_lanes:
            self._draw_road_geometry(annotated)

        # 3. Map alerts by track_id for visual styling
        alert_map: Dict[int, str] = {}
        for a in alerts:
            alert_map[a.track_id] = a.severity

        # 4. Draw Vehicles, Bounding Boxes, and Motion Trails
        for v in vehicles:
            severity = alert_map.get(v.track_id, "NORMAL")
            self._draw_vehicle(annotated, v, severity)

        # 5. Draw Alert Banner if active
        if self.active_alert_banner and timestamp < self.alert_banner_expire_time:
            self._draw_alert_banner(annotated, self.active_alert_banner)

        # 6. Draw System Status HUD (Top Left)
        self._draw_system_hud(annotated, len(vehicles), fps)

        # 7. Draw BEV Radar Mini-Map (Top Right)
        if self.show_bev_radar:
            self._draw_bev_radar(annotated, vehicles, alert_map)

        return annotated

    def _draw_road_geometry(self, canvas: np.ndarray):
        """Draws lane boundaries and centerlines in image perspective."""
        overlay = canvas.copy()

        for lr in self.lane_renders:
            left_pts = lr["left_img"].reshape((-1, 1, 2))
            right_pts = lr["right_img"].reshape((-1, 1, 2))
            center_pts = lr["center_img"].reshape((-1, 1, 2))

            # Fill lane corridor with subtle green tint
            lane_poly = np.vstack([left_pts, np.flip(right_pts, axis=0)])
            cv2.fillPoly(overlay, [lane_poly], (35, 95, 35))

            # Left/Right boundaries (Solid Green/Yellow)
            cv2.polylines(canvas, [left_pts], isClosed=False, color=(50, 220, 50), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [right_pts], isClosed=False, color=(50, 220, 50), thickness=2, lineType=cv2.LINE_AA)

            # Centerline (Cyan dashed-style line)
            for i in range(0, len(center_pts) - 1, 2):
                pt1 = tuple(center_pts[i][0])
                pt2 = tuple(center_pts[min(i + 1, len(center_pts) - 1)][0])
                cv2.line(canvas, pt1, pt2, (255, 230, 0), thickness=2, lineType=cv2.LINE_AA)

        # Blend lane fill transparency
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    def _draw_vehicle(self, canvas: np.ndarray, v: TrackedVehicle, severity: str):
        """Renders bounding box, ID tag, ground contact, and trajectory tail with entity-aware styling."""
        x1, y1, x2, y2 = map(int, v.bbox)

        # Entity-aware color scheme
        if severity == "CRITICAL":
            box_color = (0, 0, 255)      # Bright Red
            txt_bg = (0, 0, 200)
        elif severity == "WARNING":
            box_color = (0, 140, 255)    # Orange
            txt_bg = (0, 120, 220)
        elif v.entity_type == "animal":
            box_color = (0, 200, 255)    # Amber/Gold for Cattle & Stray Animals
            txt_bg = (0, 140, 200)
        elif v.entity_type == "pushcart_vendor":
            box_color = (220, 60, 220)   # Magenta for Street Vendors on wheels
            txt_bg = (160, 40, 160)
        elif v.entity_type == "pedestrian_vendor":
            box_color = (255, 200, 50)   # Light Blue for Pedestrians
            txt_bg = (180, 140, 20)
        else:
            box_color = (0, 255, 120)    # Green for Normal Vehicles
            txt_bg = (30, 140, 30)

        # 1. Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # 2. Ground Contact Point (where tire/hoof/foot touches road)
        cx, cy = map(int, v.contact_img)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 8, box_color, 1, cv2.LINE_AA)

        # 3. Motion Trajectory Trail
        if self.show_trails and len(v.history_img) > 1:
            pts = np.array(v.history_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=box_color, thickness=2, lineType=cv2.LINE_AA)

        # 4. Label Badge: [ID: 2 | Class | Speed]
        if v.entity_type in ["animal", "pushcart_vendor", "pedestrian_vendor"]:
            label = f"#{v.track_id} {v.class_name}"
        else:
            label = f"#{v.track_id} {v.class_name} | {v.speed_kmh:.0f} km/h"

        v_scale = max(0.30, min(canvas.shape[1], canvas.shape[0]) / 720.0)
        f_size = max(0.25, 0.45 * v_scale)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, f_size, 1)
        
        tag_y1 = max(0, y1 - lh - 4)
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + lw + 4, y1), txt_bg, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_size, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, canvas: np.ndarray, alert: AlertEvent):
        """Draws high-visibility alert banner at the top of the video screen."""
        h, w = canvas.shape[:2]
        scale = max(0.35, min(w, h) / 720.0)
        banner_h = max(22, int(44 * scale))
        
        bg_color = (0, 0, 220) if alert.severity == "CRITICAL" else (0, 140, 255)
        cv2.rectangle(canvas, (0, 0), (w, banner_h), bg_color, -1)
        cv2.line(canvas, (0, banner_h), (w, banner_h), (255, 255, 255), 1)

        f_scale = max(0.30, 0.55 * scale)
        icon = "[CRITICAL]" if alert.severity == "CRITICAL" else "[WARNING]"
        text = f"{icon} {alert.message}"
        cv2.putText(canvas, text, (int(15 * scale), int(banner_h * 0.70)), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_system_hud(self, canvas: np.ndarray, vehicle_count: int, fps: float):
        """Draws system metrics badge in top-left with resolution scaling."""
        h, w = canvas.shape[:2]
        scale = max(0.35, min(w, h) / 720.0)
        hud_w = int(240 * scale)
        hud_h = int(75 * scale)
        pad = int(10 * scale)
        top = int(45 * scale)

        cv2.rectangle(canvas, (pad, top), (pad + hud_w, top + hud_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (pad, top), (pad + hud_w, top + hud_h), (80, 80, 80), 1)

        f_scale = 0.45 * scale
        cv2.putText(canvas, "LATIS TRAFFIC AI", (pad + 6, top + int(18 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (pad + 6, top + int(38 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Tracked: {vehicle_count}", (pad + 6, top + int(58 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (100, 255, 100), 1, cv2.LINE_AA)

    def _draw_bev_radar(self, canvas: np.ndarray, vehicles: List[TrackedVehicle], alert_map: Dict[int, str]):
        """
        Draws a 2D top-down Bird's-Eye View (BEV) radar mini-map in the top-right corner.
        Renders the metric ground road, lanes, and moving vehicle dots.
        """
        h, w = canvas.shape[:2]
        radar_size = max(70, int(min(w, h) * 0.35))
        margin = max(5, int(w * 0.02))
        top = max(10, int(h * 0.08))
        rx1, ry1 = w - radar_size - margin, top
        rx2, ry2 = rx1 + radar_size, ry1 + radar_size

        # Create radar canvas
        radar = np.full((radar_size, radar_size, 3), 25, dtype=np.uint8)
        
        # Border
        cv2.rectangle(radar, (0, 0), (radar_size - 1, radar_size - 1), (0, 180, 255), 1)
        f_scale = max(0.25, 0.35 * (radar_size / 220.0))
        cv2.putText(radar, "BEV RADAR", (6, int(15 * (radar_size / 220.0))), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 220, 255), 1)

        x_min, x_max, y_min, y_max = self.geometry_mgr.calibrator.get_ground_bounds()
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)

        inset = int(radar_size * 0.12)
        inner_w = radar_size - 2 * inset
        inner_h = radar_size - 2 * inset

        def to_radar_coords(gx: float, gy: float) -> Tuple[int, int]:
            px = int(inset + ((gx - x_min) / x_span) * inner_w)
            py = int((radar_size - inset) - ((gy - y_min) / y_span) * inner_h)
            return int(np.clip(px, 2, radar_size - 2)), int(np.clip(py, 2, radar_size - 2))

        # Draw lane boundaries top-down
        for lane in self.geometry_mgr.lanes:
            lx_pt1 = to_radar_coords(lane.left_x, y_min)
            lx_pt2 = to_radar_coords(lane.left_x, y_max)
            rx_pt1 = to_radar_coords(lane.right_x, y_min)
            rx_pt2 = to_radar_coords(lane.right_x, y_max)
            cx_pt1 = to_radar_coords(lane.center_x, y_min)
            cx_pt2 = to_radar_coords(lane.center_x, y_max)

            cv2.line(radar, lx_pt1, lx_pt2, (70, 70, 70), 1)
            cv2.line(radar, rx_pt1, rx_pt2, (70, 70, 70), 1)
            cv2.line(radar, cx_pt1, cx_pt2, (100, 100, 50), 1, cv2.LINE_AA)

        # Plot vehicles in BEV
        for v in vehicles:
            gx, gy = v.contact_ground
            r_px, r_py = to_radar_coords(gx, gy)
            
            sev = alert_map.get(v.track_id, "NORMAL")
            if sev == "CRITICAL":
                dot_color = (0, 0, 255)        # Red alert
            elif v.entity_type == "animal":
                dot_color = (0, 200, 255)      # Amber for Cattle/Stray Animal
            elif v.entity_type == "pushcart_vendor":
                dot_color = (220, 60, 220)     # Magenta for Street Vendor
            elif v.entity_type == "pedestrian_vendor":
                dot_color = (255, 200, 50)     # Light blue for Pedestrian
            else:
                dot_color = (0, 255, 120)      # Green for Normal Vehicle

            dot_rad = max(2, int(4 * (radar_size / 220.0)))
            cv2.circle(radar, (r_px, r_py), dot_rad, dot_color, -1)
            
            # Draw velocity vector arrow
            if v.speed_kmh > 3.0:
                arrow_scale = max(1.0, radar_size / 150.0)
                arrow_dx = int(v.vx_m_s * arrow_scale)
                arrow_dy = int(-v.vy_m_s * arrow_scale)
                cv2.arrowedLine(radar, (r_px, r_py), (r_px + arrow_dx, r_py + arrow_dy), 
                                (255, 255, 255), 1, tipLength=0.3)

        # Overlay radar on the canvas
        canvas[ry1:ry2, rx1:rx2] = radar
