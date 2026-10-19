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
        Renders the metric ground road, lanes, and moving vehicle dots with adaptive bounds.
        """
        h, w = canvas.shape[:2]
        radar_size = max(90, int(min(w, h) * 0.35))
        margin = max(6, int(w * 0.02))
        top = max(10, int(h * 0.06))
        rx1, ry1 = w - radar_size - margin, top
        rx2, ry2 = rx1 + radar_size, ry1 + radar_size

        # Create radar canvas with high-tech surveillance look
        radar = np.full((radar_size, radar_size, 3), 16, dtype=np.uint8)
        
        # Border
        cv2.rectangle(radar, (0, 0), (radar_size - 1, radar_size - 1), (0, 180, 255), 1)
        f_scale = max(0.28, 0.38 * (radar_size / 220.0))
        cv2.putText(radar, "BEV RADAR", (8, int(16 * (radar_size / 220.0))), 
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 220, 255), 1, cv2.LINE_AA)

        inset = int(radar_size * 0.12)
        inner_w = radar_size - 2 * inset
        inner_h = radar_size - 2 * inset

        # Ground calibration bounds
        x_min, x_max, y_min, y_max = self.geometry_mgr.calibrator.get_ground_bounds()

        # Check if vehicles have reasonable ground projection coordinates
        valid_ground_pts = [v.contact_ground for v in vehicles if v.contact_ground[0] > -5 and v.contact_ground[1] > -5]
        use_metric_ground = len(valid_ground_pts) > 0 and (y_max - y_min) > 5.0

        if use_metric_ground:
            # Dynamically adapt radar bounds to active vehicles so none are clipped off
            all_xs = [pt[0] for pt in valid_ground_pts] + [x_min, x_max]
            all_ys = [pt[1] for pt in valid_ground_pts] + [y_min, y_max]
            gx_min = min(all_xs) - 1.0
            gx_max = max(all_xs) + 1.0
            gy_min = max(-5.0, min(all_ys) - 2.0)
            gy_max = max(max(all_ys) + 5.0, y_max)
            x_span = max(gx_max - gx_min, 1.0)
            y_span = max(gy_max - gy_min, 1.0)
        else:
            gx_min, gx_max, gy_min, gy_max, x_span, y_span = 0.0, 15.0, 0.0, 100.0, 15.0, 100.0

        # Draw structured lane boundaries top-down
        lane_fractions = [0.18, 0.38, 0.58, 0.78, 0.92]
        for f in lane_fractions:
            lx = int(inset + f * inner_w)
            cv2.line(radar, (lx, int(22 * (radar_size / 220.0))), (lx, radar_size - 8), (40, 60, 45), 1)
        for f in lane_fractions[1:-1]:
            lx = int(inset + f * inner_w)
            for y_step in range(int(24 * (radar_size / 220.0)), radar_size - 10, 10):
                cv2.line(radar, (lx, y_step), (lx, min(y_step + 5, radar_size - 10)), (75, 100, 75), 1)

        def to_radar_coords(v: TrackedVehicle) -> Tuple[int, int]:
            if use_metric_ground:
                gx, gy = v.contact_ground
                px = int(inset + ((gx - gx_min) / x_span) * inner_w)
                py = int((radar_size - inset) - ((gy - gy_min) / y_span) * inner_h)
            else:
                # High-reliability fallback: Normalized road perspective projection directly from image (u, v)
                u, v_y = v.contact_img
                v_horiz = max(1.0, 0.35 * h)
                v_bot = 0.95 * h
                y_norm = np.clip((v_bot - v_y) / max(v_bot - v_horiz, 1.0), 0.06, 0.92)
                x_norm = np.clip(u / max(float(w), 1.0), 0.08, 0.92)
                px = int(inset + x_norm * inner_w)
                py = int((radar_size - inset) - y_norm * inner_h)
            return int(np.clip(px, 6, radar_size - 6)), int(np.clip(py, 6, radar_size - 6))

        # Plot vehicles in BEV
        for v in vehicles:
            r_px, r_py = to_radar_coords(v)
            
            sev = alert_map.get(v.track_id, "NORMAL")
            if sev == "CRITICAL":
                dot_color = (0, 0, 255)        # Red alert
            elif v.entity_type == "animal":
                dot_color = (0, 200, 255)      # Amber for Cattle/Stray Animal
            elif v.entity_type == "pushcart_vendor":
                dot_color = (220, 60, 220)     # Magenta for Street Vendor
            elif v.entity_type == "pedestrian_vendor":
                dot_color = (255, 200, 50)     # Light blue for Pedestrian
            elif "Truck" in v.class_name or "Bus" in v.class_name:
                dot_color = (50, 220, 255)     # Cyan/Yellow for Heavy commercial vehicles
            else:
                dot_color = (0, 255, 120)      # Green for Normal Vehicle

            dot_rad = max(3, int(4 * (radar_size / 220.0)))
            # Halo outer circle + filled center for clear visibility
            cv2.circle(radar, (r_px, r_py), dot_rad + 2, dot_color, 1, cv2.LINE_AA)
            cv2.circle(radar, (r_px, r_py), dot_rad, dot_color, -1, cv2.LINE_AA)
            
            # Draw directional velocity vector arrow
            arrow_scale = max(1.0, radar_size / 150.0)
            if abs(v.vy_m_s) > 0.5:
                arrow_dx = int(np.clip(v.vx_m_s * arrow_scale, -12, 12))
                arrow_dy = int(np.clip(-v.vy_m_s * arrow_scale, -18, 18))
            else:
                arrow_dx = 0
                arrow_dy = -int(max(6, 12 * (radar_size / 220.0)))
            
            cv2.arrowedLine(radar, (r_px, r_py), (r_px + arrow_dx, r_py + arrow_dy), 
                            (255, 255, 255), 1, tipLength=0.35)

        # Overlay radar on the canvas
        canvas[ry1:ry2, rx1:rx2] = radar
