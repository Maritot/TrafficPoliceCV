"""
Stage 6: Deterministic Behaviour Analytics & Traffic Violation Engine.
Implements rule-based, explainable anomaly detection with hysteresis and persistence:
- WRONG_WAY_TRAVEL (Critical)
- STOPPED_IN_TRAVEL_LANE (Critical)
- LANE_DEPARTURE / BOUNDARY_CROSSING (Warning)
- HIGH_LATERAL_OSCILLATION (Warning)
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.geometry import RoadGeometryManager
from core.detector_tracker import TrackedVehicle


class AlertEvent:
    def __init__(self, event_type: str, severity: str, track_id: int, 
                 class_name: str, message: str, timestamp: float, 
                 metrics: Dict[str, float]):
        self.event_type = event_type
        self.severity = severity  # "CRITICAL", "WARNING", "ADVISORY"
        self.track_id = track_id
        self.class_name = class_name
        self.message = message
        self.timestamp = timestamp
        self.metrics = metrics


class BehaviorEngine:
    def __init__(self, geometry_mgr: RoadGeometryManager, rules_config: Dict):
        self.geometry_mgr = geometry_mgr
        self.rules_cfg = rules_config

        # Per-vehicle persistence and state tracking: track_id -> dict
        self.track_states: Dict[int, Dict] = {}

    def analyze_vehicle(self, vehicle: TrackedVehicle, timestamp: float) -> List[AlertEvent]:
        """
        Evaluates a single vehicle track against deterministic road safety rules.
        """
        tid = vehicle.track_id
        if tid not in self.track_states:
            self.track_states[tid] = {
                "wrong_way_frames": 0,
                "wrong_way_start_s": None,
                "lane_departure_frames": 0,
                "swerve_frames": 0,
                "lateral_positions": [],
                "lateral_velocities": [],
                "active_alerts": set()
            }

        state = self.track_states[tid]
        alerts: List[AlertEvent] = []

        gx, gy = vehicle.contact_ground
        vx, vy = vehicle.vx_m_s, vehicle.vy_m_s

        # Warmup period check: allow track velocity/acceleration filters to stabilize
        is_warmup = (len(vehicle.history_ground) < 8) or ((timestamp - vehicle.first_seen_time) < 0.5)

        # 1. Find assigned lane and compute Frenet road-relative coordinates
        lane = self.geometry_mgr.find_assigned_lane(gx, gy)
        if lane is None:
            return alerts

        frenet = lane.compute_frenet_state(gx, gy, vx, vy)
        s = frenet["s"]
        d = frenet["d"]
        d_norm = frenet["d_norm"]
        heading_dev = frenet["heading_dev_deg"]

        # Track continuous lateral position (gx) and velocity (vx) for true oscillation check
        state["lateral_positions"].append((gx, timestamp))
        state["lateral_velocities"].append((vx, timestamp))
        # Keep window to 3.5 seconds
        state["lateral_positions"] = [(pos, t) for pos, t in state["lateral_positions"] if timestamp - t <= 3.5]
        state["lateral_velocities"] = [(vel, t) for vel, t in state["lateral_velocities"] if timestamp - t <= 3.5]

        # -------------------------------------------------------------
        # Indian Road Rule A: CATTLE / ANIMAL ON CARRIAGEWAY (Critical)
        # -------------------------------------------------------------
        if vehicle.entity_type == "animal":
            animal_cfg = self.rules_cfg.get("animal_hazard", {})
            if animal_cfg.get("enabled", True):
                state["animal_hazard_frames"] = state.get("animal_hazard_frames", 0) + 1
                persist_needed = int(animal_cfg.get("persistence_frames", 8))
                if state["animal_hazard_frames"] >= persist_needed:
                    alert = AlertEvent(
                        event_type="CATTLE_ANIMAL_ON_CARRIAGEWAY",
                        severity="CRITICAL",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Live Road Hazard: {vehicle.class_name} #{tid} on {lane.name}",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "lane_id": lane.lane_id,
                            "lateral_offset_m": d
                        }
                    )
                    alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # Indian Road Rule B: STREET VENDOR / PUSHCART ENCROACHMENT (Warning)
        # -------------------------------------------------------------
        if vehicle.entity_type == "pushcart_vendor":
            vendor_cfg = self.rules_cfg.get("vendor_encroachment", {})
            if vendor_cfg.get("enabled", True):
                if vehicle.speed_kmh < float(vendor_cfg.get("speed_threshold_kmh", 4.0)):
                    state["vendor_dwell_frames"] = state.get("vendor_dwell_frames", 0) + 1
                    if state["vendor_dwell_frames"] >= 15:
                        alert = AlertEvent(
                            event_type="STREET_VENDOR_ENCROACHMENT",
                            severity="WARNING",
                            track_id=tid,
                            class_name=vehicle.class_name,
                            message=f"Road Encroachment: Street Vendor/Cart #{tid} obstructing {lane.name}",
                            timestamp=timestamp,
                            metrics={
                                "speed_kmh": vehicle.speed_kmh,
                                "lane_id": lane.lane_id,
                                "lateral_offset_m": d
                            }
                        )
                        alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # Indian Road Rule C: PEDESTRIAN IN LIVE TRAVEL LANE (Warning)
        # -------------------------------------------------------------
        if vehicle.entity_type == "pedestrian_vendor":
            ped_cfg = self.rules_cfg.get("pedestrian_hazard", {})
            if ped_cfg.get("enabled", True):
                state["ped_hazard_frames"] = state.get("ped_hazard_frames", 0) + 1
                if state["ped_hazard_frames"] >= int(ped_cfg.get("persistence_frames", 10)):
                    alert = AlertEvent(
                        event_type="PEDESTRIAN_IN_TRAVEL_LANE",
                        severity="WARNING",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Vulnerable Pedestrian #{tid} walking in live {lane.name}",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "lane_id": lane.lane_id,
                            "lateral_offset_m": d
                        }
                    )
                    alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # Indian Road Rule D: POTHOLE / MUD SWERVE DETECTION (Warning)
        # -------------------------------------------------------------
        swerve_cfg = self.rules_cfg.get("pothole_swerve", {})
        if swerve_cfg.get("enabled", True) and vehicle.entity_type == "vehicle" and not is_warmup:
            min_accel = float(swerve_cfg.get("min_lateral_accel_m_s2", 3.5))
            persist_needed = int(swerve_cfg.get("persistence_frames", 3))
            if abs(vehicle.ax_m_s2) > min_accel and vehicle.speed_kmh > 18.0:
                state["swerve_frames"] = state.get("swerve_frames", 0) + 1
                if state["swerve_frames"] >= persist_needed:
                    alert = AlertEvent(
                        event_type="POTHOLE_OR_MUD_SWERVE",
                        severity="WARNING",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Sharp Pothole/Mud Evasive Swerve by {vehicle.class_name} #{tid} ({vehicle.ax_m_s2:+.1f} m/s2)",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "lateral_accel_m_s2": vehicle.ax_m_s2,
                            "lane_id": lane.lane_id
                        }
                    )
                    alerts.append(alert)
            else:
                state["swerve_frames"] = max(0, state.get("swerve_frames", 0) - 1)

        # -------------------------------------------------------------
        # Rule 1: WRONG_WAY_TRAVEL (Critical)
        # -------------------------------------------------------------
        ww_cfg = self.rules_cfg.get("wrong_way", {})
        if ww_cfg.get("enabled", True):
            min_heading_deg = float(ww_cfg.get("min_heading_dev_deg", 125.0))
            persist_needed = int(ww_cfg.get("persistence_frames", 10))

            if heading_dev >= min_heading_deg and vehicle.speed_kmh > 4.0:
                state["wrong_way_frames"] += 1
                if state["wrong_way_start_s"] is None:
                    state["wrong_way_start_s"] = s
                
                # Check persistence and distance traveled opposite
                delta_s = abs(s - state["wrong_way_start_s"])
                if state["wrong_way_frames"] >= persist_needed and delta_s >= 2.0:
                    alert = AlertEvent(
                        event_type="WRONG_WAY_TRAVEL",
                        severity="CRITICAL",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Wrong-way {vehicle.class_name} #{tid} on {lane.name} (Angle: {heading_dev:.0f}°)",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "heading_dev_deg": heading_dev,
                            "lane_id": lane.lane_id,
                            "lateral_offset_m": d
                        }
                    )
                    alerts.append(alert)
                    state["active_alerts"].add("WRONG_WAY_TRAVEL")
            else:
                state["wrong_way_frames"] = max(0, state["wrong_way_frames"] - 2)
                if state["wrong_way_frames"] == 0:
                    state["wrong_way_start_s"] = None
                    state["active_alerts"].discard("WRONG_WAY_TRAVEL")

        # -------------------------------------------------------------
        # Rule 2: STOPPED_IN_TRAVEL_LANE (Critical)
        # -------------------------------------------------------------
        stopped_cfg = self.rules_cfg.get("stopped_vehicle", {})
        if stopped_cfg.get("enabled", True):
            min_stopped_s = float(stopped_cfg.get("min_stopped_seconds", 3.0))
            if vehicle.stopped_duration_s >= min_stopped_s:
                alert = AlertEvent(
                    event_type="STOPPED_IN_TRAVEL_LANE",
                    severity="CRITICAL",
                    track_id=tid,
                    class_name=vehicle.class_name,
                    message=f"Stationary {vehicle.class_name} #{tid} in {lane.name} for {vehicle.stopped_duration_s:.1f}s",
                    timestamp=timestamp,
                    metrics={
                        "speed_kmh": vehicle.speed_kmh,
                        "stopped_duration_s": vehicle.stopped_duration_s,
                        "lane_id": lane.lane_id,
                        "lateral_offset_m": d
                    }
                )
                alerts.append(alert)
                state["active_alerts"].add("STOPPED_IN_TRAVEL_LANE")
            else:
                state["active_alerts"].discard("STOPPED_IN_TRAVEL_LANE")

        # -------------------------------------------------------------
        # Rule 3: LANE_DEPARTURE / BOUNDARY_CROSSING (Warning)
        # -------------------------------------------------------------
        lane_cfg = self.rules_cfg.get("lane_keeping", {})
        if lane_cfg.get("enabled", True):
            violation_ratio = float(lane_cfg.get("violation_ratio", 1.05))
            if abs(d_norm) > violation_ratio and vehicle.speed_kmh > 8.0:
                state["lane_departure_frames"] += 1
                if state["lane_departure_frames"] >= 8:
                    alert = AlertEvent(
                        event_type="LANE_DEPARTURE",
                        severity="WARNING",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"{vehicle.class_name} #{tid} crossed lane boundary ({d:+.2f}m from center)",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "normalized_offset": d_norm,
                            "lateral_offset_m": d,
                            "lane_id": lane.lane_id
                        }
                    )
                    alerts.append(alert)
            else:
                state["lane_departure_frames"] = max(0, state["lane_departure_frames"] - 1)

        # -------------------------------------------------------------
        # Rule 4: HIGH_LATERAL_OSCILLATION (Weaving)
        # -------------------------------------------------------------
        osc_cfg = self.rules_cfg.get("oscillation", {})
        if osc_cfg.get("enabled", True) and vehicle.class_name != "Motorcycle" and not is_warmup:
            # Evaluate only when vehicle is moving at reasonable speed
            positions = [pos for pos, t in state["lateral_positions"]]
            if len(positions) >= 15 and vehicle.speed_kmh > 15.0:
                # 1. Continuous road lateral amplitude (peak-to-trough in meters)
                amplitude = max(positions) - min(positions) if positions else 0.0

                # 2. Count significant directional reversals (filter camera jitter below 0.25 m/s)
                raw_vels = [v for v, t in state["lateral_velocities"]]
                sig_vels = [v for v in raw_vels if abs(v) > 0.25]

                if len(sig_vels) >= 4:
                    signs = np.sign(sig_vels)
                    sign_changes = int(np.sum(np.diff(signs) != 0))
                else:
                    sign_changes = 0

                min_flips = int(osc_cfg.get("min_sign_flips", 3))
                amp_thresh = float(osc_cfg.get("amplitude_threshold_m", 0.60))

                if sign_changes >= min_flips and amplitude >= amp_thresh:
                    alert = AlertEvent(
                        event_type="HIGH_LATERAL_OSCILLATION",
                        severity="WARNING",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Erratic weaving detected for {vehicle.class_name} #{tid} (Amp: {amplitude:.2f}m)",
                        timestamp=timestamp,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "amplitude_m": amplitude,
                            "sign_changes": sign_changes,
                            "lane_id": lane.lane_id
                        }
                    )
                    alerts.append(alert)

        return alerts
