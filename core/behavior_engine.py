"""
Project WAV-DET - Deterministic Behaviour Analytics & Traffic Safety Engine.
Integrates RoadContextManager and WaveringDetector to eliminate false alarms:
- High-precision wavering detection (repeated lane drift, high-G swerves, unstable speed)
- Spatial drivability gating (exempts off-road tractors, fields, and shoulders)
- Bidirectional corridor flow awareness (eliminates dual-carriageway false wrong-way alarms)
- Calibrated severity levels (INFO, MODERATE, SEVERE) with explicit justifications
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.geometry import RoadGeometryManager, Lane
from core.detector_tracker import TrackedVehicle
from core.road_context import RoadContextManager
from core.wavering_detector import WaveringDetector, WaveringAssessment


class AlertEvent:
    def __init__(self, event_type: str, severity: str, track_id: int, 
                 class_name: str, message: str, timestamp: float, 
                 metrics: Dict[str, float], confidence: float = 0.90):
        self.event_type = event_type
        self.severity = severity  # "INFO", "MODERATE", "SEVERE"
        self.track_id = track_id
        self.class_name = class_name
        self.message = message
        self.timestamp = timestamp
        self.metrics = metrics
        self.confidence = confidence


class BehaviorEngine:
    def __init__(self, geometry_mgr: RoadGeometryManager, rules_config: Dict,
                 road_context: Optional[RoadContextManager] = None):
        self.geometry_mgr = geometry_mgr
        self.rules_cfg = rules_config
        self.road_context = road_context or RoadContextManager(geometry_mgr.calibrator)
        self.wavering_detector = WaveringDetector(window_seconds=3.5, min_samples=8)

        # Per-vehicle persistence and state tracking: track_id -> dict
        self.track_states: Dict[int, Dict] = {}

    def analyze_vehicle(self, vehicle: TrackedVehicle, timestamp: float) -> List[AlertEvent]:
        """
        Evaluates a vehicle track against road safety rules, spatial context,
        and kinematic wavering models.
        """
        tid = vehicle.track_id
        if tid not in self.track_states:
            self.track_states[tid] = {
                "wrong_way_frames": 0,
                "wrong_way_start_s": None,
                "lane_departure_frames": 0,
                "wavering_frames": 0,
                "active_alerts": set()
            }

        state = self.track_states[tid]
        alerts: List[AlertEvent] = []

        gx, gy = vehicle.contact_ground
        vx, vy = vehicle.vx_m_s, vehicle.vy_m_s

        # 1. Road Context & Spatial Gating
        is_on_road, zone = self.road_context.is_on_drivable_road(gx, gy)
        is_agri = self.road_context.is_agricultural_or_special_vehicle(vehicle.class_name, vehicle.speed_kmh)
        self.road_context.register_vehicle_velocity(tid, gx, gy, vy)

        # Context Filter: Vehicles off the carriageway (e.g. tractor in field, roadside parking)
        # are completely exempt from carriageway discipline and wavering alerts.
        if zone == "OFF_ROAD":
            return alerts

        # Agricultural machinery or slow utility vehicles operating on the paved shoulder
        if is_agri and zone == "SHOULDER":
            return alerts

        # Warmup check: allow track velocity filters to stabilize
        is_warmup = (len(vehicle.history_ground) < 8) or ((timestamp - vehicle.first_seen_time) < 0.5)

        # 2. Frenet Lane Alignment
        lane: Optional[Lane] = self.geometry_mgr.find_assigned_lane(gx, gy)
        lane_name = lane.name if lane else "Carriageway"
        lane_id = lane.lane_id if lane else 1

        # -------------------------------------------------------------
        # Indian Road Rule A: CATTLE / ANIMAL ON CARRIAGEWAY (Critical)
        # -------------------------------------------------------------
        if vehicle.entity_type == "animal" and zone in ["CARRIAGEWAY", "SHOULDER"]:
            animal_cfg = self.rules_cfg.get("animal_hazard", {})
            if animal_cfg.get("enabled", True):
                state["animal_hazard_frames"] = state.get("animal_hazard_frames", 0) + 1
                persist_needed = int(animal_cfg.get("persistence_frames", 8))
                if state["animal_hazard_frames"] >= persist_needed:
                    alert = AlertEvent(
                        event_type="CATTLE_ANIMAL_ON_CARRIAGEWAY",
                        severity="SEVERE",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Live Road Hazard: {vehicle.class_name} #{tid} on {lane_name}",
                        timestamp=timestamp,
                        confidence=0.94,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "lane_id": lane_id
                        }
                    )
                    alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # Indian Road Rule B: STREET VENDOR / PUSHCART ENCROACHMENT (Warning)
        # -------------------------------------------------------------
        if vehicle.entity_type == "pushcart_vendor" and zone in ["CARRIAGEWAY", "SHOULDER"]:
            vendor_cfg = self.rules_cfg.get("vendor_encroachment", {})
            if vendor_cfg.get("enabled", True):
                if vehicle.speed_kmh < float(vendor_cfg.get("speed_threshold_kmh", 4.0)):
                    state["vendor_dwell_frames"] = state.get("vendor_dwell_frames", 0) + 1
                    if state["vendor_dwell_frames"] >= 15:
                        alert = AlertEvent(
                            event_type="STREET_VENDOR_ENCROACHMENT",
                            severity="MODERATE",
                            track_id=tid,
                            class_name=vehicle.class_name,
                            message=f"Road Encroachment: Vendor/Cart #{tid} obstructing {lane_name}",
                            timestamp=timestamp,
                            confidence=0.88,
                            metrics={
                                "speed_kmh": vehicle.speed_kmh,
                                "lane_id": lane_id
                            }
                        )
                        alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # Indian Road Rule C: PEDESTRIAN IN LIVE TRAVEL LANE (Warning)
        # -------------------------------------------------------------
        if vehicle.entity_type == "pedestrian_vendor" and zone == "CARRIAGEWAY":
            ped_cfg = self.rules_cfg.get("pedestrian_hazard", {})
            if ped_cfg.get("enabled", True):
                state["ped_hazard_frames"] = state.get("ped_hazard_frames", 0) + 1
                if state["ped_hazard_frames"] >= int(ped_cfg.get("persistence_frames", 10)):
                    alert = AlertEvent(
                        event_type="PEDESTRIAN_IN_TRAVEL_LANE",
                        severity="MODERATE",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Vulnerable Pedestrian #{tid} in live {lane_name}",
                        timestamp=timestamp,
                        confidence=0.91,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "lane_id": lane_id
                        }
                    )
                    alerts.append(alert)
            return alerts

        # -------------------------------------------------------------
        # CORE WAVERING & ERRATIC BEHAVIOR DETECTION (High Precision)
        # -------------------------------------------------------------
        if vehicle.entity_type == "vehicle" and not is_warmup:
            assessment: WaveringAssessment = self.wavering_detector.analyze_trajectory(
                history_ground=vehicle.history_ground,
                class_name=vehicle.class_name,
                current_speed_kmh=vehicle.speed_kmh,
                zone=zone,
                is_agricultural=is_agri
            )

            if assessment.is_wavering:
                state["wavering_frames"] = state.get("wavering_frames", 0) + 1
                # Require 3 consecutive confirming frames to eliminate single-frame detector jitter
                if state["wavering_frames"] >= 3:
                    alert = AlertEvent(
                        event_type="WAVERING_VEHICLE",
                        severity=assessment.severity,
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"{assessment.severity} Wavering: {assessment.reason} by {vehicle.class_name} #{tid}",
                        timestamp=timestamp,
                        confidence=assessment.confidence,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "amplitude_m": assessment.metric_amplitude_m,
                            "oscillations": assessment.oscillation_count,
                            "peak_lateral_accel_m_s2": assessment.peak_lateral_accel_m_s2,
                            "heading_variance_deg": assessment.heading_variance_deg,
                            "speed_cv": assessment.speed_cv,
                            "reason": assessment.reason
                        }
                    )
                    alerts.append(alert)
            else:
                state["wavering_frames"] = max(0, state.get("wavering_frames", 0) - 1)

        # -------------------------------------------------------------
        # CALIBRATED WRONG_WAY_TRAVEL (Bidirectional Corridor Aware)
        # -------------------------------------------------------------
        ww_cfg = self.rules_cfg.get("wrong_way", {})
        if ww_cfg.get("enabled", True) and not is_warmup and zone == "CARRIAGEWAY":
            # Determine expected corridor flow direction dynamically from road context
            expected_flow = self.road_context.get_expected_flow_direction(gx)
            
            # Vehicle must have clear velocity opposing the corridor flow (> 1.8 m/s, ~6.5 km/h)
            if abs(vy) > 1.8 and np.sign(vy) != np.sign(expected_flow):
                state["wrong_way_frames"] += 1
                if state["wrong_way_start_s"] is None:
                    state["wrong_way_start_s"] = gy

                delta_dist = abs(gy - state["wrong_way_start_s"])
                # Require >= 12 frames persistence AND at least 4.0 meters of wrong-way movement
                if state["wrong_way_frames"] >= 12 and delta_dist >= 4.0:
                    alert = AlertEvent(
                        event_type="WRONG_WAY_TRAVEL",
                        severity="SEVERE",
                        track_id=tid,
                        class_name=vehicle.class_name,
                        message=f"Wrong-way {vehicle.class_name} #{tid} travelling opposite to corridor flow ({vehicle.speed_kmh:.0f} km/h)",
                        timestamp=timestamp,
                        confidence=0.92,
                        metrics={
                            "speed_kmh": vehicle.speed_kmh,
                            "corridor_flow": expected_flow,
                            "wrong_way_dist_m": delta_dist,
                            "lane_id": lane_id
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
        # CALIBRATED STOPPED_IN_TRAVEL_LANE
        # -------------------------------------------------------------
        stopped_cfg = self.rules_cfg.get("stopped_vehicle", {})
        if stopped_cfg.get("enabled", True) and zone == "CARRIAGEWAY":
            min_stopped_s = float(stopped_cfg.get("min_stopped_seconds", 5.0))
            if vehicle.stopped_duration_s >= min_stopped_s:
                alert = AlertEvent(
                    event_type="STOPPED_IN_TRAVEL_LANE",
                    severity="SEVERE" if vehicle.stopped_duration_s >= 10.0 else "MODERATE",
                    track_id=tid,
                    class_name=vehicle.class_name,
                    message=f"Stationary {vehicle.class_name} #{tid} obstructing {lane_name} for {vehicle.stopped_duration_s:.1f}s",
                    timestamp=timestamp,
                    confidence=0.96,
                    metrics={
                        "speed_kmh": vehicle.speed_kmh,
                        "stopped_duration_s": vehicle.stopped_duration_s,
                        "lane_id": lane_id
                    }
                )
                alerts.append(alert)
                state["active_alerts"].add("STOPPED_IN_TRAVEL_LANE")
            else:
                state["active_alerts"].discard("STOPPED_IN_TRAVEL_LANE")

        return alerts
