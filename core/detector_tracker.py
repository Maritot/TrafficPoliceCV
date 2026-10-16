"""
Stage 2 & 3: Vehicle Detection & Multi-Object Tracking Engine.
Runs deep object detection with ByteTrack persistent association,
extracts tire-road ground contact points, and projects tracks into metric space.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from core.calibration import HomographyCalibrator


class TrackedVehicle:
    def __init__(self, track_id: int, class_id: int, class_name: str, 
                 entity_type: str, bbox: List[float], contact_img: Tuple[float, float], 
                 contact_ground: Tuple[float, float], timestamp: float):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.entity_type = entity_type  # "vehicle", "animal", "pushcart_vendor", "pedestrian_vendor"
        self.bbox = bbox  # [x1, y1, x2, y2]
        
        # Current contact points
        self.contact_img = contact_img          # (u, v) in pixels
        self.contact_ground = contact_ground    # (X, Y) in meters
        
        # Historical state buffers (last 30 samples)
        self.history_img: List[Tuple[int, int]] = [(int(contact_img[0]), int(contact_img[1]))]
        self.history_ground: List[Tuple[float, float, float]] = [(contact_ground[0], contact_ground[1], timestamp)]
        
        # Velocity and dynamics (metric)
        self.vx_m_s = 0.0
        self.vy_m_s = 0.0
        self.ax_m_s2 = 0.0  # Lateral acceleration (for swerve detection)
        self.speed_kmh = 0.0
        self.stopped_duration_s = 0.0
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp

    def update(self, bbox: List[float], contact_img: Tuple[float, float], 
               contact_ground: Tuple[float, float], timestamp: float):
        dt = timestamp - self.last_seen_time
        self.bbox = bbox
        self.last_seen_time = timestamp

        # Apply Exponential Moving Average (EMA) smoothing to contact points to suppress detection jitter
        if len(self.history_ground) > 0:
            last_gx, last_gy, _ = self.history_ground[-1]
            last_ix, last_iy = self.history_img[-1]
            
            # 70% current, 30% previous
            smooth_gx = 0.70 * contact_ground[0] + 0.30 * last_gx
            smooth_gy = 0.70 * contact_ground[1] + 0.30 * last_gy
            smooth_ix = int(0.70 * contact_img[0] + 0.30 * last_ix)
            smooth_iy = int(0.70 * contact_img[1] + 0.30 * last_iy)
            
            contact_ground = (smooth_gx, smooth_gy)
            contact_img = (float(smooth_ix), float(smooth_iy))

        self.contact_img = contact_img
        self.contact_ground = contact_ground

        self.history_img.append((int(contact_img[0]), int(contact_img[1])))
        if len(self.history_img) > 30:
            self.history_img.pop(0)

        self.history_ground.append((contact_ground[0], contact_ground[1], timestamp))
        if len(self.history_ground) > 30:
            self.history_ground.pop(0)

        # Estimate smoothed metric velocity over last 4-6 frames
        if len(self.history_ground) >= 3:
            prev_x, prev_y, prev_t = self.history_ground[-3]
            total_dt = timestamp - prev_t
            if total_dt > 0.02:
                raw_vx = (contact_ground[0] - prev_x) / total_dt
                raw_vy = (contact_ground[1] - prev_y) / total_dt

                if len(self.history_ground) <= 5:
                    # Warmup initialization to avoid derivative spike
                    self.vx_m_s = raw_vx
                    self.vy_m_s = raw_vy
                    self.ax_m_s2 = 0.0
                else:
                    prev_vx = self.vx_m_s
                    # Low-pass filter for smooth velocity estimation
                    alpha = 0.35
                    self.vx_m_s = alpha * raw_vx + (1 - alpha) * self.vx_m_s
                    self.vy_m_s = alpha * raw_vy + (1 - alpha) * self.vy_m_s
                    self.ax_m_s2 = (self.vx_m_s - prev_vx) / max(total_dt, 1e-3)
                
                speed_m_s = np.hypot(self.vx_m_s, self.vy_m_s)
                self.speed_kmh = float(speed_m_s * 3.6)

                # Update stopped duration counter
                if self.speed_kmh < 2.5:
                    self.stopped_duration_s += dt
                else:
                    self.stopped_duration_s = max(0.0, self.stopped_duration_s - dt * 2)


class VehicleDetectionTracker:
    def __init__(self, model_weights: str, calibrator: HomographyCalibrator,
                 target_classes: List[int], conf_thresh: float = 0.35, 
                 tracker_config: str = "bytetrack.yaml",
                 class_type_map: Optional[Dict] = None):
        """
        Initializes the YOLO + ByteTrack detector for vehicles, cattle, and road hazards.
        """
        self.model = YOLO(model_weights)
        self.calibrator = calibrator
        self.target_classes = target_classes
        self.conf_thresh = conf_thresh
        self.tracker_config = tracker_config
        
        # Default mapping covering standard COCO and Indian hazard classes
        self.class_type_map = class_type_map or {
            0: {"name": "Pedestrian/Vendor", "type": "pedestrian_vendor"},
            1: {"name": "Bicycle/Pushcart", "type": "pushcart_vendor"},
            2: {"name": "Car", "type": "vehicle"},
            3: {"name": "Motorcycle", "type": "vehicle"},
            5: {"name": "Bus", "type": "vehicle"},
            7: {"name": "Truck", "type": "vehicle"},
            16: {"name": "Stray Dog", "type": "animal"},
            17: {"name": "Horse", "type": "animal"},
            18: {"name": "Sheep/Goat", "type": "animal"},
            19: {"name": "Cow/Cattle", "type": "animal"}
        }

        # Active tracked entities registry: stable_id -> TrackedVehicle
        self.tracks: Dict[int, TrackedVehicle] = {}
        # Mapping from raw YOLO tracker ID to stable LATIS ID
        self.raw_to_stable_id: Dict[int, int] = {}
        # Recently lost tracks eligible for re-attachment: stable_id -> TrackedVehicle
        self.lost_tracks: Dict[int, TrackedVehicle] = {}

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[TrackedVehicle]:
        """
        Runs object detection and tracking on a single frame.
        Applies ground-plane track stitching to prevent ID-switching on fast vehicles.
        """
        # Run YOLO with ByteTrack persistence
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=self.target_classes,
            conf=self.conf_thresh,
            verbose=False
        )

        active_stable_ids = set()
        active_vehicles: List[TrackedVehicle] = []
        claimed_stable_ids = set()

        # Step 1: Collect valid detections in the current frame
        frame_detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id is None:
                    continue

                raw_id = int(boxes.id[i].item())
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                # Tire-Road / Foot Ground Contact Point: Bottom center of bounding box
                u_contact = (xyxy[0] + xyxy[2]) / 2.0
                v_contact = xyxy[3]
                contact_img = (float(u_contact), float(v_contact))

                # Project contact point into metric ground plane (meters)
                ground_pt = self.calibrator.project_image_to_ground([[u_contact, v_contact]])[0]
                contact_ground = (float(ground_pt[0]), float(ground_pt[1]))

                meta = self.class_type_map.get(class_id, {"name": f"Class_{class_id}", "type": "vehicle"})
                class_name = meta["name"]
                entity_type = meta["type"]

                frame_detections.append({
                    "raw_id": raw_id,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": xyxy,
                    "contact_img": contact_img,
                    "contact_ground": contact_ground,
                    "class_name": class_name,
                    "entity_type": entity_type
                })

        # Step 2: Associate detections to persistent stable IDs with strict 1-to-1 matching
        for det in frame_detections:
            raw_id = det["raw_id"]
            class_id = det["class_id"]
            contact_ground = det["contact_ground"]
            contact_img = det["contact_img"]
            xyxy = det["bbox"]
            class_name = det["class_name"]
            entity_type = det["entity_type"]

            stable_id = None

            # Case A: Existing raw_id mapping
            if raw_id in self.raw_to_stable_id:
                candidate_id = self.raw_to_stable_id[raw_id]
                # Validate that candidate_id has NOT already been claimed in this frame
                if candidate_id not in claimed_stable_ids:
                    # Validate physical continuity if candidate is active in tracks
                    if candidate_id in self.tracks:
                        prev_gx = self.tracks[candidate_id].contact_ground[0]
                        # Reject if lateral jump exceeds 1.2m (adjacent lane jump)
                        if abs(contact_ground[0] - prev_gx) <= 1.2:
                            stable_id = candidate_id
                    else:
                        stable_id = candidate_id

            # Case B: Unassigned or invalidated - attempt to match a recently LOST track
            if stable_id is None:
                best_match_id = None
                min_dist = float("inf")

                # Match STRICTLY against recently lost tracks (never against active tracks)
                for tid, candidate in list(self.lost_tracks.items()):
                    if tid in claimed_stable_ids or candidate.class_id != class_id:
                        continue

                    dt = timestamp - candidate.last_seen_time
                    if dt < 0.04 or dt > 2.0:
                        continue

                    # Predict expected position along road
                    pred_x = candidate.contact_ground[0] + candidate.vx_m_s * dt
                    pred_y = candidate.contact_ground[1] + candidate.vy_m_s * dt

                    # Lateral constraint: Vehicle must be within 0.85m laterally (stay in lane)
                    if abs(contact_ground[0] - pred_x) > 0.85:
                        continue

                    # Longitudinal constraint: Vehicle must be within 4.0m longitudinally
                    if abs(contact_ground[1] - pred_y) > 4.0:
                        continue

                    dist = np.hypot(contact_ground[0] - pred_x, contact_ground[1] - pred_y)
                    if dist < 2.5 and dist < min_dist:
                        min_dist = dist
                        best_match_id = tid

                if best_match_id is not None:
                    stable_id = best_match_id
                    # Recover candidate from lost buffer
                    self.tracks[stable_id] = self.lost_tracks.pop(stable_id)
                else:
                    # Fallback: assign raw_id if unclaimed, else pick fresh unique ID
                    if raw_id not in claimed_stable_ids and raw_id not in self.tracks and raw_id not in self.lost_tracks:
                        stable_id = raw_id
                    else:
                        all_known = set(self.tracks.keys()).union(self.lost_tracks.keys()).union(claimed_stable_ids)
                        stable_id = max(all_known.union({raw_id})) + 1

            # Update mapping and claim ID
            claimed_stable_ids.add(stable_id)
            self.raw_to_stable_id[raw_id] = stable_id

            # Update or create track
            if stable_id in self.tracks:
                vehicle = self.tracks[stable_id]
                vehicle.update(xyxy, contact_img, contact_ground, timestamp)
            else:
                vehicle = TrackedVehicle(
                    track_id=stable_id,
                    class_id=class_id,
                    class_name=class_name,
                    entity_type=entity_type,
                    bbox=xyxy,
                    contact_img=contact_img,
                    contact_ground=contact_ground,
                    timestamp=timestamp
                )
                self.tracks[stable_id] = vehicle

            active_stable_ids.add(stable_id)
            active_vehicles.append(self.tracks[stable_id])

        # Move inactive tracks to lost_tracks buffer
        for tid in list(self.tracks.keys()):
            if tid not in active_stable_ids:
                if timestamp - self.tracks[tid].last_seen_time > 0.10:
                    self.lost_tracks[tid] = self.tracks.pop(tid)

        # Purge stale tracks not seen in over 2.5 seconds
        for tid in list(self.lost_tracks.keys()):
            if timestamp - self.lost_tracks[tid].last_seen_time > 2.5:
                del self.lost_tracks[tid]

        # Clean up stale raw-to-stable mappings
        active_and_lost = set(self.tracks.keys()).union(self.lost_tracks.keys())
        stale_raw = [r for r, s in self.raw_to_stable_id.items() if s not in active_and_lost]
        for r in stale_raw:
            del self.raw_to_stable_id[r]

        return active_vehicles
