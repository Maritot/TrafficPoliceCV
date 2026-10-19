"""
Project LATIS - Headless Computer Vision REST & Video Streaming API.
Provides pure JSON endpoints and MJPEG video streaming for Next.js frontend.
Contains ZERO HTML templates.
"""

import os
import sys
import time
import json
import threading
import hashlib
import yaml
import cv2
import numpy as np
import imageio_ffmpeg
from flask import Flask, Response, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.calibration import HomographyCalibrator
from core.geometry import RoadGeometryManager
from core.detector_tracker import VehicleDetectionTracker
from core.behavior_engine import BehaviorEngine, AlertEvent
from core.road_context import RoadContextManager
from core.visualizer import LATISVisualizer
from backend.database import insert_event, list_events, update_event_feedback, get_event_stats

app = Flask(__name__)

# Enable CORS for Next.js (port 3000)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response


# -------------------------------------------------------------
# AREA-WISE CCTV CAMERA DIRECTORY & REGISTRY
# -------------------------------------------------------------
CAMERA_REGISTRY = {
    "CAM-01": {
        "id": "CAM-01",
        "name": "NH-48 Km 32 Toll Plaza Express",
        "area": "Zone 1 - North Highway Corridor",
        "area_code": "ZONE_NORTH",
        "location": "NH-48 Toll Approach (Full HD 50FPS)",
        "video_src": os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4") if os.path.exists(os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4")) else os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1920x1080 Full HD",
        "corridor_type": "dual_carriageway",
        "status": "ONLINE"
    },
    "CAM-02": {
        "id": "CAM-02",
        "name": "NH-48 Sector 21 High-Speed Median",
        "area": "Zone 1 - North Highway Corridor",
        "area_code": "ZONE_NORTH",
        "location": "NH-48 Sec 21 Express Flyover",
        "video_src": os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1280x720 HD",
        "corridor_type": "highway",
        "status": "ONLINE"
    },
    "CAM-03": {
        "id": "CAM-03",
        "name": "Western Peripheral Flyover Split",
        "area": "Zone 1 - North Highway Corridor",
        "area_code": "ZONE_NORTH",
        "location": "WPE Km 14 Interchange Split",
        "video_src": os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1280x720 HD",
        "corridor_type": "highway",
        "status": "ONLINE"
    },
    "CAM-04": {
        "id": "CAM-04",
        "name": "Central Avenue Commercial Arterial",
        "area": "Zone 2 - Central Urban District",
        "area_code": "ZONE_CENTRAL",
        "location": "MG Road / Commercial Hub",
        "video_src": os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4") if os.path.exists(os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4")) else os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1920x1080 Full HD",
        "corridor_type": "dual_carriageway",
        "status": "ONLINE"
    },
    "CAM-05": {
        "id": "CAM-05",
        "name": "Inner Ring Road Transit Corridor",
        "area": "Zone 2 - Central Urban District",
        "area_code": "ZONE_CENTRAL",
        "location": "Ring Road BRT Lane",
        "video_src": os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1280x720 HD",
        "corridor_type": "highway",
        "status": "ONLINE"
    },
    "CAM-06": {
        "id": "CAM-06",
        "name": "Suburban Feeder - Farmland Perimeter",
        "area": "Zone 3 - Suburban & Rural Belt",
        "area_code": "ZONE_RURAL",
        "location": "SH-12 Agricultural Feeder Road",
        "video_src": os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4") if os.path.exists(os.path.join(PROJECT_ROOT, "uploads", "12785235-hd_1920_1080_50fps.mp4")) else os.path.join(PROJECT_ROOT, "video1.avi"),
        "resolution": "1920x1080 Full HD",
        "corridor_type": "dual_carriageway",
        "status": "ONLINE"
    }
}


# Global Telemetry & Streaming State
STATE = {
    "active_camera_id": "CAM-01",
    "mode": "highway_hd",
    "paused": False,
    "show_radar": True,
    "reset_requested": False,
    "fps": 0.0,
    "active_tracks": 0,
    "avg_speed": 0,
    "total_alerts": 0,
    "recent_alerts": [],
    "start_time": time.time()
}

_PIPELINE_LOCK = threading.Lock()


def load_pipeline(camera_id: str = "CAM-01", show_radar: bool = True):
    """Initializes the high-precision WAV-DET perception & tracking engine for the selected area camera."""
    cam_info = CAMERA_REGISTRY.get(camera_id, CAMERA_REGISTRY["CAM-01"])
    video_src = cam_info["video_src"]
    corridor_type = cam_info.get("corridor_type", "highway")

    if not os.path.exists(video_src):
        video_src = os.path.join(PROJECT_ROOT, "video1.avi")

    target_w, target_h = 1280, 720

    # Dynamic Corridor Geometry & Calibration
    if corridor_type == "dual_carriageway" or "12785235" in video_src:
        scaled_quad = [
            [int(0.05 * target_w), int(0.95 * target_h)],
            [int(0.95 * target_w), int(0.95 * target_h)],
            [int(0.62 * target_w), int(0.38 * target_h)],
            [int(0.22 * target_w), int(0.38 * target_h)]
        ]
        ground_quad = [
            [0.0, 0.0],
            [16.0, 0.0],
            [16.0, 120.0],
            [0.0, 120.0]
        ]
        lanes_cfg = [
            {"id": 1, "name": "Lane 1 (Left Carriageway)", "left_boundary_x": 0.0, "right_boundary_x": 4.0, "centerline_x": 2.0, "direction_y": -1.0, "speed_limit_kmh": 90.0},
            {"id": 2, "name": "Lane 2 (Left Carriageway)", "left_boundary_x": 4.0, "right_boundary_x": 8.0, "centerline_x": 6.0, "direction_y": -1.0, "speed_limit_kmh": 90.0},
            {"id": 3, "name": "Lane 3 (Right Carriageway)", "left_boundary_x": 8.0, "right_boundary_x": 12.0, "centerline_x": 10.0, "direction_y": 1.0, "speed_limit_kmh": 90.0},
            {"id": 4, "name": "Lane 4 (Right Carriageway)", "left_boundary_x": 12.0, "right_boundary_x": 16.0, "centerline_x": 14.0, "direction_y": 1.0, "speed_limit_kmh": 90.0}
        ]
        calibrator = HomographyCalibrator(scaled_quad, ground_quad)
        geometry_mgr = RoadGeometryManager(lanes_cfg, calibrator)
        road_context = RoadContextManager(calibrator, road_width_m=16.0, road_depth_m=120.0)
        rules_cfg = {}
        config = {
            "visualization": {
                "show_bev_radar": show_radar,
                "radar_range_x_m": [-2, 18],
                "radar_range_y_m": [0, 120],
                "radar_width_px": 240,
                "radar_height_px": 300,
                "radar_pos": "top-right"
            },
            "lanes": lanes_cfg
        }
    elif corridor_type == "indian":
        cfg_path = os.path.join(PROJECT_ROOT, "config", "indian_road_config.yaml")
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        config["visualization"]["show_bev_radar"] = show_radar
        calib_cfg = config["calibration"]
        scale_x = target_w / 1280.0
        scale_y = target_h / 720.0
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        calibrator = HomographyCalibrator(scaled_quad, calib_cfg["ground_quad"])
        geometry_mgr = RoadGeometryManager(config["lanes"], calibrator)
        road_context = RoadContextManager(calibrator, road_width_m=14.0, road_depth_m=80.0)
        rules_cfg = config.get("rules", {})
    else:
        cfg_path = os.path.join(PROJECT_ROOT, "config", "highway_config.yaml")
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        config["visualization"]["show_bev_radar"] = show_radar
        calib_cfg = config["calibration"]
        scale_x = target_w / 320.0
        scale_y = target_h / 240.0
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        calibrator = HomographyCalibrator(scaled_quad, calib_cfg["ground_quad"])
        geometry_mgr = RoadGeometryManager(config["lanes"], calibrator)
        road_context = RoadContextManager(calibrator, road_width_m=12.0, road_depth_m=100.0)
        rules_cfg = config.get("rules", {})

    cap = cv2.VideoCapture(video_src)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # High-Accuracy Perception: YOLOv8s with imgsz=1280 and conf=0.20
    model_file = "yolov8s.pt" if os.path.exists(os.path.join(PROJECT_ROOT, "yolov8s.pt")) else "yolov8n.pt"
    class_type_map = {
        0: {"name": "Pedestrian", "type": "pedestrian_vendor"},
        1: {"name": "Bicycle/Cart", "type": "pushcart_vendor"},
        2: {"name": "Car", "type": "vehicle"},
        3: {"name": "Motorcycle", "type": "vehicle"},
        5: {"name": "Bus", "type": "vehicle"},
        7: {"name": "Truck", "type": "vehicle"},
        16: {"name": "Stray Dog", "type": "animal"},
        17: {"name": "Animal Cart", "type": "animal"},
        19: {"name": "Cow/Cattle", "type": "animal"}
    }
    detector = VehicleDetectionTracker(
        model_weights=model_file,
        calibrator=calibrator,
        target_classes=[0, 1, 2, 3, 5, 7, 16, 17, 19],
        conf_thresh=0.20,
        tracker_config="bytetrack.yaml",
        class_type_map=class_type_map,
        imgsz=1280
    )
    behavior_engine = BehaviorEngine(geometry_mgr, rules_cfg, road_context)
    visualizer = LATISVisualizer(geometry_mgr, config)

    return cap, calibrator, geometry_mgr, detector, behavior_engine, visualizer, target_w, target_h, fps_in


def generate_mjpeg_stream():
    """Continuous MJPEG frame generator streaming processed video with computer vision overlays."""
    global STATE

    while True:
        cam_id = STATE.get("active_camera_id", "CAM-01")
        show_radar = STATE["show_radar"]
        try:
            (cap, calibrator, geometry_mgr, detector, 
             behavior_engine, visualizer, 
             target_w, target_h, fps_in) = load_pipeline(cam_id, show_radar)
        except Exception as e:
            print(f"[-] Pipeline load error for camera '{cam_id}': {e}")
            time.sleep(1)
            continue

        frame_idx = 0
        t_start = time.time()
        t_prev = t_start
        STATE["reset_requested"] = False
        live_persisted: dict = {}

        while cap.isOpened():
            if STATE["reset_requested"] or STATE.get("active_camera_id") != cam_id or STATE["show_radar"] != show_radar:
                break

            if STATE["paused"]:
                time.sleep(0.04)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            current_fps = 1.0 / max(dt, 1e-4)
            timestamp = frame_idx / fps_in

            # Standardize resolution to (target_w, target_h)
            h, w = frame.shape[:2]
            if (w, h) != (target_w, target_h):
                frame_std = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4 if w > target_w else cv2.INTER_CUBIC)
            else:
                frame_std = frame

            try:
                vehicles = detector.process_frame(frame_std, timestamp)
                frame_alerts = []
                for v in vehicles:
                    v_alerts = behavior_engine.analyze_vehicle(v, timestamp)
                    frame_alerts.extend(v_alerts)

                annotated_frame = visualizer.draw(frame_std, vehicles, frame_alerts, timestamp, current_fps)

                speeds = [v.speed_kmh for v in vehicles if v.speed_kmh > 0]
                avg_speed = int(np.mean(speeds)) if speeds else 0

                STATE["fps"] = current_fps
                STATE["active_tracks"] = len(vehicles)
                STATE["avg_speed"] = avg_speed

                if frame_alerts:
                    for a in frame_alerts:
                        STATE["total_alerts"] += 1
                        inc_id = f"INC-{a.track_id}-{int(timestamp * 10)}"
                        throttle_key = f"{a.track_id}_{a.event_type}"
                        ev_dir = os.path.join(PROJECT_ROOT, "uploads", "evidence")
                        os.makedirs(ev_dir, exist_ok=True)
                        ev_filename = f"{inc_id}.jpg"
                        ev_url = f"/api/media/uploads/evidence/{ev_filename}"

                        if timestamp - live_persisted.get(throttle_key, -99.0) > 4.0:
                            live_persisted[throttle_key] = timestamp
                            cv2.imwrite(os.path.join(ev_dir, ev_filename), annotated_frame)
                            insert_event({
                                "id": inc_id,
                                "video_id": f"cctv_{cam_id}",
                                "track_id": a.track_id,
                                "class_name": a.class_name,
                                "event_type": a.event_type,
                                "severity": a.severity,
                                "reason": a.message,
                                "confidence": getattr(a, "confidence", 0.90),
                                "timestamp_s": timestamp,
                                "speed_kmh": a.metrics.get("speed_kmh", 0.0),
                                "lane_id": a.metrics.get("lane_id", 1),
                                "evidence_frame_url": ev_url
                            })

                        alert_item = {
                            "id": inc_id,
                            "track_id": a.track_id,
                            "class_name": a.class_name,
                            "event_type": a.event_type,
                            "severity": a.severity,
                            "message": a.message,
                            "confidence": getattr(a, "confidence", 0.90),
                            "timestamp": round(timestamp, 2),
                            "speed_kmh": round(a.metrics.get("speed_kmh", 0.0), 1),
                            "lane_id": a.metrics.get("lane_id", 1),
                            "evidence_frame_url": ev_url
                        }
                        STATE["recent_alerts"].insert(0, alert_item)
                        if len(STATE["recent_alerts"]) > 60:
                            STATE["recent_alerts"].pop()

                ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as ex:
                print(f"[-] Pipeline exception: {ex}")
                continue

            time.sleep(max(0.008, (1.0 / fps_in) - 0.012))

        cap.release()


# -------------------------------------------------------------
# PURE REST API ENDPOINTS (ZERO HTML)
# -------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Healthcheck endpoint for monitoring container/process state."""
    cam_id = STATE.get("active_camera_id", "CAM-01")
    cam_info = CAMERA_REGISTRY.get(cam_id, CAMERA_REGISTRY["CAM-01"])
    return jsonify({
        "status": "healthy",
        "service": "LATIS Headless CV Engine",
        "uptime_s": round(time.time() - STATE["start_time"], 1),
        "active_camera_id": cam_id,
        "camera_name": cam_info["name"],
        "area": cam_info["area"],
        "active_tracks": STATE["active_tracks"]
    })


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Returns area-wise registered CCTV cameras with metadata and current state."""
    cams = []
    active_cam = STATE.get("active_camera_id", "CAM-01")
    for cid, info in CAMERA_REGISTRY.items():
        cams.append({
            "id": cid,
            "name": info["name"],
            "area": info["area"],
            "area_code": info["area_code"],
            "location": info["location"],
            "resolution": info["resolution"],
            "corridor_type": info["corridor_type"],
            "status": info["status"],
            "is_active": (cid == active_cam),
            "active_tracks": STATE["active_tracks"] if cid == active_cam else 0,
            "avg_speed_kmh": STATE["avg_speed"] if cid == active_cam else 0
        })
    return jsonify(cams)


@app.route('/api/camera_snapshot', methods=['GET'])
def camera_snapshot():
    """Returns an on-demand real-time JPEG snapshot from any registered camera."""
    cam_id = request.args.get('camera_id', 'CAM-01')
    cam_info = CAMERA_REGISTRY.get(cam_id, CAMERA_REGISTRY['CAM-01'])
    video_src = cam_info['video_src']
    if not os.path.exists(video_src):
        return jsonify({"error": "Video source not found"}), 404

    cap = cv2.VideoCapture(video_src)
    frame_no = int(request.args.get('frame', 35))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Failed to capture snapshot"}), 500

    h, w = frame.shape[:2]
    if (w, h) != (1280, 720):
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4 if w > 1280 else cv2.INTER_CUBIC)

    # Clean official HUD header banner on snapshot
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 50), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.circle(frame, (25, 25), 6, (0, 220, 100), -1)
    cv2.putText(frame, f"NODE: {cam_id}  |  {cam_info['name'].upper()}  |  {cam_info['area'].upper()}", 
                (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"STATUS: LIVE ENROUTE  •  {cam_info['resolution']}", 
                (950, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/select_camera', methods=['POST'])
def select_camera():
    """Switches active camera feed across geographic areas."""
    data = request.get_json() or {}
    cam_id = data.get("camera_id")
    if cam_id in CAMERA_REGISTRY:
        STATE["active_camera_id"] = cam_id
        STATE["reset_requested"] = True
        STATE["total_alerts"] = 0
        STATE["recent_alerts"] = []
        return jsonify({"status": "ok", "active_camera_id": cam_id, "camera": CAMERA_REGISTRY[cam_id]})
    return jsonify({"error": f"Camera '{cam_id}' not found in registry"}), 404


@app.route('/api/stream', methods=['GET'])
def stream():
    """Streams live MJPEG frames with real-time CV annotations."""
    cam_id = request.args.get('camera_id')
    if cam_id in CAMERA_REGISTRY and cam_id != STATE.get("active_camera_id"):
        STATE["active_camera_id"] = cam_id
        STATE["reset_requested"] = True

    # Backward compatibility with legacy mode parameter
    req_mode = request.args.get('mode')
    if req_mode == "indian" and STATE.get("active_camera_id") != "CAM-04":
        STATE["active_camera_id"] = "CAM-04"
        STATE["reset_requested"] = True
    elif req_mode == "highway" and STATE.get("active_camera_id") not in ["CAM-01", "CAM-02"]:
        STATE["active_camera_id"] = "CAM-01"
        STATE["reset_requested"] = True

    radar = request.args.get('radar')
    if radar is not None:
        STATE["show_radar"] = (radar.lower() in ["1", "true", "yes"])

    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/telemetry', methods=['GET'])
def telemetry():
    """Returns real-time vehicle kinematics, pipeline speed, and active camera metadata."""
    cam_id = STATE.get("active_camera_id", "CAM-01")
    cam_info = CAMERA_REGISTRY.get(cam_id, CAMERA_REGISTRY["CAM-01"])
    return jsonify({
        "camera_id": cam_id,
        "camera_name": cam_info["name"],
        "area": cam_info["area"],
        "area_code": cam_info["area_code"],
        "location": cam_info["location"],
        "resolution": cam_info["resolution"],
        "corridor_type": cam_info["corridor_type"],
        "mode": STATE["mode"],
        "paused": STATE["paused"],
        "fps": round(STATE["fps"], 1),
        "active_tracks": STATE["active_tracks"],
        "avg_speed_kmh": STATE["avg_speed"],
        "total_alerts": STATE["total_alerts"],
        "show_radar": STATE["show_radar"]
    })


@app.route('/api/incidents', methods=['GET'])
def incidents():
    """Returns the latest safety violations and road hazards."""
    limit = int(request.args.get('limit', 20))
    return jsonify(STATE["recent_alerts"][:limit])


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Legacy mode switcher."""
    data = request.get_json() or {}
    new_mode = data.get("mode")
    if new_mode == "indian":
        STATE["active_camera_id"] = "CAM-04"
        STATE["reset_requested"] = True
        return jsonify({"status": "ok", "mode": new_mode})
    elif new_mode == "highway":
        STATE["active_camera_id"] = "CAM-01"
        STATE["reset_requested"] = True
        return jsonify({"status": "ok", "mode": new_mode})
    return jsonify({"error": "Invalid mode"}), 400


@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    """Pauses or resumes video streaming."""
    STATE["paused"] = not STATE["paused"]
    return jsonify({"paused": STATE["paused"]})


@app.route('/api/toggle_radar', methods=['POST'])
def toggle_radar():
    """Toggles top-down 2D BEV radar rendering."""
    STATE["show_radar"] = not STATE["show_radar"]
    STATE["reset_requested"] = True
    return jsonify({"show_radar": STATE["show_radar"]})



@app.route('/api/events', methods=['GET'])
def get_events():
    """Returns persistent incident records with filtering and pagination."""
    severity = request.args.get('severity')
    feedback = request.args.get('feedback')
    video_id = request.args.get('video_id')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    events = list_events(severity=severity, feedback=feedback, video_id=video_id, limit=limit, offset=offset)
    return jsonify(events)


@app.route('/api/events/<event_id>/feedback', methods=['POST'])
def post_event_feedback(event_id):
    """Submits human verification (Confirmed TP vs False Alarm) for active learning."""
    data = request.get_json() or {}
    feedback = data.get('feedback')
    notes = data.get('notes', '')
    if feedback not in ['confirmed_tp', 'false_alarm', 'unreviewed']:
        return jsonify({"error": "Invalid feedback status. Must be 'confirmed_tp', 'false_alarm', or 'unreviewed'"}), 400
    success = update_event_feedback(event_id, feedback, notes)
    if success:
        return jsonify({"status": "success", "event_id": event_id, "feedback": feedback})
    return jsonify({"error": "Event not found"}), 404


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns system performance metrics, severity counts, and active learning statistics."""
    stats = get_event_stats()
    eval_path = os.path.join(PROJECT_ROOT, "training", "evaluation_report.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                stats["model_eval"] = json.load(f)
        except Exception:
            pass
    return jsonify(stats)


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Triggers ML classifier retraining using active learning feedback."""
    import subprocess
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "training", "train_wavering_classifier.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        eval_path = os.path.join(PROJECT_ROOT, "training", "evaluation_report.json")
        eval_data = {}
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
        return jsonify({"status": "success", "message": "Model retrained successfully", "report": eval_data})
    else:
        return jsonify({"status": "error", "message": res.stderr}), 500


def process_forensic_video(video_path: str, mode: str = "highway", max_frames: int = 350) -> dict:
    """
    Executes full offline forensic CV analysis on an ingested video clip.
    Generates browser-playable H.264 MP4 videos for both annotated evidence and raw input,
    extracts real vehicle kinematics, and compiles statutory violation registers.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Smart Corridor Routing: Ensure highway clips (e.g. video1.avi) use calibrated highway homography
    base_name = os.path.basename(video_path).lower()
    if "video1" in base_name or "highway" in base_name or (native_w == 320 and native_h == 240):
        effective_mode = "highway"
    elif "indian" in base_name:
        effective_mode = "indian"
    else:
        effective_mode = mode

    if effective_mode == "indian":
        cfg_path = os.path.join(PROJECT_ROOT, "config", "indian_road_config.yaml")
    else:
        cfg_path = os.path.join(PROJECT_ROOT, "config", "highway_config.yaml")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Super-resolution HD upscaling for low-resolution CCTV feeds (< 1000px)
    target_w, target_h = (1280, 720) if native_w < 1000 else (native_w, native_h)
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    scale_x = target_w / max(native_w, 1)
    scale_y = target_h / max(native_h, 1)

    calib_cfg = config["calibration"]

    # Check if this is a known benchmark video (video1.avi or indian_traffic_sample) or generic custom CCTV footage
    if "video1" in base_name or (native_w == 320 and native_h == 240):
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        ground_quad = calib_cfg["ground_quad"]
        lanes_cfg = config["lanes"]
    elif "indian" in base_name:
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        ground_quad = calib_cfg["ground_quad"]
        lanes_cfg = config["lanes"]
    else:
        # Universal Adaptive High-Definition CCTV Calibration for arbitrary surveillance footage
        scaled_quad = [
            [int(0.05 * target_w), int(0.95 * target_h)],   # Bottom-Left road edge
            [int(0.95 * target_w), int(0.95 * target_h)],   # Bottom-Right road edge
            [int(0.62 * target_w), int(0.38 * target_h)],   # Top-Right horizon convergence
            [int(0.22 * target_w), int(0.38 * target_h)]    # Top-Left horizon convergence
        ]
        ground_quad = [
            [0.0, 0.0],
            [15.0, 0.0],
            [15.0, 120.0],
            [0.0, 120.0]
        ]
        lanes_cfg = [
            {"id": 1, "name": "Lane 1 (Left)", "left_boundary_x": 0.0, "right_boundary_x": 5.0, "centerline_x": 2.5, "direction_y": 1.0, "speed_limit_kmh": 90.0},
            {"id": 2, "name": "Lane 2 (Middle)", "left_boundary_x": 5.0, "right_boundary_x": 10.0, "centerline_x": 7.5, "direction_y": 1.0, "speed_limit_kmh": 90.0},
            {"id": 3, "name": "Lane 3 (Right)", "left_boundary_x": 10.0, "right_boundary_x": 15.0, "centerline_x": 12.5, "direction_y": 1.0, "speed_limit_kmh": 90.0}
        ]

    calibrator = HomographyCalibrator(scaled_quad, ground_quad)
    geometry_mgr = RoadGeometryManager(lanes_cfg, calibrator)

    # Use high-accuracy yolov8s.pt weights for sharp detection of small and distant vehicles
    model_file = "yolov8s.pt" if os.path.exists(os.path.join(PROJECT_ROOT, "yolov8s.pt")) else "yolov8n.pt"

    class_type_map = {
        0: {"name": "Pedestrian", "type": "pedestrian_vendor"},
        1: {"name": "Bicycle/Cart", "type": "pushcart_vendor"},
        2: {"name": "Car", "type": "vehicle"},
        3: {"name": "Motorcycle", "type": "vehicle"},
        5: {"name": "Bus", "type": "vehicle"},
        7: {"name": "Truck", "type": "vehicle"},
        16: {"name": "Stray Dog", "type": "animal"},
        17: {"name": "Animal Cart", "type": "animal"},
        19: {"name": "Cow/Cattle", "type": "animal"}
    }

    detector = VehicleDetectionTracker(
        model_weights=model_file,
        calibrator=calibrator,
        target_classes=[0, 1, 2, 3, 5, 7, 16, 17, 19],
        conf_thresh=0.20,
        tracker_config="bytetrack.yaml",
        class_type_map=class_type_map,
        imgsz=1280
    )
    behavior_engine = BehaviorEngine(geometry_mgr, config.get("rules", {}))
    visualizer = LATISVisualizer(geometry_mgr, config)

    # Output file paths
    base_stem = os.path.splitext(os.path.basename(video_path))[0]
    file_hash = hashlib.md5(f"{base_stem}_{mode}_{time.time()}".encode()).hexdigest()[:8]
    upload_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    ann_filename = f"annotated_{base_stem}_{file_hash}.mp4"
    raw_filename = f"raw_{base_stem}_{file_hash}.mp4"
    ann_path = os.path.join(upload_dir, ann_filename)
    raw_path = os.path.join(upload_dir, raw_filename)

    ann_writer = imageio_ffmpeg.write_frames(
        ann_path,
        (target_w, target_h),
        fps=fps_in,
        codec='libx264',
        pix_fmt_in='bgr24',
        pix_fmt_out='yuv420p',
        ffmpeg_log_level='error'
    )
    ann_writer.send(None)

    raw_writer = imageio_ffmpeg.write_frames(
        raw_path,
        (target_w, target_h),
        fps=fps_in,
        codec='libx264',
        pix_fmt_in='bgr24',
        pix_fmt_out='yuv420p',
        ffmpeg_log_level='error'
    )
    raw_writer.send(None)

    violations = []
    seen_infractions = set()
    speeds = []
    unique_tracks = set()

    frame_idx = 0
    t_start = time.time()
    t_prev = t_start

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        if (native_w, native_h) != (target_w, target_h):
            frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            frame_resized = frame

        raw_writer.send(frame_resized.tobytes())

        timestamp = frame_idx / fps_in
        t_now = time.time()
        fps_current = 1.0 / max(t_now - t_prev, 1e-4)
        t_prev = t_now

        vehicles = detector.process_frame(frame_resized, timestamp)
        for v in vehicles:
            unique_tracks.add(v.track_id)
            if v.speed_kmh > 0:
                speeds.append(v.speed_kmh)

        frame_alerts = []
        new_infractions_this_frame = []
        for v in vehicles:
            v_alerts = behavior_engine.analyze_vehicle(v, timestamp)
            frame_alerts.extend(v_alerts)
            for a in v_alerts:
                key = f"{a.track_id}_{a.event_type}"
                if key not in seen_infractions:
                    seen_infractions.add(key)
                    # Statutory legal mapping (MVA 2019 / IPC)
                    mva_sec = "Section 177A, MVA 2019"
                    penalty = "₹ 1,000"
                    ev_type = a.event_type.upper()
                    if "SPEED" in ev_type:
                        mva_sec = "Section 183, MVA 2019"
                        penalty = "₹ 2,000"
                    elif "WEAV" in ev_type or "ERRATIC" in ev_type or "WAVER" in ev_type:
                        mva_sec = "Section 184, MVA 2019"
                        penalty = "₹ 5,000"
                    elif "CATTLE" in ev_type or "LIVESTOCK" in ev_type:
                        mva_sec = "Section 289, IPC (Livestock Obstruction)"
                        penalty = "₹ 1,000"
                    elif "VENDOR" in ev_type or "PUSH" in ev_type or "OBSTRUCT" in ev_type:
                        mva_sec = "Sec 283 IPC / Sec 201 MVA"
                        penalty = "₹ 2,000"
                    elif "WRONG" in ev_type:
                        mva_sec = "Section 184 / 119 MVA"
                        penalty = "₹ 5,000"
                    elif "SWERVE" in ev_type or "POTHOLE" in ev_type:
                        mva_sec = "Section 184, MVA 2019"
                        penalty = "₹ 1,000"

                    event_id = f"INC-{a.track_id}-{int(timestamp * 10)}"
                    vio_item = {
                        "id": event_id,
                        "ref": f"VIO-{len(violations)+1:02d}",
                        "track_id": a.track_id,
                        "entity": a.class_name,
                        "infraction": a.message,
                        "speed": f"{int(a.metrics.get('speed_kmh', v.speed_kmh))} km/h",
                        "lane": f"Lane {a.metrics.get('lane_id', 1)}",
                        "timestamp": f"{timestamp:.1f}s",
                        "mva": mva_sec,
                        "penalty": penalty,
                        "severity": a.severity,
                        "confidence": getattr(a, "confidence", 0.90),
                        "evidence_url": f"/api/media/uploads/evidence/{event_id}.jpg"
                    }
                    violations.append(vio_item)
                    new_infractions_this_frame.append((a, vio_item))

        annotated_frame = visualizer.draw(frame_resized, vehicles, frame_alerts, timestamp, fps_current)
        ann_writer.send(annotated_frame.tobytes())

        # Save snapshot evidence frames and record into SQLite events DB
        if new_infractions_this_frame:
            ev_dir = os.path.join(upload_dir, "evidence")
            os.makedirs(ev_dir, exist_ok=True)
            for n_alert, n_vio in new_infractions_this_frame:
                ev_path = os.path.join(ev_dir, f"{n_vio['id']}.jpg")
                cv2.imwrite(ev_path, annotated_frame)
                insert_event({
                    "id": n_vio["id"],
                    "video_id": os.path.basename(video_path),
                    "track_id": n_alert.track_id,
                    "class_name": n_alert.class_name,
                    "event_type": n_alert.event_type,
                    "severity": n_alert.severity,
                    "reason": n_alert.message,
                    "confidence": getattr(n_alert, "confidence", 0.90),
                    "timestamp_s": timestamp,
                    "speed_kmh": float(n_alert.metrics.get("speed_kmh", 0.0)),
                    "lane_id": int(n_alert.metrics.get("lane_id", 1)),
                    "evidence_frame_url": n_vio["evidence_url"]
                })

    cap.release()
    ann_writer.close()
    raw_writer.close()

    elapsed = round(time.time() - t_start, 2)
    avg_speed = int(np.mean(speeds)) if speeds else 0
    max_speed = int(np.max(speeds)) if speeds else 0

    return {
        "status": "completed",
        "clip_name": os.path.basename(video_path),
        "processed_video_url": f"/api/media/uploads/{ann_filename}",
        "raw_video_url": f"/api/media/uploads/{raw_filename}",
        "total_frames": frame_idx,
        "duration_s": round(frame_idx / fps_in, 1),
        "active_targets_count": len(unique_tracks),
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": max_speed,
        "total_violations": len(violations),
        "violations": violations,
        "processing_time_s": elapsed
    }


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Forensic video analysis endpoint.
    Accepts preset clip selection or file upload, runs the LATIS CV pipeline,
    and returns real processed video URLs and incident register.
    """
    try:
        uploaded_file = request.files.get('file')
        model = request.form.get('model') or "indian"
        max_frames = int(request.form.get('max_frames', 350))
        json_data = request.get_json(silent=True) or {}
        preset = request.form.get('preset') or json_data.get('preset')

        if uploaded_file and uploaded_file.filename:
            upload_dir = os.path.join(PROJECT_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            safe_name = os.path.basename(uploaded_file.filename)
            save_path = os.path.join(upload_dir, safe_name)
            uploaded_file.save(save_path)

            result = process_forensic_video(save_path, mode=model, max_frames=max_frames)
            return jsonify(result)

        elif preset:
            # Preset Case Study execution or retrieval
            if preset in ["case_highway", "highway"]:
                source_video = os.path.join(PROJECT_ROOT, "video1.avi")
                mode = "highway"
            else:
                source_video = os.path.join(PROJECT_ROOT, "indian_traffic_sample.mp4")
                mode = "indian"

            result = process_forensic_video(source_video, mode=mode, max_frames=max_frames)
            return jsonify(result)

        else:
            return jsonify({"error": "No video file or preset provided"}), 400

    except Exception as ex:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(ex), "status": "failed"}), 500


@app.route('/api/media/<path:filename>')
def serve_media(filename):
    """Serves video files directly with full streaming byte-range support."""
    return send_from_directory(PROJECT_ROOT, filename)


if __name__ == '__main__':
    print("=" * 65)
    print("   PROJECT LATIS - HEADLESS COMPUTER VISION API (PORT 5000)  ")
    print("   Pure REST & MJPEG Stream • Zero HTML • CORS Enabled       ")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
