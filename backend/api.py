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
from core.visualizer import LATISVisualizer

app = Flask(__name__)

# Enable CORS for Next.js (port 3000)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response


# Global Telemetry & Streaming State
STATE = {
    "mode": "highway",       # "highway" or "indian"
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


def load_pipeline(mode: str, show_radar: bool = True):
    """Initializes the LATIS perception and rule engine for the selected corridor."""
    if mode == "indian":
        cfg_path = os.path.join(PROJECT_ROOT, "config", "indian_road_config.yaml")
        video_src = os.path.join(PROJECT_ROOT, "indian_traffic_sample.mp4")
    else:
        cfg_path = os.path.join(PROJECT_ROOT, "config", "highway_config.yaml")
        video_src = os.path.join(PROJECT_ROOT, "video1.avi")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config '{cfg_path}' not found.")
    if not os.path.exists(video_src):
        raise FileNotFoundError(f"Video '{video_src}' not found.")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    config["visualization"]["show_bev_radar"] = show_radar

    calib_cfg = config["calibration"]
    perc_cfg = config["perception"]

    # 1. Open Video
    cap = cv2.VideoCapture(video_src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 2. Super-Resolution HD scaling if low native resolution
    target_w, target_h = 1280, 720
    upscale = (width < 1000)
    if upscale:
        scale_x = target_w / width
        scale_y = target_h / height
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        calibrator = HomographyCalibrator(scaled_quad, calib_cfg["ground_quad"])
    else:
        calibrator = HomographyCalibrator(calib_cfg["image_quad"], calib_cfg["ground_quad"])

    # 3. Geometry Manager
    geometry_mgr = RoadGeometryManager(config["lanes"], calibrator)

    # 4. Detector & ByteTracker
    weights_path = perc_cfg.get("model_weights", "yolov8n.pt")
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(PROJECT_ROOT, weights_path)

    detector = VehicleDetectionTracker(
        model_weights=weights_path,
        calibrator=calibrator,
        target_classes=perc_cfg.get("target_classes", [2, 3, 5, 7]),
        conf_thresh=float(perc_cfg.get("confidence_threshold", 0.35)),
        tracker_config=perc_cfg.get("tracker_type", "bytetrack.yaml"),
        class_type_map=perc_cfg.get("class_type_map")
    )

    # 5. Behavior Rule Engine
    behavior_engine = BehaviorEngine(geometry_mgr, config.get("rules", {}))

    # 6. Visualizer
    visualizer = LATISVisualizer(geometry_mgr, config)

    return cap, calibrator, geometry_mgr, detector, behavior_engine, visualizer, upscale, target_w, target_h, fps_in


def generate_mjpeg_stream():
    """Continuous MJPEG frame generator streaming processed video with computer vision overlays."""
    global STATE

    while True:
        mode = STATE["mode"]
        show_radar = STATE["show_radar"]
        try:
            (cap, calibrator, geometry_mgr, detector, 
             behavior_engine, visualizer, upscale, 
             target_w, target_h, fps_in) = load_pipeline(mode, show_radar)
        except Exception as e:
            print(f"[-] Pipeline load error for mode '{mode}': {e}")
            time.sleep(1)
            continue

        frame_idx = 0
        t_start = time.time()
        t_prev = t_start
        STATE["reset_requested"] = False

        while cap.isOpened():
            if STATE["reset_requested"] or STATE["mode"] != mode or STATE["show_radar"] != show_radar:
                break

            if STATE["paused"]:
                time.sleep(0.04)
                continue

            ret, frame = cap.read()
            if not ret:
                # Seamless infinite loop for live CCTV simulation
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            current_fps = 1.0 / max(dt, 1e-4)
            timestamp = frame_idx / fps_in

            if upscale:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            try:
                vehicles = detector.process_frame(frame, timestamp)

                frame_alerts = []
                for v in vehicles:
                    v_alerts = behavior_engine.analyze_vehicle(v, timestamp)
                    frame_alerts.extend(v_alerts)

                annotated_frame = visualizer.draw(frame, vehicles, frame_alerts, timestamp, current_fps)

                # Update live telemetry
                speeds = [v.speed_kmh for v in vehicles if v.speed_kmh > 0]
                avg_speed = int(np.mean(speeds)) if speeds else 0

                STATE["fps"] = current_fps
                STATE["active_tracks"] = len(vehicles)
                STATE["avg_speed"] = avg_speed

                # Record incidents
                if frame_alerts:
                    for a in frame_alerts:
                        STATE["total_alerts"] += 1
                        alert_item = {
                            "id": f"INC-{a.track_id}-{int(timestamp * 10)}",
                            "track_id": a.track_id,
                            "class_name": a.class_name,
                            "event_type": a.event_type,
                            "severity": a.severity,
                            "message": a.message,
                            "timestamp": round(timestamp, 2),
                            "speed_kmh": round(a.metrics.get("speed_kmh", 0.0), 1),
                            "lane_id": a.metrics.get("lane_id", 1)
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
    return jsonify({
        "status": "healthy",
        "service": "LATIS Headless CV Engine",
        "uptime_s": round(time.time() - STATE["start_time"], 1),
        "mode": STATE["mode"],
        "active_tracks": STATE["active_tracks"]
    })


@app.route('/api/stream', methods=['GET'])
def stream():
    """Streams live MJPEG frames with real-time CV annotations."""
    req_mode = request.args.get('mode')
    if req_mode in ["highway", "indian"] and req_mode != STATE["mode"]:
        STATE["mode"] = req_mode
        STATE["reset_requested"] = True

    radar = request.args.get('radar')
    if radar is not None:
        STATE["show_radar"] = (radar.lower() in ["1", "true", "yes"])

    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/telemetry', methods=['GET'])
def telemetry():
    """Returns real-time vehicle kinematics and pipeline speed."""
    return jsonify({
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
    """Switches active corridor scenario."""
    data = request.get_json() or {}
    new_mode = data.get("mode")
    if new_mode in ["highway", "indian"]:
        STATE["mode"] = new_mode
        STATE["reset_requested"] = True
        STATE["total_alerts"] = 0
        STATE["recent_alerts"] = []
        return jsonify({"status": "ok", "mode": new_mode})
    return jsonify({"error": "Invalid mode. Use 'highway' or 'indian'"}), 400


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
                    elif "WEAV" in ev_type or "ERRATIC" in ev_type:
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

                    violations.append({
                        "ref": f"VIO-{len(violations)+1:02d}",
                        "track_id": a.track_id,
                        "entity": a.class_name,
                        "infraction": a.message,
                        "speed": f"{int(a.metrics.get('speed_kmh', v.speed_kmh))} km/h",
                        "lane": f"Lane {a.metrics.get('lane_id', 1)}",
                        "timestamp": f"{timestamp:.1f}s",
                        "mva": mva_sec,
                        "penalty": penalty,
                        "severity": a.severity
                    })

        annotated_frame = visualizer.draw(frame_resized, vehicles, frame_alerts, timestamp, fps_current)
        ann_writer.send(annotated_frame.tobytes())

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
