"""
LATIS - Main Pipeline Execution Entry Point.
Integrates Calibration, Geometry, YOLO+ByteTrack Detection,
Behavior Analytics, and Visualizer.
"""

import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np

from core.calibration import HomographyCalibrator
from core.geometry import RoadGeometryManager
from core.detector_tracker import VehicleDetectionTracker
from core.behavior_engine import BehaviorEngine, AlertEvent
from core.visualizer import LATISVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="LATIS - Lane-Aware Traffic Intelligence System")
    parser.add_argument("--source", type=str, default="traffic_sample.mp4",
                        help="Path to video file, webcam index (e.g. 0), or RTSP stream")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--save", type=str, default="output_latis_annotated.mp4",
                        help="Output file path to save annotated video")
    parser.add_argument("--headless", action="store_true",
                        help="Run without displaying OpenCV GUI window")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"[-] Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("      LANE-AWARE TRAFFIC INTELLIGENCE SYSTEM (LATIS)      ")
    print("      Computer Vision Traffic Analytics & Safety Engine   ")
    print("=" * 65)

    # 2. Initialize Stage 0: Calibration
    calib_cfg = config["calibration"]
    calibrator = HomographyCalibrator(
        image_quad=calib_cfg["image_quad"],
        ground_quad=calib_cfg["ground_quad"]
    )
    print(f"[+] Stage 0 Calibration Initialized.")
    print(f"    - Homography Reprojection Error: {calibrator.mean_reproj_error_m:.4f} m (Gate: <= 0.15 m)")

    # 3. Initialize Stage 1: Road & Lane Geometry
    geometry_mgr = RoadGeometryManager(
        lanes_config=config["lanes"],
        calibrator=calibrator
    )
    print(f"[+] Stage 1 Geometry Loaded: {len(geometry_mgr.lanes)} active monitored lanes.")

    # 4. Initialize Stage 2 & 3: Perception (Detection + ByteTrack)
    perc_cfg = config["perception"]
    detector = VehicleDetectionTracker(
        model_weights=perc_cfg.get("model_weights", "yolov8n.pt"),
        calibrator=calibrator,
        target_classes=perc_cfg.get("target_classes", [2, 3, 5, 7]),
        conf_thresh=float(perc_cfg.get("confidence_threshold", 0.35)),
        tracker_config=perc_cfg.get("tracker_type", "bytetrack.yaml"),
        class_type_map=perc_cfg.get("class_type_map")
    )
    print(f"[+] Stage 2/3 Detector & ByteTracker Initialized with model '{perc_cfg.get('model_weights')}'.")

    # 5. Initialize Stage 6: Behavior Rule Engine
    behavior_engine = BehaviorEngine(
        geometry_mgr=geometry_mgr,
        rules_config=config.get("rules", {})
    )
    print("[+] Stage 6 Deterministic Behavior Engine Loaded.")

    # 6. Initialize Visualizer
    visualizer = LATISVisualizer(geometry_mgr=geometry_mgr, config=config)

    # 7. Open Video Source
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source '{args.source}'.")
        print("    Tip: Run 'python utils/generate_sample_video.py' first to create a test video!")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[+] Ingesting Stream: {width}x{height} @ {fps_in:.1f} FPS (Total frames: {total_frames if total_frames > 0 else 'Live'})")

    # Automatic Super-Resolution HD Upscaling for clear projections
    target_w, target_h = 1280, 720
    upscale_enabled = (width < 1000)
    if upscale_enabled:
        print(f"[+] Low native resolution ({width}x{height}) detected. Activating Super-Resolution HD (1280x720) for crystal-clear projections...")
        scale_x = target_w / width
        scale_y = target_h / height
        scaled_quad = [[pt[0] * scale_x, pt[1] * scale_y] for pt in calib_cfg["image_quad"]]
        calibrator = HomographyCalibrator(
            image_quad=scaled_quad,
            ground_quad=calib_cfg["ground_quad"]
        )
        geometry_mgr = RoadGeometryManager(lanes_config=config["lanes"], calibrator=calibrator)
        detector.calibrator = calibrator
        visualizer = LATISVisualizer(geometry_mgr=geometry_mgr, config=config)
        out_w, out_h = target_w, target_h
    else:
        out_w, out_h = width, height

    # Output Video Writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (out_w, out_h))
        print(f"[+] Recording HD annotated stream ({out_w}x{out_h}) to: {args.save}")

    print("\n[+] Running analysis pipeline... Press 'q' to stop.\n")

    frame_idx = 0
    t_start = time.time()
    t_prev = t_start
    alert_summary = {"CRITICAL": 0, "WARNING": 0}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if upscale_enabled:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            now = time.time()
            dt = now - t_prev
            t_prev = now
            current_fps = 1.0 / dt if dt > 0 else fps_in
            timestamp = frame_idx / fps_in

            # 1. Detection and Tracking
            vehicles = detector.process_frame(frame, timestamp)

            # 2. Behavior Analytics & Anomaly Detection
            frame_alerts: List[AlertEvent] = []
            for v in vehicles:
                v_alerts = behavior_engine.analyze_vehicle(v, timestamp)
                for a in v_alerts:
                    frame_alerts.append(a)
                    alert_summary[a.severity] = alert_summary.get(a.severity, 0) + 1
                    # Console log for alerts
                    print(f"  [! ALERT {a.severity}] [{timestamp:.1f}s] {a.message}")

            # 3. Visual Annotations & HUD & BEV Radar
            annotated_frame = visualizer.draw(frame, vehicles, frame_alerts, timestamp, current_fps)

            # 4. Save to output video
            if writer:
                writer.write(annotated_frame)

            # 5. Display window (if not headless)
            if not args.headless:
                cv2.imshow("LATIS Traffic Police AI", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[!] User interrupted playback.")
                    break

            if frame_idx % 50 == 0:
                elapsed = time.time() - t_start
                avg_fps = frame_idx / elapsed
                print(f"    Processed {frame_idx} frames | Avg FPS: {avg_fps:.1f} | Active Tracks: {len(vehicles)}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print("                    PIPELINE SUMMARY                     ")
    print("=" * 65)
    print(f"  Total Frames Processed : {frame_idx}")
    print(f"  Total Processing Time  : {total_time:.2f} s")
    print(f"  Average Processing FPS : {frame_idx / max(total_time, 1e-4):.2f}")
    print(f"  Critical Alerts Raised : {alert_summary.get('CRITICAL', 0)}")
    print(f"  Warning Alerts Raised  : {alert_summary.get('WARNING', 0)}")
    if args.save:
        print(f"  Annotated Video Saved  : {os.path.abspath(args.save)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
