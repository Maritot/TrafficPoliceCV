"""
Stage 1: Interactive Lane & Homography Calibration Authoring Tool.
Allows students to click 4 road corners on ANY real traffic video frame,
enter road dimensions, and export an updated config.yaml file automatically!
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np


class InteractiveLaneAuthor:
    def __init__(self, video_path: str, output_config_path: str = "config/custom_config.yaml"):
        self.video_path = video_path
        self.output_config_path = output_config_path
        self.points = []
        self.frame = None
        self.window_name = "LATIS Calibration: Click 4 Road Corners"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"[+] Point {len(self.points)} recorded: ({x}, {y})")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[-] Error: Unable to open video '{self.video_path}'")
            return

        ret, self.frame = cap.read()
        cap.release()

        if not ret or self.frame is None:
            print("[-] Error: Unable to read frame from video.")
            return

        h, w = self.frame.shape[:2]
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        instructions = [
            "CLICK 4 CORNERS IN CLOCKWISE ORDER:",
            "1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left",
            "Press 'r' to reset points, 'c' when done, 'q' to quit."
        ]

        print("\n" + "=" * 60)
        print("          LATIS INTERACTIVE CALIBRATION TOOL            ")
        print("=" * 60)
        for ins in instructions:
            print("  " + ins)
        print("=" * 60 + "\n")

        while True:
            display = self.frame.copy()

            # Draw instructions on frame
            for idx, text in enumerate(instructions):
                cv2.putText(display, text, (20, 30 + idx * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw clicked points
            for i, pt in enumerate(self.points):
                cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
                cv2.putText(display, f"P{i+1}", (pt[0] + 10, pt[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw polygon connecting clicked points
            if len(self.points) > 1:
                pts_arr = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts_arr], isClosed=(len(self.points) == 4), 
                              color=(0, 220, 255), thickness=2)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                self.points = []
                print("[!] Reset all points.")
            elif key == ord('q'):
                print("[!] Cancelled by user.")
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                if len(self.points) == 4:
                    break
                else:
                    print(f"[-] Please click all 4 points before continuing (currently {len(self.points)}).")

        cv2.destroyAllWindows()

        # Prompt user in terminal for real road dimensions
        print("\n[+] 4 Points Captured Successfully!")
        print(f"    P1 (Bottom-Left)  : {self.points[0]}")
        print(f"    P2 (Bottom-Right) : {self.points[1]}")
        print(f"    P3 (Top-Right)    : {self.points[2]}")
        print(f"    P4 (Top-Left)     : {self.points[3]}")

        # Ask for metric inputs with sensible defaults
        try:
            width_str = input("\nEnter total road carriageway width in meters [default 7.5]: ").strip()
            road_width = float(width_str) if width_str else 7.5

            length_str = input("Enter monitored road length in meters [default 40.0]: ").strip()
            road_length = float(length_str) if length_str else 40.0

            lanes_str = input("Enter number of traffic lanes [default 2]: ").strip()
            num_lanes = int(lanes_str) if lanes_str else 2
        except Exception:
            road_width, road_length, num_lanes = 7.5, 40.0, 2

        # Generate Lane Configuration
        lane_width = road_width / num_lanes
        lanes = []
        for i in range(num_lanes):
            lx = i * lane_width
            rx = (i + 1) * lane_width
            cx = (lx + rx) / 2.0
            lanes.append({
                "id": i + 1,
                "name": f"Lane {i + 1}",
                "left_boundary_x": float(lx),
                "right_boundary_x": float(rx),
                "centerline_x": float(cx),
                "direction_y": 1.0,
                "speed_limit_kmh": 60.0
            })

        # Save to YAML
        config_data = {
            "camera": {
                "name": "CUSTOM_CAM",
                "resolution": [w, h],
                "fps": 25
            },
            "calibration": {
                "image_quad": self.points,
                "ground_quad": [
                    [0.0, 0.0],
                    [float(road_width), 0.0],
                    [float(road_width), float(road_length)],
                    [0.0, float(road_length)]
                ]
            },
            "lanes": lanes,
            "perception": {
                "model_weights": "yolov8n.pt",
                "confidence_threshold": 0.35,
                "iou_threshold": 0.50,
                "tracker_type": "bytetrack.yaml",
                "target_classes": [2, 3, 5, 7]
            },
            "rules": {
                "wrong_way": {
                    "enabled": True,
                    "min_heading_dev_deg": 125.0,
                    "persistence_frames": 10
                },
                "stopped_vehicle": {
                    "enabled": True,
                    "max_speed_kmh": 2.5,
                    "min_stopped_seconds": 3.0
                },
                "lane_keeping": {
                    "enabled": True,
                    "straddle_ratio": 0.75,
                    "violation_ratio": 1.05
                },
                "oscillation": {
                    "enabled": True,
                    "min_sign_flips": 3,
                    "amplitude_threshold_m": 0.40
                }
            },
            "visualization": {
                "show_bev_radar": True,
                "show_trails": True,
                "trail_length": 25,
                "output_path": "output_custom_annotated.mp4"
            }
        }

        os.makedirs(os.path.dirname(self.output_config_path), exist_ok=True)
        with open(self.output_config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"\n[+] New calibration & geometry saved to: '{self.output_config_path}'")
        print(f"[+] You can now run:\n    python run_latis.py --source {self.video_path} --config {self.output_config_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LATIS Interactive Lane Authoring Tool")
    parser.add_argument("--source", type=str, default="traffic_sample.mp4",
                        help="Video file path to calibrate")
    parser.add_argument("--out-config", type=str, default="config/custom_config.yaml",
                        help="Path to save generated YAML configuration")
    args = parser.parse_args()

    author = InteractiveLaneAuthor(args.source, args.out_config)
    author.run()
