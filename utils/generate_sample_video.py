"""
Synthetic Traffic Video Generator for LATIS Testing.
Creates a realistic 1280x720 25 FPS video clip with:
- Oblique perspective road with 2 lanes
- Normal traveling vehicles (cars, trucks)
- WRONG-WAY vehicle (driving opposite direction)
- STOPPED vehicle in live lane
- LANE STRADDLING vehicle
Allows testing the entire CV pipeline immediately without needing external CCTV downloads!
"""

import cv2
import numpy as np
import os


def create_traffic_sample(output_filename="traffic_sample.mp4", num_frames=350, fps=25):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"[+] Generating synthetic traffic surveillance video '{output_filename}' ({num_frames} frames)...")

    # Perspective road trapezoid matching default_config.yaml
    # Bottom: [280, 680] to [1000, 680], Top: [520, 260] to [760, 260]
    p_bl = np.array([280, 680], dtype=np.float32)
    p_br = np.array([1000, 680], dtype=np.float32)
    p_tr = np.array([760, 260], dtype=np.float32)
    p_tl = np.array([520, 260], dtype=np.float32)

    # Vehicles state definitions:
    # y_pos is along road: 0.0 (near bottom) to 1.0 (far top)
    # lane: 1 (left) or 2 (right)
    vehicles = [
        # 1. Normal Car in Lane 2 (moving away from camera, y: 0.1 -> 0.95)
        {"type": "car", "color": (220, 50, 50), "lane_t": 0.75, "y_start": 0.1, "speed": 0.0035, "mode": "normal"},
        
        # 2. Normal Truck in Lane 1 (moving away, y: 0.2 -> 0.95)
        {"type": "truck", "color": (40, 150, 200), "lane_t": 0.25, "y_start": 0.05, "speed": 0.0025, "mode": "normal"},
        
        # 3. WRONG-WAY Car (starts far away at top of Lane 1 and drives TOWARD camera, y: 0.9 -> 0.1)
        {"type": "car", "color": (30, 30, 230), "lane_t": 0.28, "y_start": 0.85, "speed": -0.0040, "mode": "wrong_way"},

        # 4. STOPPED Car in Lane 2 (starts moving then stops completely around frame 75)
        {"type": "car", "color": (230, 220, 40), "lane_t": 0.72, "y_start": 0.25, "speed": 0.0020, "mode": "stopping"},

        # 5. WEAVING / Lane Straddler Car (moves forward while oscillating horizontally across boundary)
        {"type": "car", "color": (180, 40, 180), "lane_t": 0.50, "y_start": 0.15, "speed": 0.0030, "mode": "weaving"}
    ]

    for frame_idx in range(num_frames):
        # 1. Background scene: Road & roadside environment
        canvas = np.full((height, width, 3), (70, 120, 70), dtype=np.uint8)  # Grass green

        # Road surface (Dark asphalt)
        road_poly = np.array([p_bl, p_br, p_tr, p_tl], dtype=np.int32)
        cv2.fillPoly(canvas, [road_poly], (50, 50, 50))

        # Kerbs / Shoulders
        cv2.line(canvas, tuple(p_bl.astype(int)), tuple(p_tl.astype(int)), (220, 220, 220), 4)
        cv2.line(canvas, tuple(p_br.astype(int)), tuple(p_tr.astype(int)), (220, 220, 220), 4)

        # Center dashed divider line
        p_bc = (p_bl + p_br) / 2.0
        p_tc = (p_tl + p_tr) / 2.0
        dash_steps = 18
        for d in range(0, dash_steps, 2):
            t1 = d / dash_steps
            t2 = (d + 1) / dash_steps
            pt1 = (p_bc * (1 - t1) + p_tc * t1).astype(int)
            pt2 = (p_bc * (1 - t2) + p_tc * t2).astype(int)
            cv2.line(canvas, tuple(pt1), tuple(pt2), (240, 240, 240), 3)

        # 2. Render each vehicle with proper perspective scaling
        # Sort vehicles by y (depth) so far vehicles are drawn first (painters algorithm)
        current_v_states = []
        for v in vehicles:
            if v["mode"] == "stopping":
                if frame_idx < 80:
                    y = v["y_start"] + v["speed"] * frame_idx
                else:
                    y = v["y_start"] + v["speed"] * 80  # Stop completely!
                lane_t = v["lane_t"]
            elif v["mode"] == "weaving":
                y = v["y_start"] + v["speed"] * frame_idx
                # Oscillate across center (0.35 to 0.65)
                lane_t = 0.50 + 0.18 * np.sin(frame_idx * 0.15)
            else:
                y = v["y_start"] + v["speed"] * frame_idx
                lane_t = v["lane_t"]

            # Wrap around when out of bounds
            if y > 0.98:
                y = 0.05
            elif y < 0.05 and v["mode"] == "wrong_way":
                y = 0.90

            current_v_states.append((y, lane_t, v))

        # Sort descending by y (top of road first)
        current_v_states.sort(key=lambda item: item[0], reverse=True)

        for y, lane_t, v in current_v_states:
            # Perspective interpolation of road width at depth y
            left_edge = p_bl * (1 - y) + p_tl * y
            right_edge = p_br * (1 - y) + p_tr * y
            
            # Position along horizontal road cross-section
            center_pt = left_edge * (1 - lane_t) + right_edge * lane_t
            cx, cy = int(center_pt[0]), int(center_pt[1])

            # Perspective scale factor (larger near bottom y=0, smaller near top y=1)
            scale = (1.0 - y * 0.72)

            if v["type"] == "truck":
                bw = int(120 * scale)
                bh = int(170 * scale)
            else:
                bw = int(90 * scale)
                bh = int(115 * scale)

            x1 = cx - bw // 2
            y1 = cy - bh
            x2 = cx + bw // 2
            y2 = cy

            # Draw vehicle body (Car / Truck shape)
            body_color = v["color"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), body_color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 2)

            # Windshield / Roof
            windshield_margin = int(12 * scale)
            cv2.rectangle(canvas, (x1 + windshield_margin, y1 + int(bh * 0.2)), 
                          (x2 - windshield_margin, y1 + int(bh * 0.55)), (40, 40, 40), -1)

            # Headlights / Tail-lights
            light_w = max(4, int(10 * scale))
            light_h = max(3, int(6 * scale))
            if v["mode"] == "wrong_way":
                # Facing camera: Bright white headlights
                cv2.rectangle(canvas, (x1 + 6, y2 - 12), (x1 + 6 + light_w, y2 - 12 + light_h), (255, 255, 200), -1)
                cv2.rectangle(canvas, (x2 - 6 - light_w, y2 - 12), (x2 - 6, y2 - 12 + light_h), (255, 255, 200), -1)
            else:
                # Facing away: Red tail-lights
                cv2.rectangle(canvas, (x1 + 6, y2 - 10), (x1 + 6 + light_w, y2 - 10 + light_h), (0, 0, 255), -1)
                cv2.rectangle(canvas, (x2 - 6 - light_w, y2 - 10), (x2 - 6, y2 - 10 + light_h), (0, 0, 255), -1)

            # Tires (dark patches at bottom corners)
            tire_w = max(4, int(10 * scale))
            tire_h = max(6, int(14 * scale))
            cv2.rectangle(canvas, (x1 - tire_w // 2, y2 - tire_h), (x1 + tire_w // 2, y2), (10, 10, 10), -1)
            cv2.rectangle(canvas, (x2 - tire_w // 2, y2 - tire_h), (x2 + tire_w // 2, y2), (10, 10, 10), -1)

        out.write(canvas)

    out.release()
    print(f"[+] Successfully generated '{output_filename}' ({num_frames} frames, {width}x{height} @ {fps} fps).")


if __name__ == "__main__":
    create_traffic_sample()
