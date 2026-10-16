"""
Indian Road Conditions Traffic Video Generator.
Synthesizes a realistic 1280x720 25 FPS video clip with:
- Mud-covered road edges & side vegetation/bushes
- Potholes on asphalt surface
- Cattle / Cow wandering and standing in the active travel lane
- Street Vendor Pushcart (Shop on wheels) crossing the road
- Auto-Rickshaw (3-wheeler) maneuvering through mixed traffic
- Cars abruptly swerving to dodge potholes
- Wrong-way two-wheeler on shoulder
"""

import cv2
import numpy as np


def create_indian_traffic_sample(output_filename="indian_traffic_sample.mp4", num_frames=300, fps=25):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"[+] Generating Indian road conditions video '{output_filename}' ({num_frames} frames)...")

    # Perspective quad matching config/indian_road_config.yaml
    p_bl = np.array([240, 690], dtype=np.float32)
    p_br = np.array([1040, 690], dtype=np.float32)
    p_tr = np.array([780, 250], dtype=np.float32)
    p_tl = np.array([500, 250], dtype=np.float32)

    entities = [
        # 1. Cow / Cattle standing in Lane 1 (lane_t ~ 0.28, moves very slowly)
        {"type": "cow", "color": (210, 210, 210), "lane_t": 0.26, "y_start": 0.40, "speed": 0.0003, "mode": "animal"},

        # 2. Street Vendor with Pushcart / Thela crossing from left shoulder (lane_t 0.05 -> 0.45)
        {"type": "vendor_cart", "color": (120, 60, 160), "lane_t_start": 0.02, "lane_t_speed": 0.0018, "y_start": 0.65, "speed": 0.0005, "mode": "vendor"},

        # 3. Auto-Rickshaw (3-Wheeler) in Lane 2 (moving away, lane_t ~ 0.72)
        {"type": "auto_rickshaw", "color": (0, 215, 255), "lane_t": 0.72, "y_start": 0.15, "speed": 0.0035, "mode": "vehicle"},

        # 4. Car in Lane 1 that abruptly swerves into Lane 2 to avoid the cow!
        {"type": "car", "color": (40, 60, 220), "lane_t_base": 0.28, "y_start": 0.10, "speed": 0.0038, "mode": "swerve_car"},

        # 5. Wrong-Way Motorcyclist (traveling opposite toward camera on left shoulder)
        {"type": "motorcycle", "color": (30, 180, 30), "lane_t": 0.12, "y_start": 0.85, "speed": -0.0040, "mode": "wrong_way"}
    ]

    for frame_idx in range(num_frames):
        # 1. Background scene: Dusty earth & roadside bushes
        canvas = np.full((height, width, 3), (35, 75, 40), dtype=np.uint8)  # Bushes

        # Mud/dirt roadside shoulders
        shoulder_l = np.array([[120, 720], [320, 720], [530, 220], [420, 220]], dtype=np.int32)
        shoulder_r = np.array([[960, 720], [1160, 720], [840, 220], [720, 220]], dtype=np.int32)
        cv2.fillPoly(canvas, [shoulder_l], (50, 80, 105))  # Brownish dirt
        cv2.fillPoly(canvas, [shoulder_r], (50, 80, 105))

        # Main road asphalt
        road_poly = np.array([p_bl, p_br, p_tr, p_tl], dtype=np.int32)
        cv2.fillPoly(canvas, [road_poly], (55, 55, 55))

        # Mud patches covering parts of road edges and markings
        mud_patch_1 = np.array([[240, 690], [360, 690], [420, 520], [320, 530]], dtype=np.int32)
        mud_patch_2 = np.array([[950, 690], [1040, 690], [880, 480], [820, 490]], dtype=np.int32)
        cv2.fillPoly(canvas, [mud_patch_1], (45, 70, 90))
        cv2.fillPoly(canvas, [mud_patch_2], (45, 70, 90))

        # Potholes on the road surface
        cv2.ellipse(canvas, (460, 520), (35, 16), 15, 0, 360, (25, 25, 25), -1)  # Pothole 1 in Lane 1
        cv2.ellipse(canvas, (460, 520), (38, 18), 15, 0, 360, (70, 70, 70), 1)
        cv2.ellipse(canvas, (750, 420), (28, 12), -10, 0, 360, (28, 28, 28), -1) # Pothole 2 in Lane 2

        # Faded / worn centerline divider
        p_bc = (p_bl + p_br) / 2.0
        p_tc = (p_tl + p_tr) / 2.0
        dash_steps = 16
        for d in range(0, dash_steps, 2):
            t1 = d / dash_steps
            t2 = (d + 1) / dash_steps
            pt1 = (p_bc * (1 - t1) + p_tc * t1).astype(int)
            pt2 = (p_bc * (1 - t2) + p_tc * t2).astype(int)
            cv2.line(canvas, tuple(pt1), tuple(pt2), (180, 180, 170), 2)

        # 2. Render Entities (Cattle, Vendors, Auto-Rickshaw, Cars)
        active_states = []
        for e in entities:
            y = e["y_start"] + e["speed"] * frame_idx
            
            if e["mode"] == "animal":
                lane_t = e["lane_t"] + 0.02 * np.sin(frame_idx * 0.05)
            elif e["mode"] == "vendor":
                # Vendor slowly pushes cart across from shoulder into Lane 1
                lane_t = min(0.42, e["lane_t_start"] + e["lane_t_speed"] * frame_idx)
            elif e["mode"] == "swerve_car":
                # Car approaches cow around frame 60-120 and swerves sharply right!
                if 60 <= frame_idx <= 130:
                    swerve_offset = 0.35 * np.sin((frame_idx - 60) / 70.0 * np.pi)
                else:
                    swerve_offset = 0.0
                lane_t = e["lane_t_base"] + swerve_offset
            else:
                lane_t = e["lane_t"]

            # Wrap around
            if y > 0.98: y = 0.05
            elif y < 0.05 and e["mode"] == "wrong_way": y = 0.90

            active_states.append((y, lane_t, e))

        # Sort descending by y (depth ordering)
        active_states.sort(key=lambda item: item[0], reverse=True)

        for y, lane_t, e in active_states:
            left_edge = p_bl * (1 - y) + p_tl * y
            right_edge = p_br * (1 - y) + p_tr * y
            center_pt = left_edge * (1 - lane_t) + right_edge * lane_t
            cx, cy = int(center_pt[0]), int(center_pt[1])

            scale = max(0.25, 1.0 - y * 0.70)

            if e["type"] == "cow":
                # Cow / Cattle shape (wide torso, head, horns)
                bw = int(100 * scale)
                bh = int(65 * scale)
                x1, y1 = cx - bw // 2, cy - bh
                x2, y2 = cx + bw // 2, cy
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), -1)  # White/grey cow body
                # Dark patches
                cv2.circle(canvas, (cx - int(15*scale), cy - int(30*scale)), int(15*scale), (80, 80, 80), -1)
                # Head
                cv2.rectangle(canvas, (x1 - int(18*scale), y1 + int(10*scale)), (x1, y2 - int(10*scale)), (180, 180, 180), -1)
                # Hooves
                cv2.rectangle(canvas, (x1 + 6, y2 - 8), (x1 + 16, y2), (40, 40, 40), -1)
                cv2.rectangle(canvas, (x2 - 16, y2 - 8), (x2 - 6, y2), (40, 40, 40), -1)
                cv2.putText(canvas, "COW", (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale, (255, 255, 255), 1)

            elif e["type"] == "vendor_cart":
                # Street Vendor with Pushcart / Thela on wheels
                bw = int(95 * scale)
                bh = int(80 * scale)
                x1, y1 = cx - bw // 2, cy - bh
                x2, y2 = cx + bw // 2, cy
                # Cart wooden table & goods
                cv2.rectangle(canvas, (x1, y1 + int(20*scale)), (x2, y2 - int(15*scale)), (70, 110, 150), -1)
                # Goods / vegetables on cart
                cv2.circle(canvas, (cx - int(15*scale), y1 + int(25*scale)), int(10*scale), (30, 180, 50), -1)
                cv2.circle(canvas, (cx + int(15*scale), y1 + int(25*scale)), int(10*scale), (40, 80, 220), -1)
                # Cart wheels
                wheel_r = max(4, int(12 * scale))
                cv2.circle(canvas, (x1 + int(20*scale), y2 - wheel_r), wheel_r, (20, 20, 20), -1)
                cv2.circle(canvas, (x2 - int(20*scale), y2 - wheel_r), wheel_r, (20, 20, 20), -1)
                # Vendor person standing behind cart
                cv2.circle(canvas, (x1 - int(10*scale), y1 + int(10*scale)), int(8*scale), (180, 150, 120), -1)
                cv2.putText(canvas, "THELA", (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35*scale, (255, 255, 255), 1)

            elif e["type"] == "auto_rickshaw":
                # Auto-Rickshaw (Yellow roof, green lower body, black windshield)
                bw = int(80 * scale)
                bh = int(95 * scale)
                x1, y1 = cx - bw // 2, cy - bh
                x2, y2 = cx + bw // 2, cy
                # Yellow roof
                cv2.rectangle(canvas, (x1, y1), (x2, y1 + int(bh * 0.45)), (0, 215, 255), -1)
                # Green body
                cv2.rectangle(canvas, (x1, y1 + int(bh * 0.45)), (x2, y2), (25, 120, 25), -1)
                # Windshield
                cv2.rectangle(canvas, (x1 + int(8*scale), y1 + int(12*scale)), 
                              (x2 - int(8*scale), y1 + int(bh * 0.4)), (30, 30, 30), -1)
                # Rear wheels
                cv2.rectangle(canvas, (x1 - 4, y2 - int(12*scale)), (x1 + 6, y2), (10, 10, 10), -1)
                cv2.rectangle(canvas, (x2 - 6, y2 - int(12*scale)), (x2 + 4, y2), (10, 10, 10), -1)
                cv2.putText(canvas, "AUTO", (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35*scale, (0, 220, 255), 1)

            elif e["type"] == "motorcycle":
                # Motorcyclist
                bw = int(45 * scale)
                bh = int(80 * scale)
                x1, y1 = cx - bw // 2, cy - bh
                x2, y2 = cx + bw // 2, cy
                # Rider head & helmet
                cv2.circle(canvas, (cx, y1 + int(14*scale)), int(9*scale), (220, 40, 40), -1)
                # Body
                cv2.rectangle(canvas, (cx - int(10*scale), y1 + int(24*scale)), 
                              (cx + int(10*scale), y2 - int(15*scale)), (50, 120, 180), -1)
                # Headlight (shining toward camera)
                cv2.circle(canvas, (cx, y2 - int(14*scale)), int(7*scale), (255, 255, 200), -1)
                # Tires
                cv2.rectangle(canvas, (cx - int(6*scale), y2 - int(12*scale)), (cx + int(6*scale), y2), (10, 10, 10), -1)

            else:
                # Standard Car
                bw = int(90 * scale)
                bh = int(115 * scale)
                x1, y1 = cx - bw // 2, cy - bh
                x2, y2 = cx + bw // 2, cy
                cv2.rectangle(canvas, (x1, y1), (x2, y2), e["color"], -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 2)
                # Windshield
                cv2.rectangle(canvas, (x1 + int(10*scale), y1 + int(bh*0.2)), 
                              (x2 - int(10*scale), y1 + int(bh*0.55)), (40, 40, 40), -1)
                # Tail lights
                cv2.rectangle(canvas, (x1 + 6, y2 - 10), (x1 + 16, y2 - 4), (0, 0, 255), -1)
                cv2.rectangle(canvas, (x2 - 16, y2 - 10), (x2 - 6, y2 - 4), (0, 0, 255), -1)

        out.write(canvas)

    out.release()
    print(f"[+] Successfully generated '{output_filename}' ({num_frames} frames, {width}x{height} @ {fps} fps).")


if __name__ == "__main__":
    create_indian_traffic_sample()
