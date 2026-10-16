"""
LATIS Indian Road Conditions Fine-Tuning Pipeline.
Trains/Fine-tunes YOLOv8 on Indian traffic conditions:
- Auto-Rickshaws (3-wheelers)
- Cattle / Cows & Stray Animals on road
- Street Vendors with Pushcarts / Thelas
- Potholes & Muddy road conditions
- Pedestrians & Two-wheeler swarms
"""

import os
import sys
import argparse
import yaml
import shutil
import cv2
import numpy as np
from ultralytics import YOLO


INDIAN_CLASSES = [
    "car",
    "motorcycle",
    "auto_rickshaw",
    "bus",
    "truck",
    "cow_cattle",
    "stray_dog",
    "street_vendor_pushcart",
    "pedestrian",
    "pothole"
]


def create_demo_dataset(dataset_dir="datasets/indian_roads_demo"):
    """
    Generates a starter dataset structure with sample annotated images
    to demonstrate the fine-tuning pipeline immediately.
    """
    print(f"[+] Creating starter Indian Road Conditions dataset in '{dataset_dir}'...")
    images_train = os.path.join(dataset_dir, "images", "train")
    labels_train = os.path.join(dataset_dir, "labels", "train")
    images_val = os.path.join(dataset_dir, "images", "val")
    labels_val = os.path.join(dataset_dir, "labels", "val")

    os.makedirs(images_train, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # Create 12 realistic synthetic training frames and 4 val frames
    for split, img_dir, lbl_dir, count in [("train", images_train, labels_train, 12), ("val", images_val, labels_val, 4)]:
        for i in range(count):
            img_name = f"indian_road_{split}_{i:03d}.jpg"
            lbl_name = f"indian_road_{split}_{i:03d}.txt"

            # Create an Indian road scene with mud, side vegetation, road surface
            img = np.full((640, 640, 3), (45, 100, 45), dtype=np.uint8)  # Roadside bushes
            
            # Asphalt road with dirt/mud shoulders
            road_pts = np.array([[80, 640], [560, 640], [420, 150], [220, 150]], dtype=np.int32)
            cv2.fillPoly(img, [road_pts], (60, 60, 60))

            # Mud patches on road edge
            mud_pts = np.array([[80, 640], [180, 640], [250, 400], [130, 420]], dtype=np.int32)
            cv2.fillPoly(img, [mud_pts], (40, 65, 85))

            # Faded/worn centerline
            cv2.line(img, (320, 150), (320, 640), (180, 180, 160), 2)

            annotations = []

            # 1. Cow sitting in lane (class 5: cow_cattle)
            cow_cx, cow_cy = 280, 380
            cv2.rectangle(img, (cow_cx - 40, cow_cy - 25), (cow_cx + 40, cow_cy + 25), (200, 200, 200), -1)
            cv2.putText(img, "COW", (cow_cx - 20, cow_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
            # YOLO format: class cx cy w h (normalized 0-1)
            annotations.append(f"5 {cow_cx/640:.4f} {cow_cy/640:.4f} {80/640:.4f} {50/640:.4f}")

            # 2. Auto-rickshaw (class 2: auto_rickshaw)
            auto_cx, auto_cy = 400, 450
            cv2.rectangle(img, (auto_cx - 35, auto_cy - 40), (auto_cx + 35, auto_cy + 40), (0, 215, 255), -1)  # Yellow
            cv2.rectangle(img, (auto_cx - 35, auto_cy - 10), (auto_cx + 35, auto_cy + 40), (20, 120, 20), -1)  # Green bottom
            annotations.append(f"2 {auto_cx/640:.4f} {auto_cy/640:.4f} {70/640:.4f} {80/640:.4f}")

            # 3. Street Vendor / Thela on wheels (class 7: street_vendor_pushcart)
            vendor_cx, vendor_cy = 160, 520
            cv2.rectangle(img, (vendor_cx - 45, vendor_cy - 30), (vendor_cx + 45, vendor_cy + 30), (80, 50, 130), -1)
            cv2.circle(img, (vendor_cx - 35, vendor_cy + 30), 12, (20, 20, 20), -1)  # Cart wheel
            cv2.circle(img, (vendor_cx + 35, vendor_cy + 30), 12, (20, 20, 20), -1)  # Cart wheel
            annotations.append(f"7 {vendor_cx/640:.4f} {vendor_cy/640:.4f} {90/640:.4f} {60/640:.4f}")

            # 4. Pothole on road surface (class 9: pothole)
            pot_cx, pot_cy = 340, 560
            cv2.ellipse(img, (pot_cx, pot_cy), (30, 15), 0, 0, 360, (25, 25, 25), -1)
            annotations.append(f"9 {pot_cx/640:.4f} {pot_cy/640:.4f} {60/640:.4f} {30/640:.4f}")

            # Save image and label
            cv2.imwrite(os.path.join(img_dir, img_name), img)
            with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                f.write("\n".join(annotations))

    # Create dataset YAML
    data_yaml_path = os.path.join(dataset_dir, "indian_traffic.yaml")
    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(INDIAN_CLASSES)}
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[+] Dataset configured at: {data_yaml_path}")
    return data_yaml_path


def train_indian_model(data_yaml: str, epochs: int = 5, batch_size: int = 4, 
                       base_weights: str = "yolov8n.pt", img_size: int = 640):
    """
    Fine-tunes YOLO on Indian traffic data with domain-specific augmentations:
    - High mosaic and mixup (cluttered mixed traffic)
    - Random perspective & scale (varying camera mount angles)
    - HSV-V jitter (monsoon haze, headlight glare, bright sun)
    """
    print("\n" + "=" * 65)
    print("      LATIS MODEL FINE-TUNING: INDIAN ROAD CONDITIONS     ")
    print("=" * 65)
    print(f"  Base Pre-trained Weights : {base_weights}")
    print(f"  Dataset Configuration    : {data_yaml}")
    print(f"  Target Classes ({len(INDIAN_CLASSES)})    : {', '.join(INDIAN_CLASSES)}")
    print(f"  Epochs                   : {epochs}")
    print(f"  Batch Size               : {batch_size}")
    print("=" * 65 + "\n")

    model = YOLO(base_weights)

    # Domain-specific augmentation parameters for Indian roads
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project="runs/indian_roads",
        name="latis_indian_traffic",
        # Indian Road Domain Augmentations:
        hsv_h=0.015,   # Hue variance
        hsv_s=0.7,     # High saturation variance (colorful vehicles & vendors)
        hsv_v=0.4,     # Brightness variance (shadows under flyovers & direct sunlight)
        degrees=5.0,   # Mild tilt for camera sway
        translate=0.1, # Translation
        scale=0.5,     # Large vehicles (trucks) to tiny cattle/dogs
        mosaic=1.0,    # Dense multi-object scenes
        mixup=0.15,    # Occlusion simulation (dust/mud)
        erasing=0.2,   # Mud splatter simulation on vehicles/pavement
        verbose=True
    )

    best_weights = os.path.join(str(results.save_dir), "weights", "best.pt")
    if os.path.exists(best_weights):
        target_path = "indian_traffic_yolo.pt"
        shutil.copy(best_weights, target_path)
        print("\n" + "=" * 65)
        print(f"[+] Fine-tuning Complete! Best model exported to: {os.path.abspath(target_path)}")
        print(f"[+] You can now use this model in your pipeline by updating:")
        print(f"    perception.model_weights: '{target_path}'")
        print("=" * 65 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Indian Road Conditions")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Generate a demo dataset and run training loop immediately")
    parser.add_argument("--data", type=str, default="",
                        help="Path to external custom data.yaml (e.g. from DriveIndia or IDD)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--base", type=str, default="yolov8n.pt",
                        help="Base pre-trained YOLO checkpoint")
    args = parser.parse_args()

    data_yaml = args.data
    if not data_yaml or args.demo:
        data_yaml = create_demo_dataset()

    train_indian_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        base_weights=args.base
    )


if __name__ == "__main__":
    main()
