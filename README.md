# VĀHAN-LATIS: National Intelligent Traffic Management System (ITMS)
### AI-Powered Fixed-Camera Traffic Analytics, Cloud CCTV Monitoring & Forensic Intelligence Portal

Developed in conformance with the **Ministry of Road Transport & Highways (MoRTH)** and **Indian Road Congress (IRC:SP:110-2015)** Intelligent Transport Systems (ITS) standards.

---

## 🏛️ System Overview

**VĀHAN-LATIS** is a production-oriented, modular intelligent traffic surveillance and forensic audit platform. It bridges deep learning perception with rigorous projective geometry and deterministic road safety mechanics:

1. **Planar DLT Homography Calibration:** Maps 2D camera pixels $(u, v)$ to real-world ground coordinates $(X, Y)$ in meters with $< 0.15\text{ m}$ reprojection error.
2. **Curvilinear Frenet Road Geometry:** Resolves longitudinal progress ($s$) and lateral lane displacement ($d$) with normalized lane-boundary violation metrics.
3. **Multi-Class Detection & ByteTrack Ground Contact Association:** Custom tire-road contact point projection with strict 1-to-1 frame association and metric lateral lane gating ($< 0.85\text{ m}$) to eliminate ID-switch aliasing.
4. **Indian Mixed-Traffic Condition Perception:** Custom fine-tuned weights for Cattle/Livestock, Street Vendor Mobile Pushcarts, Pedestrians in travel lanes, and Pothole/Mud evasive swerves.
5. **Headless Computer Vision REST API (Port 5000):** Pure Python backend streaming real-time MJPEG frames with dynamic 2D Bird's-Eye View (BEV) radar projections and JSON telemetry.
6. **Government-Grade Next.js 14 Frontend Portal (Port 3000):** Uncluttered, minimalist, high-accessibility UI with two dedicated modules: **Live CCTV Monitoring** and **Video Analysis & Audit**.

---

## 📂 Project Architecture

```text
TrafficPoliceCV/
├── backend/
│   └── api.py                    # Headless Python REST & MJPEG streaming API (Port 5000)
├── config/
│   ├── default_config.yaml       # Default 720p perspective configuration
│   ├── highway_config.yaml       # Calibrated parameters for NH-48 Expressway (video1.avi)
│   └── indian_road_config.yaml   # Parameters for Indian mixed road hazards
├── core/
│   ├── calibration.py            # Homography, camera matrix & perspective projection
│   ├── geometry.py               # Road corridors, centerlines, Frenet frame (s, d)
│   ├── detector_tracker.py       # YOLOv8 + ByteTrack 1-to-1 metric track stitcher
│   ├── behavior_engine.py        # Deterministic safety rule engine (Wrong-way, Weaving, Cattle, Vendors)
│   └── visualizer.py             # Super-resolution video overlays & 2D BEV radar
├── frontend/
│   ├── app/                      # Next.js 14 App Router (page.tsx, layout.tsx, globals.css)
│   ├── components/
│   │   ├── Navbar.tsx            # Minimalist government portal header & module tabs
│   │   ├── LiveMonitoring.tsx    # Module 1: Cloud CCTV Grid & Live AI Stream
│   │   ├── VideoAnalysis.tsx     # Module 2: Forensic Video Ingestion & Audit Register
│   │   └── ChallanModal.tsx      # Statutory e-Challan legal notice generator (MVA 2019)
│   └── public/videos/            # Demo videos for instant forensic playback
├── utils/
│   ├── lane_authoring_tool.py    # Point-and-click OpenCV calibration & lane boundary GUI
│   ├── generate_sample_video.py  # Synthetic traffic video generator
│   └── generate_indian_traffic_sample.py
├── run_latis.py                  # Standalone CLI execution pipeline
├── video1.avi                    # Real highway test video (500 frames)
├── indian_traffic_sample.mp4     # Indian mixed traffic test video
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
* Python 3.10+
* Node.js 18+ & npm

### 2. Python Backend Setup
```powershell
# Install Python computer vision dependencies
pip install ultralytics opencv-python pyyaml numpy scipy flask

# Start the headless Computer Vision API on Port 5000
python backend/api.py
```

### 3. Next.js Frontend Setup
```powershell
cd frontend

# Install Node dependencies
npm install

# Start the Next.js portal on Port 3000
npm run dev
# Or build and launch production server:
npm run build
npm run start
```

Open **`http://localhost:3000`** in your browser.

---

## ⚖️ Statutory Legal Compliance (Motor Vehicles Act 2019)
* **Section 183:** Driving at excessive speed (Speed limit enforcement)
* **Section 184:** Dangerous / reckless driving (Erratic lane weaving)
* **Section 177A:** Violation of mandatory lane discipline markings
* **Section 283 IPC / Section 201 MVA:** Unauthorized road obstruction by street vendor carts
* **Section 289 IPC:** Cattle / livestock hazard on expressways

---

## 📄 License
Academic & Research Project under the Intelligent Transportation Systems (ITS) Initiative.
