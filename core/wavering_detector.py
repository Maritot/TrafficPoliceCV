"""
Project WAV-DET - Kinematic Wavering & Erratic Behavior Detection Engine.
Extracts mathematical trajectory features (lateral drift, spectral zero-crossings,
high-G lateral swerves, heading variance, and speed stability) to classify
erratic driving with calibrated severity and detailed justifications.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WaveringAssessment:
    is_wavering: bool
    severity: str           # "NORMAL" | "INFO" | "MODERATE" | "SEVERE"
    reason: str             # Specific human-readable diagnosis
    confidence: float       # Calibrated confidence 0.0 to 1.0
    metric_amplitude_m: float
    oscillation_count: int
    peak_lateral_accel_m_s2: float
    heading_variance_deg: float
    speed_cv: float


class WaveringDetector:
    """
    Evaluates vehicle trajectory histories using kinematic regression,
    spectral zero-crossings, and dynamical stability metrics.
    """

    def __init__(self, window_seconds: float = 3.5, min_samples: int = 8):
        self.window_seconds = window_seconds
        self.min_samples = min_samples

    def analyze_trajectory(self, 
                           history_ground: List[Tuple[float, float, float]], 
                           class_name: str, 
                           current_speed_kmh: float,
                           zone: str = "CARRIAGEWAY",
                           is_agricultural: bool = False) -> WaveringAssessment:
        """
        Analyzes a vehicle's ground trajectory [(x, y, t)] over the time window.
        
        Args:
            history_ground: List of (ground_x_meters, ground_y_meters, timestamp_seconds)
            class_name: Vehicle class name
            current_speed_kmh: Vehicle speed in km/h
            zone: "CARRIAGEWAY" | "SHOULDER" | "OFF_ROAD"
            is_agricultural: True if tractor, harvester, or farm machinery
        """
        # Baseline normal return for insufficient trajectory history
        normal_return = WaveringAssessment(
            is_wavering=False,
            severity="NORMAL",
            reason="Stable trajectory within lane boundaries",
            confidence=0.95,
            metric_amplitude_m=0.0,
            oscillation_count=0,
            peak_lateral_accel_m_s2=0.0,
            heading_variance_deg=0.0,
            speed_cv=0.0
        )

        if len(history_ground) < self.min_samples:
            return normal_return

        # Context Filter 1: Off-Road Vehicles & Agricultural Machinery
        # Vehicles off the carriageway (e.g. tractor in field, parked car on shoulder) are never wavering
        if zone == "OFF_ROAD":
            return WaveringAssessment(
                is_wavering=False,
                severity="NORMAL",
                reason=f"{class_name} operating off-road (field/ditch) - exempt from carriageway discipline",
                confidence=0.98,
                metric_amplitude_m=0.0,
                oscillation_count=0,
                peak_lateral_accel_m_s2=0.0,
                heading_variance_deg=0.0,
                speed_cv=0.0
            )

        if is_agricultural and current_speed_kmh < 30.0:
            return WaveringAssessment(
                is_wavering=False,
                severity="NORMAL",
                reason=f"Agricultural vehicle ({class_name}) operating at normal low speed ({current_speed_kmh:.0f} km/h)",
                confidence=0.95,
                metric_amplitude_m=0.0,
                oscillation_count=0,
                peak_lateral_accel_m_s2=0.0,
                heading_variance_deg=0.0,
                speed_cv=0.0
            )

        # Context Filter 2: Stationary / Stopped Vehicles (< 3 km/h)
        if current_speed_kmh < 4.0:
            return normal_return

        # Extract recent trajectory within sliding time window
        t_latest = history_ground[-1][2]
        window_pts = [pt for pt in history_ground if (t_latest - pt[2]) <= self.window_seconds]

        if len(window_pts) < self.min_samples:
            return normal_return

        xs = np.array([pt[0] for pt in window_pts])
        ys = np.array([pt[1] for pt in window_pts])
        ts = np.array([pt[2] for pt in window_pts])

        # Feature 1: Linear regression to establish nominal trajectory centerline
        # In a stable lane travel, vehicle follows a line/curve in (x, y) space
        if len(np.unique(ys)) > 2:
            poly_fit = np.polyfit(ys, xs, deg=1)
            nominal_xs = np.polyval(poly_fit, ys)
            lateral_deviations = xs - nominal_xs  # Lateral drift in meters
        else:
            lateral_deviations = xs - np.mean(xs)

        amp_m = float(np.max(np.abs(lateral_deviations)))

        # Feature 2: Zero-Crossing / Sign-Flip Rate (Lateral Oscillation)
        # Smooth deviations with moving average to eliminate sensor noise
        smoothed_devs = np.convolve(lateral_deviations, np.ones(3)/3, mode='valid')
        diffs = np.diff(smoothed_devs)
        
        # Count inflection points where lateral movement changes direction by > 0.35m
        sign_flips = 0
        last_peak = smoothed_devs[0]
        direction = 0  # 1 for increasing, -1 for decreasing

        for i in range(1, len(smoothed_devs)):
            diff = smoothed_devs[i] - smoothed_devs[i-1]
            if abs(diff) > 0.05:
                curr_dir = 1 if diff > 0 else -1
                if direction != 0 and curr_dir != direction:
                    if abs(smoothed_devs[i] - last_peak) > 0.40:
                        sign_flips += 1
                        last_peak = smoothed_devs[i]
                direction = curr_dir

        # Feature 3: Lateral Acceleration & Jerk
        dts = np.diff(ts)
        valid_dts = np.where(dts > 0.01, dts, 0.02)
        v_lateral = np.diff(lateral_deviations) / valid_dts
        
        if len(v_lateral) > 1:
            dt2 = (valid_dts[:-1] + valid_dts[1:]) / 2.0
            a_lateral = np.diff(v_lateral) / np.where(dt2 > 0.01, dt2, 0.02)
            peak_lat_accel = float(np.max(np.abs(a_lateral)))
        else:
            peak_lat_accel = 0.0

        # Feature 4: Heading Variance along trajectory
        dxs = np.diff(xs)
        dys = np.diff(ys)
        headings = np.arctan2(dxs, np.where(np.abs(dys) > 1e-4, dys, 1e-4))
        heading_deg = np.degrees(headings)
        heading_var = float(np.std(heading_deg)) if len(heading_deg) > 2 else 0.0

        # Feature 5: Speed Instability (Coefficient of Variation)
        step_distances = np.hypot(dxs, dys)
        speeds = (step_distances / valid_dts) * 3.6  # km/h
        mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else current_speed_kmh
        std_speed = float(np.std(speeds)) if len(speeds) > 0 else 0.0
        speed_cv = float(std_speed / max(mean_speed, 1.0))

        # -------------------------------------------------------------
        # CALIBRATED DECISION RULE HIERARCHY (High Precision, Low Alarm)
        # -------------------------------------------------------------

        # Case 1: Repeated Lane Weaving / Wavering (Zig-zagging lateral oscillations)
        # >= 3 direction changes with substantial lateral amplitude (> 0.65m)
        if sign_flips >= 3 and amp_m >= 0.65:
            severity = "SEVERE" if (amp_m >= 1.6 or sign_flips >= 4 or peak_lat_accel >= 6.0) else "MODERATE"
            conf = min(0.96, 0.65 + (sign_flips * 0.08) + (amp_m * 0.10))
            return WaveringAssessment(
                is_wavering=True,
                severity=severity,
                reason=f"Repeated lane drift & weaving ({sign_flips} direction changes, peak deviation {amp_m:.1f}m across {self.window_seconds:.1f}s)",
                confidence=round(conf, 2),
                metric_amplitude_m=round(amp_m, 2),
                oscillation_count=sign_flips,
                peak_lateral_accel_m_s2=round(peak_lat_accel, 2),
                heading_variance_deg=round(heading_var, 1),
                speed_cv=round(speed_cv, 2)
            )

        # Case 2: Severe Acute Swerve (Emergency or reckless sudden lateral jerk)
        # Peak lateral acceleration >= 3.2 m/s2 while moving at highway speed (> 30 km/h)
        if peak_lat_accel >= 3.2 and current_speed_kmh >= 30.0 and amp_m >= 0.70:
            conf = min(0.96, 0.72 + (peak_lat_accel / 15.0))
            return WaveringAssessment(
                is_wavering=True,
                severity="SEVERE",
                reason=f"Sudden high-G lateral swerve ({peak_lat_accel:.1f} m/s² peak jerk, {amp_m:.1f}m lateral displacement)",
                confidence=round(conf, 2),
                metric_amplitude_m=round(amp_m, 2),
                oscillation_count=sign_flips,
                peak_lateral_accel_m_s2=round(peak_lat_accel, 2),
                heading_variance_deg=round(heading_var, 1),
                speed_cv=round(speed_cv, 2)
            )

        # Case C: Moderate Unstable Drift (Heading wobble with single broad lateral straddle)
        if sign_flips == 2 and amp_m >= 1.4 and heading_var >= 14.0:
            return WaveringAssessment(
                is_wavering=True,
                severity="MODERATE",
                reason=f"Persistent lane straddling & heading instability ({amp_m:.1f}m drift, {heading_var:.1f}° heading variance)",
                confidence=0.82,
                metric_amplitude_m=round(amp_m, 2),
                oscillation_count=sign_flips,
                peak_lateral_accel_m_s2=round(peak_lat_accel, 2),
                heading_variance_deg=round(heading_var, 1),
                speed_cv=round(speed_cv, 2)
            )

        # Case D: Minor Informational Wobble / Speed Surging
        if speed_cv >= 0.55 and current_speed_kmh >= 25.0 and amp_m >= 0.8:
            return WaveringAssessment(
                is_wavering=True,
                severity="INFO",
                reason=f"Unstable velocity & throttle surging (speed CV: {speed_cv:.2f}, lateral drift: {amp_m:.1f}m)",
                confidence=0.74,
                metric_amplitude_m=round(amp_m, 2),
                oscillation_count=sign_flips,
                peak_lateral_accel_m_s2=round(peak_lat_accel, 2),
                heading_variance_deg=round(heading_var, 1),
                speed_cv=round(speed_cv, 2)
            )

        # All criteria within safe highway tolerance
        return normal_return
