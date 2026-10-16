"""
Stage 0: Metric Ground-Plane Homography Calibration Engine.
Translates pixel perspective coordinates into metric ground-plane coordinates (meters)
and vice versa using four-point projective planar transformation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Union


class HomographyCalibrator:
    def __init__(self, image_quad: List[List[float]], ground_quad: List[List[float]]):
        """
        Initialize calibrator with 4 matching image and ground control points.
        :param image_quad: 4 points in image space [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        :param ground_quad: 4 points in metric space [[X0,Y0], [X1,Y1], [X2,Y2], [X3,Y3]] in meters
        """
        self.src_pts = np.array(image_quad, dtype=np.float32)
        self.dst_pts = np.array(ground_quad, dtype=np.float32)

        if len(self.src_pts) != 4 or len(self.dst_pts) != 4:
            raise ValueError("Exactly 4 points are required for planar homography.")

        # Compute Homography: Image -> Ground (metric)
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        # Compute Inverse Homography: Ground -> Image
        self.H_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

        # Calculate reprojection error on the calibration points
        self.mean_reproj_error_m = self._compute_reprojection_error()

    def _compute_reprojection_error(self) -> float:
        """Compute the root-mean-square reprojection residual in meters."""
        projected = self.project_image_to_ground(self.src_pts)
        errors = np.linalg.norm(projected - self.dst_pts, axis=1)
        return float(np.mean(errors))

    def project_image_to_ground(self, points: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Transform (u, v) image coordinates to (X, Y) metric ground coordinates in meters.
        :param points: (N, 2) array of pixel coordinates
        :return: (N, 2) array of metric ground coordinates
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float32))
        homogeneous_pts = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])
        
        # Matrix multiplication: [X_h, Y_h, W] = H * [u, v, 1]^T
        ground_h = np.dot(self.H, homogeneous_pts.T).T
        
        # Normalize by scale factor W
        eps = 1e-8
        W = ground_h[:, 2:3]
        W[np.abs(W) < eps] = eps
        ground_pts = ground_h[:, :2] / W
        return ground_pts

    def project_ground_to_image(self, points: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Transform (X, Y) metric ground coordinates to (u, v) image pixel coordinates.
        :param points: (N, 2) array of metric ground coordinates
        :return: (N, 2) array of pixel coordinates
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float32))
        homogeneous_pts = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])
        
        # Matrix multiplication: [u_h, v_h, W] = H_inv * [X, Y, 1]^T
        img_h = np.dot(self.H_inv, homogeneous_pts.T).T
        
        # Normalize by scale factor W
        eps = 1e-8
        W = img_h[:, 2:3]
        W[np.abs(W) < eps] = eps
        img_pts = img_h[:, :2] / W
        return img_pts

    def get_ground_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) in metric space."""
        min_x = float(np.min(self.dst_pts[:, 0]))
        max_x = float(np.max(self.dst_pts[:, 0]))
        min_y = float(np.min(self.dst_pts[:, 1]))
        max_y = float(np.max(self.dst_pts[:, 1]))
        return min_x, max_x, min_y, max_y
