"""Head pose estimation and attention direction from MediaPipe transform matrix.

MediaPipe's facial_transformation_matrixes gives a 4x4 model-space matrix.
We extract Euler angles (pitch, yaw, roll) via cv2.Rodrigues on the 3x3
rotation sub-matrix.

Attention is classified as:
  - FORWARD   : pitch and yaw within configured ranges
  - UP/DOWN   : pitch outside range
  - LEFT/RIGHT: yaw outside range
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum


class Attention(str, Enum):
    FORWARD = "Forward"
    UP      = "Up"
    DOWN    = "Down"
    LEFT    = "Left"
    RIGHT   = "Right"
    UNKNOWN = "Unknown"


@dataclass
class HeadPoseState:
    pitch: float = 0.0   # degrees, + = looking down
    yaw:   float = 0.0   # degrees, + = looking right
    roll:  float = 0.0   # degrees
    attention: Attention = Attention.UNKNOWN
    on_screen_ratio: float = 0.0   # fraction of frames looking forward (rolling)


class HeadPoseAnalyzer:
    def __init__(
        self,
        pitch_range: tuple[float, float] = (-20.0, 20.0),
        yaw_range:   tuple[float, float] = (-30.0, 30.0),
    ) -> None:
        self.pitch_range = pitch_range
        self.yaw_range   = yaw_range
        self._total   = 0
        self._forward = 0
        self._state   = HeadPoseState()

    def update(self, transform_matrix: np.ndarray) -> HeadPoseState:
        pitch, yaw, roll = _euler_from_matrix(transform_matrix)
        attention = _classify(pitch, yaw, self.pitch_range, self.yaw_range)

        self._total += 1
        if attention == Attention.FORWARD:
            self._forward += 1

        self._state = HeadPoseState(
            pitch=round(pitch, 1),
            yaw=round(yaw, 1),
            roll=round(roll, 1),
            attention=attention,
            on_screen_ratio=round(self._forward / self._total, 3),
        )
        return self._state

    def reset(self) -> None:
        self._total = self._forward = 0
        self._state = HeadPoseState()


# ── helpers ──────────────────────────────────────────────────────────────────

def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Extract pitch, yaw, roll (degrees) from a 4x4 transform matrix."""
    rot = m[:3, :3].astype(np.float64)
    # Rodrigues → rotation vector → convert to degrees
    rvec, _ = cv2.Rodrigues(rot)
    pitch = float(np.degrees(rvec.flatten()[0]))
    yaw   = float(np.degrees(rvec.flatten()[1]))
    roll  = float(np.degrees(rvec.flatten()[2]))
    return pitch, yaw, roll


def _classify(
    pitch: float, yaw: float,
    pitch_range: tuple[float, float],
    yaw_range:   tuple[float, float],
) -> Attention:
    p_lo, p_hi = pitch_range
    y_lo, y_hi = yaw_range

    if not (p_lo <= pitch <= p_hi) :
        return Attention.DOWN if pitch > p_hi else Attention.UP
    if not (y_lo <= yaw <= y_hi):
        return Attention.RIGHT if yaw > y_hi else Attention.LEFT
    return Attention.FORWARD


def draw_pose_axes(
    frame: np.ndarray,
    transform_matrix: np.ndarray,
    origin_landmark,
    axis_len: float = 0.08,
) -> None:
    """Draw RGB XYZ axes on the nose tip to visualise head orientation."""
    h, w = frame.shape[:2]
    origin = np.array([origin_landmark.x * w, origin_landmark.y * h], dtype=np.float32)

    rot = transform_matrix[:3, :3].astype(np.float64)
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.zeros((3, 1), dtype=np.float64)

    # camera intrinsics approximation
    focal = w
    cam_matrix = np.array([[focal, 0, w / 2],
                            [0, focal, h / 2],
                            [0,     0,     1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    axes_3d = np.float32([[axis_len, 0, 0],
                           [0, axis_len, 0],
                           [0, 0, axis_len]])
    pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam_matrix, dist)
    pts = pts.reshape(-1, 2).astype(int)
    o   = tuple(origin.astype(int))

    cv2.line(frame, o, tuple(pts[0]), (0, 0, 255), 2)   # X — red
    cv2.line(frame, o, tuple(pts[1]), (0, 255, 0), 2)   # Y — green
    cv2.line(frame, o, tuple(pts[2]), (255, 0, 0), 2)   # Z — blue
