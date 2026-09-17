"""Mouth Aspect Ratio (MAR) — yawn detection.

MediaPipe 478-landmark indices used (outer lip):
  Top    : 13
  Bottom : 14
  Left   : 78
  Right  : 308
  Mid-top-left  : 82
  Mid-top-right : 312
  Mid-bot-left  : 87
  Mid-bot-right : 317

MAR = (|p_top - p_bot| + |p_mid_tl - p_mid_bl| + |p_mid_tr - p_mid_br|)
      / (3 * |p_left - p_right|)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


_MOUTH_IDX = (13, 14, 78, 308, 82, 87, 312, 317)
#              top bot lft rgt mtl mbl mtr mbr


def compute_mar(landmarks: list) -> float:
    top, bot, lft, rgt, mtl, mbl, mtr, mbr = (
        np.array([landmarks[i].x, landmarks[i].y]) for i in _MOUTH_IDX
    )
    vertical = (np.linalg.norm(top - bot)
                + np.linalg.norm(mtl - mbl)
                + np.linalg.norm(mtr - mbr))
    horizontal = np.linalg.norm(lft - rgt)
    return float(vertical / (3.0 * horizontal + 1e-6))


@dataclass
class MouthState:
    mar: float = 0.0
    yawn_count: int = 0
    yawning: bool = False
    _open_frames: int = field(default=0, repr=False)
    _yawn_in_progress: bool = field(default=False, repr=False)


class MouthAnalyzer:
    """Stateful yawn tracker using MAR."""

    def __init__(self, mar_threshold: float = 0.65, yawn_frames: int = 20) -> None:
        self.mar_threshold = mar_threshold
        self.yawn_frames = yawn_frames
        self._state = MouthState()

    def update(self, landmarks: list) -> MouthState:
        mar = compute_mar(landmarks)
        s = self._state
        s.mar = round(mar, 4)

        if mar > self.mar_threshold:
            s._open_frames += 1
            if s._open_frames >= self.yawn_frames:
                s.yawning = True
                s._yawn_in_progress = True
        else:
            if s._yawn_in_progress:
                s.yawn_count += 1
            s._open_frames = 0
            s.yawning = False
            s._yawn_in_progress = False

        return s

    def reset(self) -> None:
        self._state = MouthState()
