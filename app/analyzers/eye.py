"""Eye Aspect Ratio (EAR) — blink and prolonged-closure detection.

MediaPipe 478-landmark indices used:
  Left eye  : 362, 385, 387, 263, 373, 380
  Right eye : 33,  160, 158, 133, 153, 144
  (same 6-point layout as dlib: P1..P6 around each eye)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# (p1,p2,p3,p4,p5,p6) for each eye — vertical pairs: (p2,p6),(p3,p5); horizontal: (p1,p4)
_LEFT_IDX  = (362, 385, 387, 263, 373, 380)
_RIGHT_IDX = (33,  160, 158, 133, 153, 144)


def _ear(landmarks: list, indices: tuple[int, ...]) -> float:
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal + 1e-6))


@dataclass
class EyeState:
    ear: float = 0.0
    blink_count: int = 0
    prolonged_closure: bool = False
    # internal
    _closed_frames: int = field(default=0, repr=False)
    _blink_in_progress: bool = field(default=False, repr=False)


class EyeAnalyzer:
    """Stateful per-session blink tracker."""

    def __init__(self, ear_threshold: float = 0.20, closed_frames: int = 15) -> None:
        self.ear_threshold = ear_threshold
        self.closed_frames = closed_frames
        self._state = EyeState()

    def update(self, landmarks: list) -> EyeState:
        left  = _ear(landmarks, _LEFT_IDX)
        right = _ear(landmarks, _RIGHT_IDX)
        avg   = (left + right) / 2.0

        s = self._state
        s.ear = round(avg, 4)

        if avg < self.ear_threshold:
            s._closed_frames += 1
            s._blink_in_progress = True
            s.prolonged_closure = s._closed_frames >= self.closed_frames
        else:
            if s._blink_in_progress:
                # eye just re-opened → completed blink (not prolonged closure)
                if s._closed_frames < self.closed_frames:
                    s.blink_count += 1
            s._closed_frames = 0
            s._blink_in_progress = False
            s.prolonged_closure = False

        return s

    def reset(self) -> None:
        self._state = EyeState()
