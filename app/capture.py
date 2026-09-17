"""Video capture abstraction — webcam or file."""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator


@dataclass
class FrameResult:
    frame: np.ndarray
    frame_index: int
    timestamp_ms: float


class VideoCapture:
    def __init__(self, source: int | str, width: int = 640, height: int = 480) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._index = 0

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def frames(self) -> Generator[FrameResult, None, None]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            yield FrameResult(frame=frame, frame_index=self._index, timestamp_ms=ts)
            self._index += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoCapture":
        return self

    def __exit__(self, *_) -> None:
        self.release()
