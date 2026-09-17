"""MediaPipe Face Landmarker wrapper."""
from __future__ import annotations

import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as lm_module


@dataclass
class FaceResult:
    landmarks: list                          # NormalizedLandmark list (478 points)
    blendshapes: dict[str, float] = field(default_factory=dict)
    transform_matrix: np.ndarray | None = None  # 4×4 facial transformation matrix
    present: bool = True


class FaceDetector:
    """Thin wrapper around MediaPipe FaceLandmarker (IMAGE mode)."""

    def __init__(self, model_path: str, num_faces: int = 1,
                 detection_conf: float = 0.5, presence_conf: float = 0.5,
                 tracking_conf: float = 0.5) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=detection_conf,
            min_face_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr_frame: np.ndarray) -> Optional[FaceResult]:
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(bgr_frame[:, :, ::-1]),
        )
        result = self._landmarker.detect(rgb)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {
                bs.category_name: bs.score
                for bs in result.face_blendshapes[0]
            }
        matrix: np.ndarray | None = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
        return FaceResult(landmarks=landmarks, blendshapes=blendshapes, transform_matrix=matrix)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
