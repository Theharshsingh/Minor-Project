"""Participation analysis — speaking, nodding, face-presence ratio.

Signals derived purely from observable behaviour:
  - speaking      : jaw open blendshape oscillates above threshold
  - nodding       : pitch angle alternates sign within a short window
  - face_present  : rolling ratio of frames where a face was detected
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ParticipationState:
    speaking: bool = False
    nodding: bool = False
    face_present_ratio: float = 0.0   # rolling window
    speaking_ratio: float = 0.0       # rolling window
    nod_count: int = 0


class ParticipationAnalyzer:
    """
    Parameters
    ----------
    fps              : frames per second (used to size rolling windows)
    speaking_window  : seconds of jaw history to inspect
    jaw_threshold    : jawOpen blendshape score to count as mouth-open
    jaw_open_ratio   : fraction of window frames that must be open → speaking
    nod_window       : seconds of pitch history for nod detection
    nod_min_reversal : minimum pitch sign reversals in window → nod
    presence_window  : seconds for face-present rolling ratio
    """

    def __init__(
        self,
        fps: float = 30.0,
        speaking_window: float = 1.0,
        jaw_threshold: float = 0.25,
        jaw_open_ratio: float = 0.30,
        nod_window: float = 2.0,
        nod_min_reversal: int = 2,
        presence_window: float = 10.0,
    ) -> None:
        self._jaw_thr      = jaw_threshold
        self._jaw_ratio    = jaw_open_ratio
        self._nod_min_rev  = nod_min_reversal

        win_jaw  = max(1, int(fps * speaking_window))
        win_nod  = max(1, int(fps * nod_window))
        win_pres = max(1, int(fps * presence_window))

        self._jaw_buf:      deque[float] = deque(maxlen=win_jaw)
        self._pitch_buf:    deque[float] = deque(maxlen=win_nod)
        self._present_buf:  deque[int]   = deque(maxlen=win_pres)

        self._state = ParticipationState()
        self._prev_nod_speaking = False   # edge-detect nod from speaking

    def update(
        self,
        blendshapes: dict[str, float],
        pitch: float,
        face_present: bool,
    ) -> ParticipationState:
        s = self._state

        # ── face presence ────────────────────────────────────────────────────
        self._present_buf.append(1 if face_present else 0)
        s.face_present_ratio = round(np.mean(self._present_buf), 3)

        if not face_present:
            self._jaw_buf.append(0.0)
            self._pitch_buf.append(0.0)
            s.speaking = False
            s.nodding  = False
            return s

        # ── speaking (jaw oscillation) ───────────────────────────────────────
        jaw = blendshapes.get("jawOpen", 0.0)
        self._jaw_buf.append(jaw)
        open_frames = sum(1 for v in self._jaw_buf if v > self._jaw_thr)
        s.speaking_ratio = round(open_frames / len(self._jaw_buf), 3)
        s.speaking = s.speaking_ratio >= self._jaw_ratio

        # ── nodding (pitch sign reversals) ───────────────────────────────────
        self._pitch_buf.append(pitch)
        if len(self._pitch_buf) == self._pitch_buf.maxlen:
            signs = np.sign(np.diff(list(self._pitch_buf)))
            reversals = int(np.sum(np.abs(np.diff(signs)) > 0))
            was_nodding = s.nodding
            s.nodding = reversals >= self._nod_min_rev
            if s.nodding and not was_nodding:
                s.nod_count += 1

        return s

    def reset(self) -> None:
        self._jaw_buf.clear()
        self._pitch_buf.clear()
        self._present_buf.clear()
        self._state = ParticipationState()
