"""Engagement score engine.

> DISCLAIMER: The engagement score is a project-defined estimate of
> observable behaviour only. It is NOT a scientifically validated measure
> of emotion, concentration, or learning outcome.

Score = weighted sum of four sub-scores, each normalised to [0, 1]:

  attention     — fraction of rolling window frames where head is forward
  blink_rate    — penalises abnormally low OR high blink rates
  expression    — per-expression weight from expression.py
  participation — speaking + nodding + face-presence composite

All weights are configurable in config.yaml → engagement.weights.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np

from app.analyzers.eye import EyeState
from app.analyzers.mouth import MouthState
from app.analyzers.head_pose import HeadPoseState, Attention
from app.analyzers.expression import ExpressionState, expression_engagement_score
from app.analyzers.participation import ParticipationState


# Normal blink rate: 12–20 blinks/min → ~0.2–0.33 blinks/sec
_BLINK_RATE_LOW  = 0.10   # blinks/sec — below this → drowsy
_BLINK_RATE_HIGH = 0.60   # blinks/sec — above this → stressed / noisy


@dataclass
class EngagementSnapshot:
    score: float = 0.0          # 0–100 composite
    attention_score:     float = 0.0
    blink_score:         float = 0.0
    expression_score:    float = 0.0
    participation_score: float = 0.0
    label: str = "Unknown"      # Low / Moderate / High


def _label(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


class EngagementEngine:
    """
    Parameters
    ----------
    weights         : dict with keys attention, blink_rate, expression, participation
    window_seconds  : rolling window length in seconds
    fps             : frames per second
    """

    def __init__(
        self,
        weights: dict[str, float],
        window_seconds: float = 30.0,
        fps: float = 30.0,
    ) -> None:
        total = sum(weights.values())
        self._w = {k: v / total for k, v in weights.items()}   # normalise
        win = max(1, int(window_seconds * fps))

        # rolling buffers — one value per frame
        self._attn_buf:  deque[float] = deque(maxlen=win)
        self._blink_buf: deque[int]   = deque(maxlen=win)   # cumulative blink count
        self._expr_buf:  deque[float] = deque(maxlen=win)
        self._part_buf:  deque[float] = deque(maxlen=win)

        self._last_blink_count = 0
        self._snapshot = EngagementSnapshot()

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        eye:   EyeState,
        mouth: MouthState,
        head:  HeadPoseState,
        expr:  ExpressionState,
        part:  ParticipationState,
        fps:   float = 30.0,
    ) -> EngagementSnapshot:

        # 1. attention sub-score
        attn = 1.0 if head.attention == Attention.FORWARD else 0.0
        # penalise prolonged eye closure
        if eye.prolonged_closure:
            attn *= 0.3
        self._attn_buf.append(attn)

        # 2. blink-rate sub-score
        new_blinks = eye.blink_count - self._last_blink_count
        self._last_blink_count = eye.blink_count
        self._blink_buf.append(new_blinks)
        blink_score = self._blink_rate_score(fps)

        # 3. expression sub-score
        expr_score = expression_engagement_score(expr.expression)
        # yawning reduces expression score
        if mouth.yawning:
            expr_score *= 0.4
        self._expr_buf.append(expr_score)

        # 4. participation sub-score
        p_score = (
            0.5 * part.face_present_ratio
            + 0.3 * float(part.speaking)
            + 0.2 * float(part.nodding)
        )
        self._part_buf.append(p_score)

        # weighted composite
        a = float(np.mean(self._attn_buf))
        b = blink_score
        e = float(np.mean(self._expr_buf))
        p = float(np.mean(self._part_buf))

        raw = (self._w["attention"]     * a
             + self._w["blink_rate"]    * b
             + self._w["expression"]    * e
             + self._w["participation"] * p)

        score = round(float(np.clip(raw * 100, 0, 100)), 1)
        self._snapshot = EngagementSnapshot(
            score=score,
            attention_score=round(a * 100, 1),
            blink_score=round(b * 100, 1),
            expression_score=round(e * 100, 1),
            participation_score=round(p * 100, 1),
            label=_label(score),
        )
        return self._snapshot

    def reset(self) -> None:
        self._attn_buf.clear()
        self._blink_buf.clear()
        self._expr_buf.clear()
        self._part_buf.clear()
        self._last_blink_count = 0
        self._snapshot = EngagementSnapshot()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _blink_rate_score(self, fps: float) -> float:
        """Map rolling blink rate (blinks/sec) to a 0-1 score."""
        if not self._blink_buf:
            return 0.5
        window_secs = len(self._blink_buf) / fps
        rate = sum(self._blink_buf) / max(window_secs, 1e-6)
        if rate < _BLINK_RATE_LOW:
            return float(rate / _BLINK_RATE_LOW)          # ramps 0→1
        if rate > _BLINK_RATE_HIGH:
            return float(max(0.0, 1.0 - (rate - _BLINK_RATE_HIGH)))
        # within normal range → linear 0.5→1.0→0.5 peak at midpoint
        mid = (_BLINK_RATE_LOW + _BLINK_RATE_HIGH) / 2
        return float(1.0 - abs(rate - mid) / (mid - _BLINK_RATE_LOW + 1e-6) * 0.5)
