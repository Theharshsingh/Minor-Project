"""Facial expression classification from MediaPipe blendshapes.

Primary path  : rule-based mapping of blendshape scores → 6 categories.
Optional path : drop-in scikit-learn classifier loaded from a .pkl file.

Categories: NEUTRAL, HAPPY, SAD, SURPRISED, ANGRY, CONFUSED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np


class Expression(str, Enum):
    NEUTRAL   = "Neutral"
    HAPPY     = "Happy"
    SAD       = "Sad"
    SURPRISED = "Surprised"
    ANGRY     = "Angry"
    CONFUSED  = "Confused"


# Blendshape keys used per expression (subset of MediaPipe's 52 blendshapes)
_RULES: dict[Expression, list[str]] = {
    Expression.HAPPY:     ["mouthSmileLeft", "mouthSmileRight"],
    Expression.SAD:       ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    Expression.SURPRISED: ["jawOpen", "eyeWideLeft", "eyeWideRight"],
    Expression.ANGRY:     ["browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"],
    Expression.CONFUSED:  ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
}
_THRESHOLD = 0.35   # blendshape score must exceed this to count


@dataclass
class ExpressionState:
    expression: Expression = Expression.NEUTRAL
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)  # raw blendshape subset


class ExpressionClassifier:
    """Rule-based classifier; optionally replaced by a sklearn model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._sklearn_clf = None
        if model_path and Path(model_path).exists():
            import pickle
            with open(model_path, "rb") as f:
                self._sklearn_clf = pickle.load(f)

    def predict(self, blendshapes: dict[str, float]) -> ExpressionState:
        if self._sklearn_clf is not None:
            return self._sklearn_predict(blendshapes)
        return self._rule_predict(blendshapes)

    # ── rule-based ────────────────────────────────────────────────────────────

    def _rule_predict(self, bs: dict[str, float]) -> ExpressionState:
        scores: dict[Expression, float] = {}
        for expr, keys in _RULES.items():
            vals = [bs.get(k, 0.0) for k in keys]
            scores[expr] = float(np.mean(vals))

        best_expr  = max(scores, key=lambda e: scores[e])
        best_score = scores[best_expr]

        if best_score < _THRESHOLD:
            best_expr, best_score = Expression.NEUTRAL, 1.0 - best_score

        subset = {k: round(bs.get(k, 0.0), 3)
                  for keys in _RULES.values() for k in keys}
        return ExpressionState(expression=best_expr,
                               confidence=round(best_score, 3),
                               scores=subset)

    # ── sklearn override ──────────────────────────────────────────────────────

    def _sklearn_predict(self, bs: dict[str, float]) -> ExpressionState:
        all_keys = sorted(bs.keys())
        vec = np.array([bs[k] for k in all_keys], dtype=np.float32).reshape(1, -1)
        label   = self._sklearn_clf.predict(vec)[0]
        proba   = float(np.max(self._sklearn_clf.predict_proba(vec)))
        expr    = Expression(label) if label in Expression._value2member_map_ else Expression.NEUTRAL
        return ExpressionState(expression=expr, confidence=round(proba, 3))


# ── expression → engagement weight ───────────────────────────────────────────

_ENGAGEMENT_WEIGHT: dict[Expression, float] = {
    Expression.NEUTRAL:   0.6,
    Expression.HAPPY:     1.0,
    Expression.SAD:       0.3,
    Expression.SURPRISED: 0.8,
    Expression.ANGRY:     0.2,
    Expression.CONFUSED:  0.5,
}


def expression_engagement_score(expr: Expression) -> float:
    """Return a 0-1 engagement contribution for the given expression."""
    return _ENGAGEMENT_WEIGHT.get(expr, 0.5)
