"""Evaluation utilities — measure accuracy of rule-based classifiers
on synthetic labelled datasets.

Run standalone:
    python -m app.evaluation

Produces a printed report of per-class precision/recall and overall accuracy
for the expression classifier and attention classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class EvalResult:
    module:   str
    accuracy: float          # 0-1
    n_samples: int
    per_class: dict[str, dict]   # label → {tp, fp, fn, precision, recall, f1}

    def __str__(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  {self.module}",
            f"  Accuracy : {self.accuracy:.1%}  ({self.n_samples} samples)",
            f"{'='*55}",
            f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6}",
            f"  {'-'*40}",
        ]
        for label, m in self.per_class.items():
            lines.append(
                f"  {label:<14} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%}"
            )
        return "\n".join(lines)


def evaluate(
    predict_fn: Callable,
    samples: list[tuple],          # (input, expected_label)
    module_name: str,
) -> EvalResult:
    labels = sorted({s[1] for s in samples})
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    correct = 0

    for inp, expected in samples:
        predicted = predict_fn(inp)
        if predicted == expected:
            correct += 1
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected]  += 1

    per_class = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        rec  = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp[l], "fp": fp[l], "fn": fn[l]}

    return EvalResult(
        module=module_name,
        accuracy=correct / len(samples),
        n_samples=len(samples),
        per_class=per_class,
    )


# ── expression classifier evaluation ─────────────────────────────────────────

def _expression_samples() -> list[tuple[dict, str]]:
    """Synthetic blendshape → expression label pairs."""
    base = {k: 0.0 for k in [
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "browInnerUp", "browDownLeft", "browDownRight", "noseSneerLeft",
        "noseSneerRight", "jawOpen", "eyeWideLeft", "eyeWideRight",
        "browOuterUpLeft", "browOuterUpRight",
    ]}

    def bs(**kw):
        d = dict(base)
        d.update(kw)
        return d

    return [
        # Neutral (all low)
        (bs(), "Neutral"),
        (bs(mouthSmileLeft=0.05, jawOpen=0.05), "Neutral"),
        # Happy
        (bs(mouthSmileLeft=0.8, mouthSmileRight=0.8), "Happy"),
        (bs(mouthSmileLeft=0.6, mouthSmileRight=0.7), "Happy"),
        (bs(mouthSmileLeft=0.9, mouthSmileRight=0.9), "Happy"),
        # Sad
        (bs(mouthFrownLeft=0.7, mouthFrownRight=0.7, browInnerUp=0.6), "Sad"),
        (bs(mouthFrownLeft=0.5, mouthFrownRight=0.6, browInnerUp=0.5), "Sad"),
        # Surprised
        (bs(jawOpen=0.9, eyeWideLeft=0.8, eyeWideRight=0.8), "Surprised"),
        (bs(jawOpen=0.7, eyeWideLeft=0.6, eyeWideRight=0.7), "Surprised"),
        # Angry
        (bs(browDownLeft=0.7, browDownRight=0.7, noseSneerLeft=0.6, noseSneerRight=0.6), "Angry"),
        (bs(browDownLeft=0.8, browDownRight=0.8, noseSneerLeft=0.5, noseSneerRight=0.5), "Angry"),
        # Confused
        (bs(browInnerUp=0.7, browOuterUpLeft=0.6, browOuterUpRight=0.6), "Confused"),
        (bs(browInnerUp=0.8, browOuterUpLeft=0.7, browOuterUpRight=0.7), "Confused"),
    ]


def eval_expression() -> EvalResult:
    from app.analyzers.expression import ExpressionClassifier
    clf = ExpressionClassifier()
    samples = _expression_samples()
    return evaluate(
        predict_fn=lambda bs: clf.predict(bs).expression.value,
        samples=samples,
        module_name="Expression Classifier (rule-based)",
    )


# ── attention classifier evaluation ──────────────────────────────────────────

def _attention_samples() -> list[tuple[np.ndarray, str]]:
    import cv2

    def rot_matrix(pitch_deg=0.0, yaw_deg=0.0) -> np.ndarray:
        rvec = np.array([np.radians(pitch_deg), np.radians(yaw_deg), 0.0])
        rot, _ = cv2.Rodrigues(rvec)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = rot
        return m

    return [
        (rot_matrix(0,   0),   "Forward"),
        (rot_matrix(5,   5),   "Forward"),
        (rot_matrix(-5, -5),   "Forward"),
        (rot_matrix(0,  40),   "Right"),
        (rot_matrix(0, -40),   "Left"),
        (rot_matrix(30,  0),   "Down"),
        (rot_matrix(-30, 0),   "Up"),
        (rot_matrix(0,  35),   "Right"),
        (rot_matrix(0, -35),   "Left"),
        (rot_matrix(25,  0),   "Down"),
    ]


def eval_attention() -> EvalResult:
    from app.analyzers.head_pose import HeadPoseAnalyzer
    analyzer = HeadPoseAnalyzer(pitch_range=(-20, 20), yaw_range=(-30, 30))
    samples  = _attention_samples()
    return evaluate(
        predict_fn=lambda m: analyzer.update(m).attention.value,
        samples=samples,
        module_name="Attention Classifier (head pose)",
    )


# ── EAR blink evaluation ──────────────────────────────────────────────────────

def _blink_samples():
    """Sequence-level: (ear_sequence, expected_blink_count)."""
    from unittest.mock import MagicMock

    def lms(ear):
        l = [MagicMock() for _ in range(478)]
        for lm in l:
            lm.x, lm.y = 0.0, 0.0
        for p1, p2, p3, p4, p5, p6 in [
            (362, 385, 387, 263, 373, 380),
            (33,  160, 158, 133, 153, 144),
        ]:
            l[p1].x, l[p1].y = 0.0, 0.0
            l[p4].x, l[p4].y = 1.0, 0.0
            l[p2].x, l[p2].y = 0.25, ear
            l[p6].x, l[p6].y = 0.25, 0.0
            l[p3].x, l[p3].y = 0.75, ear
            l[p5].x, l[p5].y = 0.75, 0.0
        return l

    # (open_frames, close_frames, open_frames) → 1 blink
    return [
        ([0.35]*10 + [0.10]*3 + [0.35]*5,  1),
        ([0.35]*10 + [0.10]*3 + [0.35]*5
                  + [0.10]*3 + [0.35]*5,   2),
        ([0.35]*20,                          0),
        ([0.10]*20,                          0),   # prolonged closure, not a blink
    ], lms


def eval_blink() -> EvalResult:
    from app.analyzers.eye import EyeAnalyzer
    sequences, lms_fn = _blink_samples()
    samples = []
    for seq, expected in sequences:
        analyzer = EyeAnalyzer(ear_threshold=0.20, closed_frames=15)
        for ear in seq:
            analyzer.update(lms_fn(ear))
        predicted = analyzer._state.blink_count
        samples.append((predicted, expected))

    correct = sum(1 for p, e in samples if p == e)
    return EvalResult(
        module="EAR Blink Counter",
        accuracy=correct / len(samples),
        n_samples=len(samples),
        per_class={"exact_match": {
            "precision": correct / len(samples),
            "recall":    correct / len(samples),
            "f1":        correct / len(samples),
        }},
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [eval_expression(), eval_attention(), eval_blink()]
    for r in results:
        print(r)
    print()
    overall = np.mean([r.accuracy for r in results])
    print(f"  Overall mean accuracy across modules: {overall:.1%}")
