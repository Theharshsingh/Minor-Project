"""Entry point — Milestone 9: CSV/PDF report generation."""
from __future__ import annotations

import sys
import cv2
import yaml
import numpy as np

from app.capture import VideoCapture
from app.detector import FaceDetector
from app.analyzers.eye import EyeAnalyzer
from app.analyzers.mouth import MouthAnalyzer
from app.analyzers.head_pose import HeadPoseAnalyzer, HeadPoseState, draw_pose_axes
from app.analyzers.expression import ExpressionClassifier, Expression
from app.analyzers.participation import ParticipationAnalyzer
from app.engagement import EngagementEngine, EngagementSnapshot
from app.report import SessionRecord, save_csv, save_pdf
import datetime, time, collections


CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58),  (58, 132),  (132, 93),  (93, 234),
    (234, 127), (127, 162), (162, 21),  (21, 54),
    (54, 103),  (103, 67),  (67, 109),  (109, 10),
]


def draw_landmarks(frame: np.ndarray, landmarks: list) -> None:
    h, w = frame.shape[:2]
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)
    for pt in pts.values():
        cv2.circle(frame, pt, 1, (0, 200, 255), -1)


_EXPR_COLOR: dict[str, tuple] = {
    "Neutral":   (200, 200, 200),
    "Happy":     (0,   255, 100),
    "Sad":       (255, 100,  50),
    "Surprised": (0,   220, 255),
    "Angry":     (0,    0,  255),
    "Confused":  (180, 100, 255),
}


def _score_color(score: float) -> tuple:
    if score >= 70:
        return (0, 220, 0)
    if score >= 40:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_engagement_bar(frame: np.ndarray, snap: EngagementSnapshot) -> None:
    h, w   = frame.shape[:2]
    bar_w  = int(w * 0.40)
    bar_h  = 18
    x, y   = w - bar_w - 10, 10
    filled = int(bar_w * snap.score / 100)
    color  = _score_color(snap.score)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Engagement: {snap.score:.0f}%  [{snap.label}]",
                (x, y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    sub = (f"Attn:{snap.attention_score:.0f}  "
           f"Blink:{snap.blink_score:.0f}  "
           f"Expr:{snap.expression_score:.0f}  "
           f"Part:{snap.participation_score:.0f}")
    cv2.putText(frame, sub, (x, y + bar_h + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def draw_hud(frame: np.ndarray, eye_state, mouth_state,
             head_state, expr_state, part_state, frame_idx: int) -> None:
    attn_color = (0, 255, 0) if head_state.attention.value == "Forward" else (0, 165, 255)
    expr_color = _EXPR_COLOR.get(expr_state.expression.value, (200, 200, 200))
    spk_color  = (0, 255, 180) if part_state.speaking else (200, 200, 200)
    lines = [
        (f"Frame   : {frame_idx}",                                                          (200, 200, 200)),
        (f"EAR     : {eye_state.ear:.3f}  Blinks : {eye_state.blink_count}",                (200, 200, 200)),
        (f"MAR     : {mouth_state.mar:.3f}  Yawns  : {mouth_state.yawn_count}",             (200, 200, 200)),
        (f"Pitch   : {head_state.pitch:+.1f}  Yaw : {head_state.yaw:+.1f}  Roll : {head_state.roll:+.1f}", (200, 200, 200)),
        (f"Attn    : {head_state.attention.value}  ({head_state.on_screen_ratio:.0%} on-screen)", attn_color),
        (f"Expr    : {expr_state.expression.value}  ({expr_state.confidence:.0%})",         expr_color),
        (f"Speaking: {'Yes' if part_state.speaking else 'No'}  ({part_state.speaking_ratio:.0%})  Nods: {part_state.nod_count}", spk_color),
        (f"Present : {part_state.face_present_ratio:.0%}",                                  (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 22 + i * 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    alert_y = 200
    if eye_state.prolonged_closure:
        cv2.putText(frame, "⚠ EYES CLOSED", (10, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        alert_y += 28
    if mouth_state.yawning:
        cv2.putText(frame, "⚠ YAWNING", (10, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        alert_y += 28
    if head_state.attention.value != "Forward":
        cv2.putText(frame, f"⚠ LOOKING {head_state.attention.value.upper()}", (10, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    source = cfg["input"]["source"]
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg

    thr = cfg["thresholds"]
    eye_analyzer = EyeAnalyzer(
        ear_threshold=thr["ear_blink"],
        closed_frames=thr["ear_closed_frames"],
    )
    mouth_analyzer = MouthAnalyzer(
        mar_threshold=thr["mar_yawn"],
        yawn_frames=thr["yawn_frames"],
    )
    head_analyzer = HeadPoseAnalyzer(
        pitch_range=tuple(thr["head_pitch_range"]),
        yaw_range=tuple(thr["head_yaw_range"]),
    )
    expr_classifier = ExpressionClassifier(
        model_path=cfg.get("expression", {}).get("model_path")
    )
    part_analyzer = ParticipationAnalyzer(fps=cfg["input"]["fps"])
    eng_cfg    = cfg["engagement"]
    eng_engine = EngagementEngine(
        weights=eng_cfg["weights"],
        window_seconds=eng_cfg["window_seconds"],
        fps=cfg["input"]["fps"],
    )

    # report accumulators
    timeline:        list[float]       = []
    expr_counter:    collections.Counter = collections.Counter()
    speaking_frames: int = 0
    total_frames:    int = 0
    session_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at  = datetime.datetime.now().isoformat(timespec="seconds")
    t_start     = time.time()
    last_sec:   int = -1
    sec_scores: list[float] = []

    print("[INFO] Starting — press Q to quit.")
    with (
        VideoCapture(source, cfg["input"]["width"], cfg["input"]["height"]) as cap,
        FaceDetector(
            cfg["mediapipe"]["model_path"],
            num_faces=cfg["mediapipe"]["num_faces"],
            detection_conf=cfg["mediapipe"]["min_face_detection_confidence"],
            presence_conf=cfg["mediapipe"]["min_face_presence_confidence"],
            tracking_conf=cfg["mediapipe"]["min_tracking_confidence"],
        ) as detector,
    ):
        for fr in cap.frames():
            result  = detector.detect(fr.frame)
            display = fr.frame.copy()

            if result:
                draw_landmarks(display, result.landmarks)
                eye_state   = eye_analyzer.update(result.landmarks)
                mouth_state = mouth_analyzer.update(result.landmarks)
                expr_state  = expr_classifier.predict(result.blendshapes)
                head_state  = HeadPoseState()
                if result.transform_matrix is not None:
                    head_state = head_analyzer.update(result.transform_matrix)
                    draw_pose_axes(display, result.transform_matrix, result.landmarks[1])
                part_state = part_analyzer.update(
                    result.blendshapes, head_state.pitch, face_present=True
                )
                snap = eng_engine.update(
                    eye_state, mouth_state, head_state, expr_state, part_state,
                    fps=cfg["input"]["fps"],
                )
                draw_hud(display, eye_state, mouth_state, head_state,
                         expr_state, part_state, fr.frame_index)
                draw_engagement_bar(display, snap)
                expr_counter[expr_state.expression.value] += 1
                if part_state.speaking:
                    speaking_frames += 1
            else:
                cv2.putText(display, "No Face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                part_analyzer.update({}, 0.0, face_present=False)
                eye_analyzer.reset()
                mouth_analyzer.reset()
                head_analyzer.reset()

            # per-second timeline bucket
            total_frames += 1
            elapsed_sec = int(time.time() - t_start)
            sec_scores.append(eng_engine._snapshot.score)
            if elapsed_sec != last_sec:
                if sec_scores:
                    timeline.append(round(sum(sec_scores) / len(sec_scores), 1))
                sec_scores = []
                last_sec = elapsed_sec

            cv2.imshow("Engagement Analyzer — Milestone 7", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    ps = part_analyzer._state
    es = eng_engine._snapshot
    duration = round(time.time() - t_start, 1)
    print(f"[INFO] Session ended. Blinks: {eye_analyzer._state.blink_count}  "
          f"Yawns: {mouth_analyzer._state.yawn_count}  "
          f"On-screen: {head_analyzer._state.on_screen_ratio:.0%}  "
          f"Nods: {ps.nod_count}  Present: {ps.face_present_ratio:.0%}  "
          f"Engagement: {es.score:.1f}% [{es.label}]")

    if total_frames > 0 and timeline:
        record = SessionRecord(
            session_id       = session_id,
            started_at       = started_at,
            duration_seconds = duration,
            total_frames     = total_frames,
            face_present_pct = round(ps.face_present_ratio * 100, 1),
            blink_count      = eye_analyzer._state.blink_count,
            yawn_count       = mouth_analyzer._state.yawn_count,
            nod_count        = ps.nod_count,
            on_screen_pct    = round(head_analyzer._state.on_screen_ratio * 100, 1),
            avg_engagement   = round(sum(timeline) / len(timeline), 1),
            min_engagement   = round(min(timeline), 1),
            max_engagement   = round(max(timeline), 1),
            engagement_label = es.label,
            top_expression   = expr_counter.most_common(1)[0][0] if expr_counter else "N/A",
            speaking_pct     = round(speaking_frames / total_frames * 100, 1),
            timeline         = timeline,
        )
        csv_path = save_csv(record)
        pdf_path = save_pdf(record)
        print(f"[INFO] Reports saved:\n  CSV → {csv_path}\n  PDF → {pdf_path}")


if __name__ == "__main__":
    main()
