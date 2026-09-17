"""CSV and PDF report generation from a session record.

Usage (called automatically at session end in main.py):
    from app.report import SessionRecord, save_csv, save_pdf
"""
from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from fpdf import FPDF


@dataclass
class SessionRecord:
    """Aggregated metrics collected over one session."""
    session_id:       str
    started_at:       str                        # ISO-8601
    duration_seconds: float
    total_frames:     int
    face_present_pct: float                      # 0-100
    blink_count:      int
    yawn_count:       int
    nod_count:        int
    on_screen_pct:    float                      # 0-100
    avg_engagement:   float                      # 0-100
    min_engagement:   float
    max_engagement:   float
    engagement_label: str                        # Low / Moderate / High
    top_expression:   str
    speaking_pct:     float                      # 0-100
    # per-second engagement timeline (for chart in PDF)
    timeline:         list[float] = field(default_factory=list, repr=False)


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(record: SessionRecord, out_dir: str = "reports") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"session_{record.session_id}.csv"

    # summary row
    summary = {k: v for k, v in asdict(record).items() if k != "timeline"}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    # timeline sheet appended below summary
    if record.timeline:
        tl_path = Path(out_dir) / f"session_{record.session_id}_timeline.csv"
        pd.DataFrame({"second": range(len(record.timeline)),
                      "engagement": record.timeline}).to_csv(tl_path, index=False)
        return tl_path   # return the richer file

    return path


# ── PDF ───────────────────────────────────────────────────────────────────────

class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Engagement Session Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6,
                  "DISCLAIMER: Engagement score is a project-defined estimate of observable "
                  "behaviour only. Not a validated measure of emotion or learning.",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def save_pdf(record: SessionRecord, out_dir: str = "reports") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"session_{record.session_id}.pdf"

    pdf = _PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── session info ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Session Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    info_rows = [
        ("Session ID",  record.session_id),
        ("Started",     record.started_at),
        ("Duration",    f"{record.duration_seconds:.1f} s  "
                        f"({record.duration_seconds/60:.1f} min)"),
        ("Total Frames", str(record.total_frames)),
    ]
    _table(pdf, info_rows)
    pdf.ln(4)

    # ── engagement summary ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Engagement Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    eng_rows = [
        ("Average Engagement", f"{record.avg_engagement:.1f}%  [{record.engagement_label}]"),
        ("Min / Max",          f"{record.min_engagement:.1f}% / {record.max_engagement:.1f}%"),
        ("Face Present",       f"{record.face_present_pct:.1f}%"),
        ("On-Screen (forward)", f"{record.on_screen_pct:.1f}%"),
    ]
    _table(pdf, eng_rows)
    pdf.ln(4)

    # ── behaviour metrics ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Behaviour Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    beh_rows = [
        ("Blinks",          str(record.blink_count)),
        ("Yawns",           str(record.yawn_count)),
        ("Nods",            str(record.nod_count)),
        ("Speaking",        f"{record.speaking_pct:.1f}%"),
        ("Top Expression",  record.top_expression),
    ]
    _table(pdf, beh_rows)
    pdf.ln(4)

    # ── engagement timeline (ASCII sparkline) ─────────────────────────────────
    if record.timeline:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Engagement Timeline (per second)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", size=7)
        spark = _sparkline(record.timeline)
        pdf.multi_cell(0, 5, spark)
        pdf.ln(2)

    # ── disclaimer ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "This report reflects observable behaviour signals captured by a webcam. "
        "It is not suitable for grading, hiring, surveillance, or any high-stakes decision.")

    pdf.output(str(path))
    return path


# ── helpers ───────────────────────────────────────────────────────────────────

def _table(pdf: FPDF, rows: list[tuple[str, str]]) -> None:
    col_w = 60
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 7, label + ":", border=0)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7, value, border=0, new_x="LMARGIN", new_y="NEXT")


def _sparkline(values: list[float], width: int = 80) -> str:
    """Render a text sparkline using ASCII characters."""
    chars  = " ._-=+*#@"
    step   = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    return "".join(chars[min(8, int(v / 100 * 8))] for v in sampled)
