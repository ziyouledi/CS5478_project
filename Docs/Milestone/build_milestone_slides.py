#!/usr/bin/env python3
"""Append the four milestone slides to the proposal deck, in the proposal's visual style.

Inputs (in data/):
  pca_cumulative_variance.csv     from Experiment Plan/results/action-pca-job1024096/
  expert_eval.json                evaluate_students.py output for the expert run        (optional)
  student_evals/*.json            evaluate_students.py outputs for student runs          (optional)
  learning_curves_successes.json  analyze_learning_curves.py output                      (optional)
Speaker notes are read from milestone_script_2min.md.
When the learning-curve inputs are missing, the results panel shows a labelled placeholder and the
numbers read "[pending]". Re-run after the evaluation jobs finish to fill them in.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # overridable with --data_dir

# Proposal palette.
NAVY = RGBColor(0x17, 0x21, 0x3A)
DARK = RGBColor(0x0B, 0x16, 0x33)
MUTED = RGBColor(0x62, 0x70, 0x8D)
BLUE = RGBColor(0x2C, 0x67, 0xF2)
ORANGE = RGBColor(0xFF, 0x8A, 0x5B)
DEEP_ORANGE = RGBColor(0xD9, 0x60, 0x2F)
GREEN = RGBColor(0x28, 0xB7, 0x8D)
PURPLE = RGBColor(0x8A, 0x68, 0xE8)
BORDER = RGBColor(0xD9, 0xE0, 0xEC)
CARD = RGBColor(0xF6, 0xF8, 0xFC)
LIGHT_ORANGE = RGBColor(0xFF, 0xF0, 0xEA)
LIGHT_BLUE = RGBColor(0xEA, 0xF0, 0xFF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"
PENDING = "[pending]"


# ---------------------------------------------------------------- drawing helpers


def text(slide, x, y, w, h, runs, size=10, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs; each paragraph is a str or a list of (text, overrides)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    paragraphs = runs if isinstance(runs, list) else [runs]
    for index, paragraph in enumerate(paragraphs):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.alignment = align
        pieces = paragraph if isinstance(paragraph, list) else [(paragraph, {})]
        for piece, overrides in pieces:
            run = p.add_run()
            run.text = piece
            font = run.font
            font.name = FONT
            font.size = Pt(overrides.get("size", size))
            font.bold = overrides.get("bold", bold)
            font.italic = overrides.get("italic", False)
            font.color.rgb = overrides.get("color", color)
        if index > 0:
            p.space_before = Pt(3)
    return box


def card(slide, x, y, w, h, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, weight=0.75):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(weight)
    box.shadow.inherit = False
    if name:
        box.name = name
    return box


def arrow(slide, x1, y1, x2, y2, color=BLUE):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(1.25)
    end = connector.line._get_or_add_ln()
    tail = end.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    end.append(tail)
    return connector


def circle_number(slide, x, y, d, label, fill):
    dot = card(slide, x, y, d, d, fill=fill, line=None, shape=MSO_SHAPE.OVAL)
    text(slide, x, y, d, d, label, size=9, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return dot


def header(slide, title, number):
    text(slide, 0.5, 0.3, 8.4, 0.55, title, size=21, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(slide, 9.1, 0.3, 0.4, 0.3, number, size=8, color=BLUE, bold=True, align=PP_ALIGN.RIGHT, name="Page number")


def new_slide(prs, layout):
    slide = prs.slides.add_slide(layout)
    for placeholder in list(slide.placeholders):
        placeholder._element.getparent().remove(placeholder._element)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    return slide


def style_chart(chart, x_title, y_title, x_max, y_max, x_major, y_major, legend=True):
    chart.font.name = FONT
    chart.font.size = Pt(8)
    chart.font.color.rgb = MUTED
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(7.5)
        chart.legend.font.color.rgb = NAVY
    for axis, title, maximum, major in (
        (chart.category_axis, x_title, x_max, x_major),
        (chart.value_axis, y_title, y_max, y_major),
    ):
        axis.minimum_scale = 0
        axis.maximum_scale = maximum
        axis.major_unit = major
        axis.has_title = True
        axis.axis_title.text_frame.text = title
        run = axis.axis_title.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(8)
        run.font.bold = False
        run.font.color.rgb = MUTED
        run.font.name = FONT
        axis.tick_labels.font.size = Pt(7.5)
        axis.tick_labels.font.color.rgb = MUTED
        axis.format.line.color.rgb = BORDER
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = BORDER
    chart.value_axis.major_gridlines.format.line.width = Pt(0.5)
    chart.category_axis.has_major_gridlines = False


def style_series(series, color, width=1.75, dash=None, marker=False):
    series.smooth = False
    line = series.format.line
    line.color.rgb = color
    line.width = Pt(width)
    if dash is not None:
        line.dash_style = dash
    if marker:
        series.marker.style = XL_MARKER_STYLE.CIRCLE
        series.marker.size = 7
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = color
        series.marker.format.line.color.rgb = color
    else:
        series.marker.style = XL_MARKER_STYLE.NONE


# ---------------------------------------------------------------- data


def load_pca():
    curves = {"robot20": [], "finger18": []}
    with (DATA / "pca_cumulative_variance.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            curves[row["basis"]].append((int(row["k"]), float(row["cumulative"])))
    return curves


def load_learning_curves():
    expert_path = DATA / "expert_eval.json"
    student_paths = sorted((DATA / "student_evals").glob("*.json")) if (DATA / "student_evals").is_dir() else []
    summary_path = DATA / "learning_curves_successes.json"
    if not (expert_path.is_file() and student_paths and summary_path.is_file()):
        return None

    def curve(path):
        report = json.loads(path.read_text(encoding="utf-8"))
        points = sorted((c["iteration"], c["mean_successes_per_episode"]) for c in report["checkpoints"])
        return report, points

    _, expert_points = curve(expert_path)
    reference = expert_points[-1][1]
    students = []
    for path in student_paths:
        report, points = curve(path)
        students.append((report["intent_record"]["intent"], report["intent_record"].get("seed"), points))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"expert": expert_points, "reference": reference, "students": students, "summary": summary}


def fmt_iterations(value):
    return f"{value:,.0f}" if value is not None else "not reached"


# ---------------------------------------------------------------- slides


def slide_progress(prs, layout, curves):
    slide = new_slide(prs, layout)
    header(slide, "Milestone: the full pipeline now runs end to end", "08")
    n_done = len({(s[0], s[1]) for s in curves["students"]}) if curves else 4

    steps = [
        ("Expert PPO", "goal visible · frozen", LIGHT_BLUE, BLUE, NAVY, "Done"),
        ("Rollout + PCA", "1.23 M expert actions", WHITE, NAVY, NAVY, "Done"),
        ("Intent modules", "Hand · Wrist · Obj", LIGHT_ORANGE, ORANGE, NAVY, "Done"),
        ("Student PPO", "Stage 1 · 7 intents × 3 seeds", DARK, DARK, WHITE, "Running"),
        ("Evaluator", "per-episode successes", WHITE, NAVY, NAVY, "Done"),
    ]
    x, y, w, h, gap = 0.5, 1.15, 1.56, 0.82, 0.30
    for index, (title, sub, fill, line, ink, status) in enumerate(steps):
        left = x + index * (w + gap)
        card(slide, left, y, w, h, fill=fill, line=line, name=f"Step {index + 1}")
        text(slide, left + 0.05, y + 0.12, w - 0.1, 0.3, title, size=10.5, color=ink, bold=True, align=PP_ALIGN.CENTER)
        text(slide, left + 0.05, y + 0.42, w - 0.1, 0.3, sub, size=8, color=ink if ink == WHITE else MUTED,
             align=PP_ALIGN.CENTER)
        badge_color = GREEN if status == "Done" else ORANGE
        text(slide, left, y + h + 0.06, w, 0.2, ("✓ " if status == "Done" else "● ") + status, size=8,
             color=badge_color, bold=True, align=PP_ALIGN.CENTER)
        if index < len(steps) - 1:
            arrow(slide, left + w + 0.02, y + h / 2, left + w + gap - 0.02, y + h / 2)

    stats = [
        ("163.8 M", "expert training transitions"),
        (f"{curves['reference']:.1f}" if curves else "26.2", "expert successes per episode\n(1,024 eval episodes)"),
        ("5×10⁻⁶", "max error: rebuilt expert vs\nrecorded actions"),
        (f"{n_done} / 21", "Stage 1 runs trained\nand evaluated"),
    ]
    sx, sy, sw, sh, sgap = 0.5, 2.7, 2.1, 1.35, 0.2
    for index, (value, label) in enumerate(stats):
        left = sx + index * (sw + sgap)
        card(slide, left, sy, sw, sh, fill=CARD, line=BORDER, name=f"Stat {index + 1}")
        text(slide, left + 0.18, sy + 0.17, sw - 0.3, 0.5, value, size=24, color=ORANGE if index == 3 else NAVY,
             bold=True)
        text(slide, left + 0.18, sy + 0.72, sw - 0.3, 0.5, label.split("\n"), size=8.5, color=MUTED)

    text(slide, 0.5, 4.4, 9.0, 0.55, [
        [("Infrastructure: ", {"bold": True, "color": NAVY}),
         ("Isaac Lab 2.3 on the NUS SoC Slurm cluster (Tesla T4). Each run is a reproducible Slurm job: frozen "
          "expert hash, PCA basis hash, seed and config are recorded with every result.", {"color": MUTED})],
    ], size=9)
    slide.notes_slide.notes_text_frame.text = SCRIPT[0]
    return slide


def slide_results(prs, layout, curves, pca):
    slide = new_slide(prs, layout)
    header(slide, "Early results: Wrist-2 sits inside Blind's seed noise" if curves
           else "Early results: first curves and the expert's actions", "09")

    # Left: learning curves.
    card(slide, 0.5, 1.05, 4.4, 3.95, name="Learning curve card")
    text(slide, 0.68, 1.15, 4.0, 0.25, "Stage 1 learning curves (goal visible)", size=9.5, bold=True)
    if curves is None:
        holder = card(slide, 0.7, 1.5, 4.0, 2.55, fill=CARD, line=BORDER, name="Learning curve placeholder")
        holder.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        text(slide, 0.8, 2.35, 3.8, 0.8, [
            "Learning curves go here",
            [("expert · Blind (3 seeds) · Wrist-2 (seed 0)", {"size": 8, "color": MUTED})],
            [("filled when evaluation job 1024771 finishes", {"size": 8, "color": MUTED})],
        ], size=10, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
        blind_n80 = blind_auc = wrist_n80 = wrist_auc = PENDING
    else:
        reference = curves["reference"]
        data = XyChartData()
        replica = [s for s in curves["students"] if s[2] == curves["expert"]]
        students = [s for s in curves["students"] if s[2] != curves["expert"]]
        expert = data.add_series("Expert = Blind · seed 0" if replica else "Expert (seed 0)")
        for iteration, value in curves["expert"]:
            expert.add_data_point(iteration, value / reference)
        order = []
        for intent, seed, points in students:
            label = f"{intent.capitalize()} · seed {seed}"
            series = data.add_series(label)
            for iteration, value in points:
                series.add_data_point(iteration, value / reference)
            order.append(intent)
        line80 = data.add_series("80% of expert")
        line80.add_data_point(0, 0.8)
        line80.add_data_point(5000, 0.8)
        frame = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(0.6), Inches(1.42),
                                       Inches(4.2), Inches(2.7), data)
        frame.name = "Learning curve chart"
        chart = frame.chart
        style_chart(chart, "PPO iteration", "successes / expert final", 5000, 1.1, 1000, 0.2)
        palette = {"blind": [MUTED, RGBColor(0x9A, 0xA4, 0xBA)], "wrist-2": [ORANGE]}
        seen = {}
        plots = list(chart.plots[0].series)
        style_series(plots[0], NAVY, width=2.25)
        for series, intent in zip(plots[1:1 + len(order)], order):
            colors = palette.get(intent, [BLUE])
            style_series(series, colors[seen.get(intent, 0) % len(colors)], width=1.5)
            seen[intent] = seen.get(intent, 0) + 1
        style_series(plots[-1], MUTED, width=1, dash=MSO_LINE_DASH_STYLE.DASH)
        conditions = curves["summary"]["conditions"]
        blind = conditions.get("blind", {})
        wrist = conditions.get("wrist-2", {})
        blind_values = [v for v in blind.get("n80_iterations_values", []) if v is not None]
        blind_n80 = (f"{min(blind_values):,.0f}–{max(blind_values):,.0f}" if blind_values else "not reached")
        blind_aucs = [r["auc_normalized"] for r in curves["summary"]["runs"] if r["intent"] == "blind"]
        blind_auc = f"{min(blind_aucs):.2f}–{max(blind_aucs):.2f}" if blind_aucs else PENDING
        wrist_auc = next((f"{r['auc_normalized']:.2f}" for r in curves["summary"]["runs"] if r["intent"] == "wrist-2"),
                         PENDING)
        wrist_values = wrist.get("n80_iterations_values", [None])
        wrist_n80 = fmt_iterations(wrist_values[0]) if wrist else PENDING
    text(slide, 0.68, 4.2, 4.1, 0.7, [
        [("N₈₀ (iterations): ", {"bold": True, "color": NAVY}),
         (f"Blind {blind_n80} across 3 seeds · Wrist-2 {wrist_n80} (1 seed)", {"color": MUTED})],
        [("AUC: ", {"bold": True, "color": NAVY}),
         (f"Blind {blind_auc} · Wrist-2 {wrist_auc}. Blind seed 0 retrains the expert bit for bit (same "
          "checkpoint hash).", {"color": MUTED})],
    ], size=8.5)

    # Right: PCA spectrum.
    card(slide, 5.1, 1.05, 4.4, 3.95, name="PCA card")
    text(slide, 5.28, 1.15, 4.0, 0.25, "The RL expert's actions are not low-dimensional", size=9.5, bold=True)
    data = XyChartData()
    robot = data.add_series("Expert, 20-D (hand + wrist)")
    for k, value in [(0, 0.0)] + pca["robot20"]:
        robot.add_data_point(k, value)
    finger = data.add_series("Expert, 18-D (fingers)")
    for k, value in [(0, 0.0)] + pca["finger18"]:
        finger.add_data_point(k, value)
    human = data.add_series("Human grasps (Santello 1998)")
    human.add_data_point(2, 0.8)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(5.2), Inches(1.42),
                                   Inches(4.2), Inches(2.7), data)
    frame.name = "PCA chart"
    chart = frame.chart
    style_chart(chart, "principal components k", "cumulative explained variance", 20, 1.0, 2, 0.2)
    plots = list(chart.plots[0].series)
    style_series(plots[0], BLUE, width=2.25)
    style_series(plots[1], PURPLE, width=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    style_series(plots[2], ORANGE, width=0, marker=True)
    plots[2].format.line.fill.background()
    robot20 = dict(pca["robot20"])
    k95 = next(k for k, v in pca["robot20"] if v >= 0.95)
    text(slide, 5.28, 4.2, 4.1, 0.7, [
        [("k = 2 → ", {"bold": True, "color": NAVY}), (f"{robot20[2]:.0%}", {"bold": True, "color": BLUE}),
         ("   k = 4 → ", {"bold": True, "color": NAVY}), (f"{robot20[4]:.0%}", {"bold": True, "color": BLUE}),
         ("   95% needs ", {"bold": True, "color": NAVY}), (f"{k95} dims", {"bold": True, "color": BLUE})],
        [("Not a controller artifact: the measured joint motion is smooth (6% at joint limits) yet needs 15 "
          "dims. Human data are static grasps.", {"color": MUTED})],
    ], size=8.5)
    slide.notes_slide.notes_text_frame.text = SCRIPT[1]
    return slide


def slide_challenges(prs, layout):
    slide = new_slide(prs, layout)
    header(slide, "Challenges, and how we are handling them", "10")
    text(slide, 1.0, 1.0, 3.6, 0.25, "Challenge", size=9, color=DEEP_ORANGE, bold=True)
    text(slide, 5.25, 1.0, 4.0, 0.25, "Response", size=9, color=BLUE, bold=True)
    rows = [
        ("High-dimensional expert", "Even its smooth joint motion needs 13–15 dims for 95%; low-k hand intent "
         "may carry little.",
         "Run Hand-k as planned; a weak result is itself a finding. M2, a simpler rotation task, tests whether "
         "motion there compresses better."),
        ("Misleading built-in metric", "Isaac Lab's success metric is a running average: 3.4 vs 16.2 at the "
         "same reward.",
         "Own evaluator: successes counted per episode, one episode per env, identical start states for every "
         "checkpoint."),
        ("Seed variance", "Blind's N₈₀ already varies 2× across 3 seeds; small speed-ups would sit inside it.",
         "Use Blind's seed spread as the noise floor; add seeds to the key conditions if needed."),
    ]
    y, h, gap = 1.35, 0.95, 0.2
    for index, (title, problem, response) in enumerate(rows):
        top = y + index * (h + gap)
        card(slide, 0.5, top, 4.35, h, fill=LIGHT_ORANGE, line=ORANGE, name=f"Challenge {index + 1}")
        circle_number(slide, 0.65, top + (h - 0.3) / 2, 0.3, str(index + 1), ORANGE)
        text(slide, 1.1, top + 0.1, 3.6, h - 0.15, [
            [(title, {"bold": True, "color": NAVY, "size": 9.5})],
            [(problem, {"color": MUTED, "size": 8.5})],
        ], size=9)
        arrow(slide, 4.9, top + h / 2, 5.1, top + h / 2, color=BLUE)
        card(slide, 5.15, top, 4.35, h, fill=LIGHT_BLUE, line=BLUE, name=f"Response {index + 1}")
        text(slide, 5.3, top + 0.1, 4.05, h - 0.15, response, size=8.5, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    slide.notes_slide.notes_text_frame.text = SCRIPT[2]
    return slide


def slide_next(prs, layout):
    slide = new_slide(prs, layout)
    header(slide, "Next steps: Stage 1 by Oct 31, demo Nov 18", "11")
    phases = [
        ("Now – Oct 31", "Finish Stage 1", "Remaining 17 runs; N₈₀ vs k; hand vs object at equal k", ORANGE, LIGHT_ORANGE, True),
        ("Nov 1 – Nov 10", "Stage 2 · goal hidden", "Asymmetric actor-critic; ~27 runs; Recovery curve", BLUE, LIGHT_BLUE, False),
        ("Nov 10 – Nov 16", "Extension", "Second task (M2): rotation about a fixed axis; does it compress better?", PURPLE, WHITE, False),
        ("Nov 18 · end of Nov", "Demo · report", "Demo on Nov 18; final report at the end of November", NAVY, CARD, False),
    ]
    x, y, w, h, gap = 0.5, 1.4, 2.1, 1.4, 0.2
    for index, (dates, title, detail, ink, fill, current) in enumerate(phases):
        left = x + index * (w + gap)
        text(slide, left, y - 0.3, w, 0.22, dates, size=8.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
        box = card(slide, left, y, w, h, fill=fill, line=ink, weight=1.5 if current else 0.75, name=f"Phase {index + 1}")
        text(slide, left + 0.15, y + 0.15, w - 0.3, 0.3, title, size=11, color=ink if ink != NAVY else NAVY, bold=True)
        text(slide, left + 0.15, y + 0.55, w - 0.3, 1.05, detail, size=8.5, color=NAVY)
        if current:
            text(slide, left + 0.15, y + h - 0.32, w - 0.3, 0.2, "● in progress", size=8, color=DEEP_ORANGE, bold=True)
        if index < len(phases) - 1:
            arrow(slide, left + w + 0.02, y + h / 2, left + w + gap - 0.02, y + h / 2, color=MUTED)

    text(slide, 0.5, 3.4, 1.4, 0.3, "Priorities", size=10.5, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    priorities = [
        ("1", "Stage 1 conclusion: how many intent dimensions speed up learning", ORANGE),
        ("2", "Stage 2 Recovery curve: how much each intent recovers without the goal", BLUE),
        ("3", "M2 second task: does intent dimension grow with contact complexity?", PURPLE),
    ]
    for index, (number, label, color) in enumerate(priorities):
        top = 3.4 + index * 0.45
        circle_number(slide, 2.0, top, 0.3, number, color)
        text(slide, 2.45, top, 7.0, 0.3, label, size=9.5, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    slide.notes_slide.notes_text_frame.text = SCRIPT[3]
    return slide


SCRIPT_FILE = HERE / "milestone_script_2min.md"


def load_script() -> list[str]:
    """Speaker notes come from the script document: the paragraph under each "**Slide NN**" heading."""
    notes: list[str] = []
    for block in SCRIPT_FILE.read_text(encoding="utf-8").split("**Slide ")[1:]:
        _, _, body = block.partition("\n")
        notes.append(" ".join(body.split()))
    if len(notes) != 4:
        raise ValueError(f"{SCRIPT_FILE} must have four **Slide NN** sections, found {len(notes)}")
    return notes


SCRIPT = load_script()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", default=str(HERE.parent / "Project proposal" / "Team 34_project_proposal.pptx"))
    parser.add_argument("--output", default=str(HERE / "Team 34_milestone.pptx"))
    parser.add_argument("--data_dir", default=str(HERE / "data"))
    args = parser.parse_args()
    global DATA
    DATA = Path(args.data_dir).resolve()

    prs = Presentation(args.proposal)
    layout = prs.slides[1].slide_layout
    curves = load_learning_curves()
    pca = load_pca()
    slide_progress(prs, layout, curves)
    slide_results(prs, layout, curves, pca)
    slide_challenges(prs, layout)
    slide_next(prs, layout)
    prs.save(args.output)
    print(f"saved {args.output} (learning curves: {'filled' if curves else 'placeholder'})")


if __name__ == "__main__":
    main()
