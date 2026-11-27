#!/usr/bin/env python3
"""Build the final presentation (10 min) in the proposal / milestone visual style.

v2 structure (2026-11-13): short set-up slides (02-04), then one result per slide, each with a figure
(left) and a 2x1 video (8 s pair A, then 8 s pair B). The separate demo slide is gone.

Inputs: figures/*.png (make_figures.py); per-slide videos videos/slideNN.mp4 (+ optional
videos/slideNN_poster.png) — a labelled placeholder is drawn while a video is missing; speaker notes
from final_script_v2.md. Output: "Team 34_final_presentation.pptx". The proposal deck supplies the theme.
"""
from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
VID = HERE / "videos"
SCRIPT = HERE / "final_script_v2.md"
PROPOSAL = HERE.parent / "Project proposal" / "Team 34_project_proposal.pptx"
OUT = HERE / "Team 34_final_presentation.pptx"

NAVY = RGBColor(0x17, 0x21, 0x3A)
MUTED = RGBColor(0x62, 0x70, 0x8D)
BLUE = RGBColor(0x2C, 0x67, 0xF2)
ORANGE = RGBColor(0xD9, 0x60, 0x2F)
GREEN = RGBColor(0x1E, 0x9C, 0x77)
BORDER = RGBColor(0xD9, 0xE0, 0xEC)
CARD = RGBColor(0xF6, 0xF8, 0xFC)
LIGHT_BLUE = RGBColor(0xEA, 0xF0, 0xFF)
LIGHT_ORANGE = RGBColor(0xFF, 0xF0, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

# 2x1 video = two 616x584 tiles side by side (make_demo_video.py) -> aspect 1232:584
VIDEO_ASPECT = 1232 / 584


def text(slide, x, y, w, h, runs, size=12, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(5)
        pieces = para if isinstance(para, list) else [(para, {})]
        for piece, o in pieces:
            r = p.add_run()
            r.text = piece
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return box


def card(slide, x, y, w, h, fill=CARD, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = 0.08
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line
        b.line.width = Pt(0.75)
    b.shadow.inherit = False
    return b


def arrow(slide, x1, y1, x2, y2, color=BLUE):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}))


def header(slide, title, number, kicker=None):
    if kicker:
        text(slide, 0.5, 0.18, 6, 0.25, kicker.upper(), size=9, color=BLUE, bold=True)
    text(slide, 0.5, 0.38, 8.6, 0.55, title, size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 9.1, 0.3, 0.4, 0.3, number, size=8, color=BLUE, bold=True, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=12):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            head, body = it
            paras.append([(head + " ", {"bold": True}), (body, {})])
        else:
            paras.append([("• " + it, {})])
    return text(slide, x, y, w, h, paras, size=size)


def fit_picture(slide, name, x, y, w, h):
    """Largest size that fits the w x h box, centred horizontally, top-aligned."""
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y), Inches(pw), Inches(ph))


def video_slot(slide, number, x, y, w, pair_a, pair_b, stacked=False):
    """2x1 video (A: 0-8 s, B: 8-16 s); stacked=True: the two clips one above the other (1x2).
    Embeds videos/slideNN.mp4 if present, else a placeholder."""
    h = w * 4 / VIDEO_ASPECT if stacked else w / VIDEO_ASPECT
    video = VID / f"slide{number:02d}.mp4"
    poster = VID / f"slide{number:02d}_poster.png"
    if video.exists():
        kw = {"poster_frame_image": str(poster)} if poster.exists() else {}
        slide.shapes.add_movie(str(video), Inches(x), Inches(y), Inches(w), Inches(h), mime_type="video/mp4", **kw)
    else:
        b = card(slide, x, y, w, h, fill=CARD, line=MUTED, shape=MSO_SHAPE.RECTANGLE)
        b.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        label = (f"VIDEO PLACEHOLDER\nslide{number:02d}.mp4 · 1×2 stacked" if stacked
                 else f"VIDEO PLACEHOLDER · videos/slide{number:02d}.mp4 · 2×1, 8 s + 8 s")
        text(slide, x, y + 0.08, w, 0.4, label,
             size=7.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
        if stacked:
            half = h / 2
            slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 0.1), Inches(y + half),
                                       Inches(x + w - 0.1), Inches(y + half)).line.color.rgb = BORDER
            cells = [(x + 0.1, y + i * half + 0.3, w - 0.2, half - 0.4) for i in range(2)]
        else:
            half = w / 2
            slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + half), Inches(y + 0.1),
                                       Inches(x + half), Inches(y + h - 0.1)).line.color.rgb = BORDER
            cells = [(x + i * half + 0.1, y + 0.35, half - 0.2, h - 0.45) for i in range(2)]
        for i, (cx, cy, cw, ch) in enumerate(cells):
            text(slide, cx, cy, cw, ch,
                 [[("A  ", {"bold": True, "color": BLUE}), (pair_a[i], {})],
                  [("B  ", {"bold": True, "color": ORANGE}), (pair_b[i], {})]],
                 size=10, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return y + h


def takeaway(slide, msg, y=5.0):
    card(slide, 0.5, y, 9.0, 0.42, fill=LIGHT_BLUE, line=None)
    text(slide, 0.7, y, 8.6, 0.42, msg, size=12 if len(msg) <= 95 else 11, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)


def load_notes():
    src = SCRIPT.read_text(encoding="utf-8")
    notes = {}
    for m in re.finditer(r"\*\*Slide (\d+)\*\*[^\n]*\n(.*?)(?=\n\*\*Slide |\Z)", src, re.S):
        notes[int(m.group(1))] = m.group(2).strip()
    return notes


def fresh_deck():
    prs = Presentation(str(PROPOSAL))
    ids = prs.slides._sldIdLst
    for sld in list(ids):
        prs.part.drop_rel(sld.rId)
        ids.remove(sld)
    return prs


def new_slide(prs):
    layout = min(prs.slide_layouts, key=lambda l: len(l.placeholders))
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    return s


# Result-slide grids (takeaway bar at the bottom of both):
# side  (06, 07, 10): figure left, 2x1 video top right, bullets under the video.
# stack (08, 09; wide two-panel figures): figure left, 1x2 stacked video full height on the right,
#       bullets under the figure in two columns (fig_items left, items right).
FIG_X, FIG_Y, FIG_H = 0.35, 1.0, 3.85
SIDE_FIG_W, SIDE_COL_X, SIDE_COL_W = 4.35, 4.9, 4.7
STACK_VID_H = 3.85
STACK_VID_W = STACK_VID_H * VIDEO_ASPECT / 4
STACK_FIG_W, STACK_FIG_H = 9.6 - STACK_VID_W - 0.25 - FIG_X, 2.6


def result_slide(prs, number, kicker_no, title, figure, pair_a, pair_b, items, msg, label=None, size=10.5,
                 fig_items=None):
    s = new_slide(prs)
    header(s, title, kicker_no, "Results")
    if fig_items is None:
        pic = fit_picture(s, figure, FIG_X, FIG_Y, SIDE_FIG_W, FIG_H)
        bottom = video_slot(s, number, SIDE_COL_X, 1.05, SIDE_COL_W, pair_a, pair_b)
        bullets(s, SIDE_COL_X, bottom + 0.15, SIDE_COL_W, 4.9 - bottom - 0.15, items, size=size)
    else:
        pic = fit_picture(s, figure, FIG_X, FIG_Y, STACK_FIG_W, STACK_FIG_H)
        video_slot(s, number, 9.6 - STACK_VID_W, 1.0, STACK_VID_W, pair_a, pair_b, stacked=True)
        fb = (pic.top + pic.height) / 914400 + 0.2
        cw = (STACK_FIG_W - 0.3) / 2
        bullets(s, FIG_X + 0.1, fb, cw, 4.9 - fb, fig_items, size=size)
        bullets(s, FIG_X + 0.1 + cw + 0.2, fb, cw, 4.9 - fb, items, size=size)
    if label:
        text(s, FIG_X + 0.1, (pic.top + pic.height) / 914400 + 0.08, 2.5, 0.2, label, size=10, color=MUTED, bold=True)
    takeaway(s, msg, y=5.0)
    return s


def build():
    prs = fresh_deck()
    notes = load_notes()
    slides = []

    # 01 Title
    s = new_slide(prs); slides.append(s)
    card(s, 0, 0, 10, 5.625, fill=NAVY, line=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.7, 1.2, 8.6, 0.4, "LOW-DIMENSIONAL MANIPULATION INTENT · TEAM 34", size=11, color=RGBColor(0x9D, 0xB7, 0xF5), bold=True)
    text(s, 0.7, 1.7, 8.6, 1.3, "How little must a dexterous hand be told?", size=34, color=WHITE, bold=True)
    text(s, 0.7, 3.05, 8.6, 0.8, "When a slow planner talks to a fast robot hand: what should it say, and how often?",
         size=15, color=RGBColor(0xD9, 0xE0, 0xEC))
    text(s, 0.7, 4.6, 8.6, 0.3, "Final presentation · 18 Nov 2026", size=10, color=RGBColor(0x9D, 0xB7, 0xF5))

    # 02 Problem
    s = new_slide(prs); slides.append(s)
    header(s, "Problem: what should a slow planner tell a fast hand?", "01", "Problem and motivation")
    boxes = [("Planner", "language model or human\ndecides what to do · slow", LIGHT_ORANGE),
             ("Intent", "a few numbers\noften sent slowly", WHITE),
             ("Controller", "moves 20 joints\n60 times a second", LIGHT_BLUE)]
    for i, (t, sub, fill) in enumerate(boxes):
        x = 0.6 + i * 3.1
        card(s, x, 1.3, 2.5, 1.2, fill=fill, line=BORDER if fill == WHITE else None)
        text(s, x, 1.42, 2.5, 0.35, t, size=14, bold=True, align=PP_ALIGN.CENTER)
        text(s, x, 1.82, 2.5, 0.6, sub, size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
        if i < 2:
            arrow(s, x + 2.55, 1.9, x + 3.05, 1.9)
    qs = [("Which numbers?", "the hand's motion, the object's motion, or the goal"),
          ("How often?", "every step (60 Hz) down to twice a second"),
          ("How late, how coarse?", "delay and number of bits")]
    for i, (h, b) in enumerate(qs):
        x = 0.6 + i * 3.1
        text(s, x, 2.85, 2.6, 0.3, h, size=13, bold=True, color=BLUE)
        text(s, x, 3.17, 2.6, 0.7, b, size=11.5)
    takeaway(s, "Test bed: a simulated Shadow Hand turning a cube to a target orientation.", y=4.45)

    # 03 Approach
    s = new_slide(prs); slides.append(s)
    header(s, "Approach: hide the goal, send an intent instead", "02", "Approach")
    card(s, 0.5, 1.1, 2.6, 1.05, fill=LIGHT_ORANGE, line=None)
    text(s, 0.65, 1.2, 2.3, 0.9, [[("Expert", {"bold": True})], "sees the target, solves the task"], size=11.5)
    arrow(s, 3.15, 1.62, 3.65, 1.62)
    card(s, 3.7, 1.1, 2.6, 1.05, fill=WHITE)
    text(s, 3.85, 1.2, 2.3, 0.9, [[("Intent", {"bold": True})], "the only hint about the target"], size=11.5)
    arrow(s, 6.35, 1.62, 6.85, 1.62)
    card(s, 6.9, 1.1, 2.6, 1.05, fill=LIGHT_BLUE, line=None)
    text(s, 7.05, 1.2, 2.3, 0.9, [[("Student", {"bold": True})], "cannot see the target"], size=11.5)
    rows = [("Action level", "hand-k: the k most common whole-hand movement patterns of the expert · wrist-2 · wrist-2 + finger-k"),
            ("Object level", "obj-1 / obj-4: how far the cube still has to turn, or where it should be shortly"),
            ("Goal level", "goal-4: the target orientation itself (4 numbers) · goal-yaw on Task 2 (2 numbers)"),
            ("Worse channel", "send it less often, late, or with fewer bits")]
    for i, (k, v) in enumerate(rows):
        y = 2.35 + i * 0.42
        text(s, 0.6, y, 1.6, 0.38, k, size=11.5, bold=True, color=BLUE)
        text(s, 2.2, y, 7.3, 0.38, v, size=11.5)
    text(s, 0.6, 4.08, 8.9, 0.3, [[("Two tasks: ", {"bold": True}), ("Task 1 any 3-D orientation · Task 2 rotation about the vertical axis only", {})]], size=11.5)
    takeaway(s, "Score = recovery:   0 % = as good as no intent,   100 % = as good as the expert", y=4.55)

    # 04 Implementation
    s = new_slide(prs); slides.append(s)
    header(s, "Implementation", "03", "Implementation")
    text(s, 0.8, 1.6, 8.4, 2.4, [
        "Isaac Lab Shadow Hand simulation, all policies trained with PPO on the NUS SoC GPU cluster (Slurm).",
        "About 180 training runs, 3–5 seeds per condition; every result is measured on 1,024 test episodes.",
    ], size=18)

    # 05 Tasks (video only)
    s = new_slide(prs); slides.append(s)
    header(s, "Two tasks: what the hand does without an intent", "04", "Results")
    vw = 6.6
    bottom = video_slot(s, 5, (10 - vw) / 2, 1.0, vw,
                        ("Expert · Task 1", "Expert · Task 2"), ("No intent · Task 1", "No intent · Task 2"))
    for i, (h, b) in enumerate([("Task 1 · any 3-D orientation", "expert 26.1 goals / episode · no intent: holds the cube, recovery 0 %"),
                                ("Task 2 · about the vertical axis", "expert 31.1 · no intent spins the cube: 19.7 = 63 % of the expert")]):
        x = (10 - vw) / 2 + i * vw / 2
        text(s, x + 0.05, bottom + 0.1, vw / 2 - 0.1, 0.6, [[(h, {"bold": True, "color": BLUE})], b], size=10.5, align=PP_ALIGN.CENTER)
    takeaway(s, "How much the planner must say depends on the task.", y=5.0)

    # 06 Content
    slides.append(result_slide(
        prs, 6, "05", "Content: the expert's action carries the most", "m1_recovery.png",
        ("No intent", "hand-8"), ("hand-2", "hand-8"),
        [("Task 1, 60 Hz:", "hand-8 recovers 87 %, hand-2 50 %; the target itself (goal-4) 77 %; remaining turn (obj-1) 0 %."),
         ("Wrist vs fingers:", "wrist-2 29 %; + 2 finger patterns 55 % ≈ hand-2."),
         ("Not simple:", "8 patterns explain 81 % of the expert's motion; 2 explain > 80 % of human grasp postures (Santello et al., 1998).")],
        "At 60 Hz, more action components = more recovery, up to about 8. Beyond 8 components there is no further gain",
        label="(Goal hidden)"))

    # 07 Rate
    slides.append(result_slide(
        prs, 7, "06", "Rate: action intents collapse, goals degrade gently", "rate_sweep.png",
        ("60 Hz · hand-8", "60 Hz · goal-4"), ("2 Hz · hand-8", "2 Hz · goal-4"),
        [("5 seeds, trained at each rate:", "hand-8 80 / 19 / 3 % vs goal-4 78 / 71 / 44 % at 60 / 10 / 2 Hz."),
         ("Not under-training:", "twice as long does not close the gap."),
         ("Slow expert", "(sees the target, acts at 10 Hz): 12 %; at 2 Hz: 0 %.")],
        "The hand needs a fast loop; only what sits above it can be slow."))

    # 08 Why
    slides.append(result_slide(
        prs, 8, "07", "Why: what counts is how stale the signal gets, not the rate", "staleness_ab.png",
        ("goal-4 · 2 Hz", "goal-4 · 2 Hz"), ("slow motion", "refresh / success timeline"),
        [("Action:", "flips direction on ~25 % of steps → a held action is soon wrong."),
         ("Goal:", "changes only when reached, every ~0.4 s.")],
        "Not the rate itself: how much the signal changes between updates, against the task tolerance.",
        fig_items=[("A → B:", "against staleness instead of rate, all intents follow one trend (|ρ| 0.86 vs 0.50)."),
                   ("Exception:", "obj-4 moves with the cube; held at 10 Hz it is out of tolerance 66 % of the time (goal 11 %).")]))

    # 09 Delay and bits
    slides.append(result_slide(
        prs, 9, "08", "Delay and precision: the goal has a budget", "channel_delay_bits.png",
        ("hand-8 · no delay", "hand-8 · 0.1 s delay"), ("goal-4 · 0.1 s delay", "goal-4 · 1 s delay"),
        [("Delay:", "hand-8 → 0.3 % at 0.1 s (drops the cube in 91 % of episodes); goal-4 keeps most at 0.1 s, a third at 1 s.")],
        "≈ 32 bits per goal and well under 1 s: a planner that needs seconds per decision is outside it.",
        fig_items=[("Bits:", "8 per number lossless; 4 bits keeps 16 %: the rounding error is as large as the success tolerance.")]))

    # 10 Task 2
    slides.append(result_slide(
        prs, 10, "09", "Task 2: a stale goal is worse than no goal", "m2_rate.png",
        ("60 Hz · goal-yaw", "60 Hz · no intent"), ("2 Hz · goal-yaw", "2 Hz · no intent"),
        [("60 Hz:", "goal-yaw (2 numbers) 36 goals vs 20 without intent, above the expert (31)."),
         ("2 Hz:", "16 vs 20: the spinning trick wins."),
         ("hand-8:", "level with the goal at 60 Hz, worst of all at 2 Hz (12).")],
        "A slow goal can do more harm than sending nothing."))

    # 11 Discussion
    s = new_slide(prs); slides.append(s)
    header(s, "Discussion and conclusions", "10", "Discussion")
    cols = [("Achievements", BLUE, ["One pipeline that scores an intent by content, rate, delay and precision.",
                                    "Actions need a fast planner (< 0.1 s); goals tolerate a slow one, within ≈ 32 bits and < 1 s.",
                                    "Mechanism: how stale the signal gets, not the rate, predicts the loss."]),
            ("Limitations", ORANGE, ["Simulation only; one expert; 3–5 seeds per student.",
                                     "The goal intent is the true goal, not a planner's guess.",
                                     "Here even the goal changes faster than a language model answers."]),
            ("Next", GREEN, ["Send the goal only when it changes.",
                             "Send a short action sequence instead of one action.",
                             "Replace the oracle with a real slow planner."])]
    for i, (h, col, items) in enumerate(cols):
        x = 0.5 + i * 3.05
        card(s, x, 1.1, 2.85, 3.0, fill=CARD)
        text(s, x + 0.18, 1.22, 2.5, 0.3, h, size=13, bold=True, color=col)
        bullets(s, x + 0.18, 1.62, 2.5, 2.4, items, size=11)
    takeaway(s, "An intent survives a slow channel only while its value stays valid between updates.", y=4.4)

    # 12 Close
    s = new_slide(prs); slides.append(s)
    card(s, 0, 0, 10, 5.625, fill=NAVY, line=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.7, 1.9, 8.6, 1.0, "Before you choose an interface, measure how fast its signal goes stale.", size=26, color=WHITE, bold=True)
    text(s, 0.7, 3.3, 8.6, 0.4, "Thank you — questions?", size=16, color=RGBColor(0x9D, 0xB7, 0xF5))

    for i, s in enumerate(slides, start=1):
        if i in notes:
            s.notes_slide.notes_text_frame.text = notes[i]
    prs.save(str(OUT))
    print("wrote", OUT, len(slides), "slides")


if __name__ == "__main__":
    build()
