#!/usr/bin/env python3
"""Build the final presentation (10 min) in the proposal / milestone visual style.

Inputs: figures/*.png (make_figures.py), demo_video.mp4 + demo_poster.png (make_demo_video.py; optional,
a labelled placeholder is drawn when missing), speaker notes from final_script_10min.md.
Output: "Team 34_final_presentation.pptx". The proposal deck supplies the theme; its slides are removed.
"""
from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
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


def picture(slide, name, x, y, w=None, h=None):
    path = FIG / name
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def takeaway(slide, msg, y=5.0):
    card(slide, 0.5, y, 9.0, 0.42, fill=LIGHT_BLUE, line=None)
    text(slide, 0.7, y, 8.6, 0.42, msg, size=12, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)


def load_notes():
    src = (HERE / "final_script_10min.md").read_text(encoding="utf-8")
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


def build():
    prs = fresh_deck()
    notes = load_notes()
    slides = []

    # 01 Title
    s = new_slide(prs); slides.append(s)
    card(s, 0, 0, 10, 5.625, fill=NAVY, line=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.7, 1.2, 8.6, 0.4, "LOW-DIMENSIONAL MANIPULATION INTENT · TEAM 34", size=11, color=RGBColor(0x9D, 0xB7, 0xF5), bold=True)
    text(s, 0.7, 1.7, 8.6, 1.3, "How little must a dexterous hand be told?", size=34, color=WHITE, bold=True)
    text(s, 0.7, 3.05, 8.6, 0.8, "What a k-number intent should carry, and how often it must arrive — measured on Shadow Hand in-hand reorientation",
         size=15, color=RGBColor(0xD9, 0xE0, 0xEC))
    text(s, 0.7, 4.6, 8.6, 0.3, "Final presentation · 18 Nov 2026", size=10, color=RGBColor(0x9D, 0xB7, 0xF5))

    # 02 Problem
    s = new_slide(prs); slides.append(s)
    header(s, "Problem: what should a slow planner tell a fast hand?", "01", "Problem and motivation")
    boxes = [("High-level planner", "LLM · teleoperator · BCI\nseconds per decision", LIGHT_ORANGE),
             ("Intent", "k numbers\nevery N steps", WHITE),
             ("Low-level controller", "PPO policy\n60 Hz, 20 joints", LIGHT_BLUE)]
    for i, (t, sub, fill) in enumerate(boxes):
        x = 0.6 + i * 3.1
        card(s, x, 1.25, 2.5, 1.15, fill=fill, line=BORDER if fill == WHITE else None)
        text(s, x, 1.35, 2.5, 0.35, t, size=13, bold=True, align=PP_ALIGN.CENTER)
        text(s, x, 1.75, 2.5, 0.6, sub, size=11, color=MUTED, align=PP_ALIGN.CENTER)
        if i < 2:
            arrow(s, x + 2.55, 1.82, x + 3.05, 1.82)
    qs = [("Content", "which numbers? synergies, wrist, object motion, or the goal"),
          ("Rate", "how often? 60 Hz down to 2 Hz"),
          ("Channel", "how much delay and how few bits can it survive?")]
    for i, (h, b) in enumerate(qs):
        x = 0.6 + i * 3.1
        text(s, x, 2.7, 2.6, 0.3, h, size=13, bold=True, color=BLUE)
        text(s, x, 3.0, 2.6, 0.9, b, size=11.5)
    takeaway(s, "Prior systems pick one interface and never measure its information budget. We measure it.", y=4.45)

    # 03 Approach
    s = new_slide(prs); slides.append(s)
    header(s, "Approach: oracle teacher, goal-blind student, compressed intent", "02", "Approach")
    card(s, 0.5, 1.15, 2.6, 1.3, fill=LIGHT_ORANGE, line=None)
    text(s, 0.65, 1.25, 2.3, 1.1, [[("Expert (PPO)", {"bold": True})], "sees the goal orientation", "P_E = 26.1 successes / episode"], size=11.5)
    arrow(s, 3.15, 1.8, 3.65, 1.8)
    card(s, 3.7, 1.15, 2.6, 1.3, fill=WHITE)
    text(s, 3.85, 1.25, 2.3, 1.1, [[("Intent channel", {"bold": True})], "k numbers, refreshed every N steps", "(+ delay d, b bits at test)"], size=11.5)
    arrow(s, 6.35, 1.8, 6.85, 1.8)
    card(s, 6.9, 1.15, 2.6, 1.3, fill=LIGHT_BLUE, line=None)
    text(s, 7.05, 1.25, 2.3, 1.1, [[("Student (PPO)", {"bold": True})], "goal hidden from the actor", "critic still sees it"], size=11.5)
    rows = [("hand-k", "expert action on its top-k action PCs (k = 2, 4, 8, 20)"),
            ("wrist-2", "the two wrist commands"),
            ("obj-k", "object rotation ahead (axis-angle 1 or quaternion 4)"),
            ("goal-4 / goal-yaw", "goal quaternion / (sin ψ, cos ψ) of the yaw goal")]
    for i, (k, v) in enumerate(rows):
        y = 2.7 + i * 0.36
        text(s, 0.6, y, 1.8, 0.32, k, size=11.5, bold=True, color=BLUE)
        text(s, 2.4, y, 7.0, 0.32, v, size=11.5)
    takeaway(s, "Recovery = (P − P_blind) / (P_expert − P_blind):   0 = no intent,   1 = expert", y=4.35)

    # 04 Implementation
    s = new_slide(prs); slides.append(s)
    header(s, "Implementation: one pipeline, ~170 student runs on Slurm", "03", "Implementation")
    left = [("Simulator", "Isaac Lab 2.3 Shadow Hand cube repose, 60 Hz control, 20-D action"),
            ("Training", "PPO (RL-Games), 2048 envs, 5000 iterations = 164 M transitions; asymmetric critic"),
            ("Tasks", "Stage 1 (any orientation) and M2 (rotate about z)"),
            ("Scale", "~170 student runs (3 seeds per condition) on NUS SoC T4 GPUs, ≤ 8 at a time"),
            ("Evaluation", "every 200-iteration checkpoint × 1024 episodes, successes per episode")]
    bullets(s, 0.5, 1.15, 5.4, 3.3, left, size=11.5)
    card(s, 6.2, 1.15, 3.3, 2.75, fill=CARD)
    text(s, 6.4, 1.25, 3.0, 3.0, [[("Lessons", {"bold": True, "color": BLUE})],
                                  "Isaac Lab's success metric is a running average: it reads 15.5 where the expert achieves 26.1 per episode → own evaluator.",
                                  "Resuming training restores weights bit-exactly but restarts the learning rate → checked the transient before extending runs.",
                                  "Hold, delay and quantisation live in one IntentChannel module, shared by training, evaluation and video."], size=10.5)

    # 05 Demo
    s = new_slide(prs); slides.append(s)
    header(s, "Demo: what the hand does with each intent", "04", "Demonstration")
    video, poster = HERE / "demo_video.mp4", HERE / "demo_poster.png"
    if video.exists() and poster.exists():
        s.shapes.add_movie(str(video), Inches(1.35), Inches(1.05), Inches(7.3), Inches(4.1),
                           poster_frame_image=str(poster), mime_type="video/mp4")
    else:
        card(s, 1.35, 1.05, 7.3, 4.1, fill=CARD)
        text(s, 1.35, 1.05, 7.3, 4.1, "[demo video pending]", size=16, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.35, 5.2, 7.3, 0.3, "expert · blind · hand-8 60 Hz · hand-8 10 Hz · goal-4 60 vs 2 Hz · M2 blind vs goal-yaw",
         size=10, color=MUTED, align=PP_ALIGN.CENTER)

    # 06 Content
    s = new_slide(prs); slides.append(s)
    header(s, "Content: the expert's action carries the most", "05", "Results")
    picture(s, "m1_recovery.png", 0.4, 1.05, w=5.95)
    bullets(s, 6.55, 1.15, 3.0, 3.6, [
        ("Goal visible:", "no intent speeds learning (7 conditions, ANOVA p = 0.82)."),
        ("Goal hidden:", "hand-8 recovers 0.87; obj-1 nothing; wrist-2 0.29 (unlike in human grasps, the expert's wrist moves with its fingers)."),
        ("High-dimensional:", "8 action PCs = 81 % of variance (2 human synergies > 80 % of grasps)."),
    ], size=11)
    takeaway(s, "At 60 Hz, more action components = more recovery, up to about 8.", y=5.0)

    # 07 Rate
    s = new_slide(prs); slides.append(s)
    header(s, "Rate: action intents collapse, goals degrade gently", "06", "Results")
    picture(s, "rate_sweep.png", 0.4, 1.0, h=3.85)
    bullets(s, 6.55, 1.15, 3.0, 3.6, [
        ("hand-8:", "0.87 → 0.18 at 10 Hz, trained at that rate. The action flips sign on ~25 % of steps."),
        ("goal-4:", "0.77 → 0.71 → 0.65 → 0.43. The goal changes only on success, every ~0.4 s."),
        ("obj-4:", "the level in between is below goal-4 at every rate: 0.33 at 60 Hz, 0.02 at 2 Hz."),
    ], size=11)
    takeaway(s, "From 10 Hz down, goal-4 beats every other interface (p ≤ 0.012), the subgoal level included.", y=5.0)

    # 08 Why
    s = new_slide(prs); slides.append(s)
    header(s, "Why: what counts is how stale the signal gets, not the rate", "07", "Results")
    picture(s, "staleness_ab.png", 0.4, 1.15, w=6.2)
    bullets(s, 6.8, 1.15, 2.75, 3.6, [
        ("A:", "against the rate, every intent has its own curve."),
        ("B:", "against how wrong the held value is, the points follow one trend (|ρ| = 0.86 vs 0.50, 12 conditions)."),
        ("Exception:", "obj-4 drifts with the cube. Held at 10 Hz it is out of tolerance 66 % of the time; the goal 11 %."),
    ], size=10.5)
    takeaway(s, "Not the rate itself: how much the signal changes between updates, against the task tolerance.", y=5.0)

    # 09 Channel
    s = new_slide(prs); slides.append(s)
    header(s, "Channel: the goal has a budget, and it is tight", "08", "Results")
    picture(s, "channel_delay_bits.png", 0.45, 1.0, w=6.4)
    bullets(s, 7.05, 1.15, 2.5, 3.6, [
        ("Delay:", "hand-8 → 0.003 at 0.1 s (91 % falls). goal-4 loses half between 0.5 and 1 s; 21 % left at 2 s."),
        ("Bits:", "8 lossless; 4 bits keeps 16 %: a step of 0.13 ≈ the success tolerance."),
    ], size=11)
    takeaway(s, "≈ 32 bits per goal and < 1 s latency: a planner that needs seconds per decision is outside it.", y=5.0)

    # 10 Checks
    s = new_slide(prs); slides.append(s)
    header(s, "Checks: longer training and a second task", "09", "Results")
    picture(s, "training_length.png", 0.4, 1.05, w=4.4)
    picture(s, "m2_rate.png", 5.0, 1.05, w=4.6)
    text(s, 0.5, 4.3, 4.3, 0.8, "Resumed to 10 000 iterations: 60 / 5 Hz gain ≈ 0.12, 2 Hz only 0.04 → the rate gap is not under-training.", size=10.5)
    text(s, 5.1, 4.3, 4.4, 0.8, "M2: goal-yaw (2 numbers) ≈ goal-4. Blind spins the cube (63 % of expert). At 2 Hz both goal intents fall below blind (16 vs 20 successes): a stale goal is worse than none.", size=10.5)

    # 11 Discussion
    s = new_slide(prs); slides.append(s)
    header(s, "Discussion and conclusions", "10", "Discussion")
    cols = [("Achievements", BLUE, ["Reproducible pipeline that scores an intent by content, rate, delay and precision.",
                                    "Budget per level: actions need < 0.1 s; goals ≈ 32 bits and < 1 s.",
                                    "A mechanism: signal staleness, not rate, predicts the loss."]),
            ("Limitations", ORANGE, ["Simulation only; one expert seed, 3 student seeds.",
                                     "goal-4 is the hidden goal itself: we compare interface levels, not compression.",
                                     "Zero-order hold only: no action sequences, no event-triggered goals."]),
            ("Unexpected", GREEN, ["The subgoal level loses to the goal level at every rate.",
                                   "A stale goal can be worse than no goal (M2, 2 Hz).",
                                   "Here even goals change faster than a language model answers."])]
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
