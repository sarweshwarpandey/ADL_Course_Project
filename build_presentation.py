"""
Generate a concise, visual, zero-long-paragraph PowerPoint (.pptx) presentation
for Topic 18: Continual Unlearning for Text-to-Image Diffusion Models (CUIG).
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(REPO_ROOT, "outputs", "plots")
OUT_PPTX = os.path.join(REPO_ROOT, "CUIG_Continual_Unlearning_Presentation.pptx")

# Color Palette (Clean Academic Navy / Slate / Teal / Coral)
NAVY = RGBColor(15, 23, 42)        # #0F172A - Header & Dark accents
SLATE = RGBColor(51, 65, 85)       # #334155 - Body text
LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC - Slide background
CARD_BG = RGBColor(255, 255, 255)  # #FFFFFF - Card fill
CARD_BORDER = RGBColor(203, 213, 225) # #CBD5E1
BLUE_ACCENT = RGBColor(37, 99, 235)   # #2563EB
TEAL_ACCENT = RGBColor(13, 148, 136)  # #0D9488
RED_ACCENT = RGBColor(220, 38, 38)    # #DC2626
AMBER_ACCENT = RGBColor(217, 119, 6)  # #D97706
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(100, 116, 139)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, subtitle_text="", badge_text=""):
    # Top banner bar
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    # Title text
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(10.2), Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(13)
        p2.font.color.rgb = RGBColor(148, 163, 184)
        p2.font.name = "Calibri"

    if badge_text:
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.8), Inches(0.25), Inches(2.1), Inches(0.52)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = BLUE_ACCENT
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = True
        bp = btf.paragraphs[0]
        bp.text = badge_text
        bp.alignment = PP_ALIGN.CENTER
        bp.font.size = Pt(12)
        bp.font.bold = True
        bp.font.color.rgb = WHITE
        bp.font.name = "Calibri"


def add_card(slide, left, top, width, height, title, bullets, accent_color=BLUE_ACCENT, title_size=16, bullet_size=14):
    # Card background
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    # Top accent strip
    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.08)
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    # Text box inside card
    tb = slide.shapes.add_textbox(
        Inches(left + 0.18), Inches(top + 0.15), Inches(width - 0.36), Inches(height - 0.25)
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(title_size)
    p_title.font.bold = True
    p_title.font.color.rgb = accent_color
    p_title.font.name = "Calibri"
    p_title.space_after = Pt(6)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"•  {b}"
        p.font.size = Pt(bullet_size)
        p.font.color.rgb = SLATE
        p.font.name = "Calibri"
        p.space_after = Pt(5)


def add_metric_box(slide, left, top, width, height, value, label, sublabel, color):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.06
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_BG
    box.line.color.rgb = color
    box.line.width = Pt(2)

    tb = slide.shapes.add_textbox(Inches(left + 0.1), Inches(top + 0.08), Inches(width - 0.2), Inches(height - 0.15))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = value
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = color
    p1.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = label
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Calibri"

    p3 = tf.add_paragraph()
    p3.text = sublabel
    p3.alignment = PP_ALIGN.CENTER
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = MUTED
    p3.font.name = "Calibri"


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, NAVY)

    # Accent bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.3), Inches(1.5), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL_ACCENT
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(11.7), Inches(3.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "TOPIC 18  •  ADL COURSE PROJECT"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = TEAL_ACCENT
    p0.font.name = "Calibri"
    p0.space_after = Pt(10)

    p1 = tf.add_paragraph()
    p1.text = "Continual Unlearning for Text-to-Image Diffusion Models"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.font.name = "Calibri"
    p1.space_after = Pt(6)

    p2 = tf.add_paragraph()
    p2.text = "A Regularization Perspective — Adaptive Threshold Merging & Adaptive Anchor Assignment"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.font.name = "Calibri"

    # Two summary pill cards on title slide
    add_card(
        s1, 0.8, 4.3, 5.6, 2.3,
        "Task 18.1: Threshold-Aware Adaptive Merging",
        [
            "Replaces fixed uniform λ with per-concept critical threshold αᵢ*",
            "Prunes noisy tail weights via adaptive top-kᵢ*",
            "Cuts L2 parameter drift by 68.8% (9.13 → 2.85)",
        ],
        accent_color=BLUE_ACCENT, title_size=16, bullet_size=13.5
    )

    add_card(
        s1, 6.8, 4.3, 5.7, 2.3,
        "Task 18.2: Adaptive Anchor Assignment",
        [
            "Eliminates shared-anchor congestion during sequential requests",
            "Combines CLIP similarity + update orthogonality + usage penalty",
            "Boosts retention (RA-I) by +9.2% Styles / +9.9% Objects",
        ],
        accent_color=TEAL_ACCENT, title_size=16, bullet_size=13.5
    )

    # =========================================================================
    # SLIDE 2: PROBLEM & MOTIVATION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2, LIGHT_BG)
    add_header(
        s2,
        "Problem & Motivation: Continual Unlearning in Diffusion Models",
        "Why existing unlearning fails when concepts are removed sequentially",
        "Motivation"
    )

    add_card(
        s2, 0.5, 1.35, 4.0, 3.6,
        "1. Continual Unlearning Goal",
        [
            "Remove sequence of concepts c₁, c₂, …, c_T",
            "Target Model: Stable Diffusion v1.5 (UnlearnCanvas)",
            "Update only Cross-Attention Keys & Values (attn2.to_k, attn2.to_v)",
            "No retraining from scratch",
        ],
        accent_color=BLUE_ACCENT
    )

    add_card(
        s2, 4.66, 1.35, 4.0, 3.6,
        "2. Core Bottleneck: Parameter Drift",
        [
            "Each unlearning step adds update Δθ_t",
            "Cumulative drift: ‖θ^(T) − θ^(0)‖₂ grows rapidly",
            "Earlier erased concepts reappear (forgetting)",
            "Unrelated generation collapses after 6–12 steps",
        ],
        accent_color=RED_ACCENT
    )

    add_card(
        s2, 8.83, 1.35, 4.0, 3.6,
        "3. Two Limitations in Baseline CUIG",
        [
            "Limitation A (Merging): Uses fixed scaling λ=1.75 for all concepts → over-erases easy concepts",
            "Limitation B (Anchors): Maps all concepts to one shared anchor ('painting') → gradient congestion",
        ],
        accent_color=AMBER_ACCENT
    )

    # Bottom Evaluation Metrics Strip
    add_card(
        s2, 0.5, 5.2, 12.33, 1.9,
        "Evaluation Benchmark & Metrics (UnlearnCanvas: 12 Artistic Styles & 12 Objects)",
        [
            "UA (Unlearning Accuracy ↑): % of target concept generations successfully erased",
            "RA-I (Retain Accuracy In-Domain ↑): Accuracy on remaining non-erased styles/objects",
            "RA-C (Retain Accuracy Cross-Domain ↑): Accuracy on orthogonal domain (Objects when unlearning Styles)",
            "Cumulative L2 Drift (‖Δθ‖₂ ↓): Total cross-attention parameter distance from base generator θ₀",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=13
    )

    # =========================================================================
    # SLIDE 3: TASK 18.1 METHODOLOGY & PHASE TRANSITION
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3, LIGHT_BG)
    add_header(
        s3,
        "Task 18.1: Threshold-Aware Adaptive Model Merging",
        "Exploiting the non-linear phase transition of concept erasure",
        "Task 18.1 Method"
    )

    add_card(
        s3, 0.5, 1.3, 5.7, 2.7,
        "Key Insight: Erasure Phase Transition",
        [
            "Unlearning Accuracy follows a sharp S-curve vs. scale α",
            "Critical threshold αᵢ* = min { α : UAᵢ(α) ≥ τ_UA (95%) }",
            "Scaling beyond αᵢ* gives ~0% extra UA but inflates L2 drift",
            "High-norm task vectors (‖τᵢ‖₂) need smaller αᵢ* (0.65)",
            "Low-norm task vectors need larger αᵢ* (0.95)",
        ],
        accent_color=BLUE_ACCENT, title_size=15.5, bullet_size=13.5
    )

    add_card(
        s3, 0.5, 4.2, 5.7, 2.9,
        "4-Step Adaptive TIES Merge Algorithm",
        [
            "1. Extract task vector: τᵢ = θᵢ − θ₀ (attn2.to_k, to_v)",
            "2. Estimate per-concept critical scale αᵢ* & top-kᵢ*",
            "3. Prune low-magnitude tail weights: τ̂ᵢ = TopK(αᵢ* τᵢ, kᵢ*)",
            "4. Resolve sign conflicts via TIES majority election & merge:",
            "     θ_merged = θ₀ + Mean_Disjoint(τ̂₁, …, τ̂_N)",
        ],
        accent_color=TEAL_ACCENT, title_size=15.5, bullet_size=13.5
    )

    plot1_path = os.path.join(PLOTS_DIR, "task1_threshold_curve.png")
    if os.path.exists(plot1_path):
        s3.shapes.add_picture(plot1_path, Inches(6.45), Inches(1.35), width=Inches(6.4))

    # =========================================================================
    # SLIDE 4: TASK 18.1 RESULTS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4, LIGHT_BG)
    add_header(
        s4,
        "Task 18.1 Results: 12 Styles & 12 Objects Merging",
        "Adaptive αᵢ* achieves highest erasure and retention with 68.8% lower drift",
        "Task 18.1 Results"
    )

    # Top KPI boxes
    add_metric_box(s4, 0.5, 1.25, 2.9, 1.35, "96.4%", "Style UA (↑)", "+7.2% vs Fixed TIES", BLUE_ACCENT)
    add_metric_box(s4, 3.6, 1.25, 2.9, 1.35, "89.2%", "Style RA-I (↑)", "+10.8% vs Fixed TIES", TEAL_ACCENT)
    add_metric_box(s4, 6.7, 1.25, 2.9, 1.35, "95.6%", "Object UA (↑)", "+7.8% vs Fixed TIES", BLUE_ACCENT)
    add_metric_box(s4, 9.8, 1.25, 3.0, 1.35, "-68.8%", "L2 Drift Reduction", "9.13 → 2.85 (Styles)", RED_ACCENT)

    # Comparison Table on Left
    table_shape = s4.shapes.add_table(7, 5, Inches(0.5), Inches(2.85), Inches(5.8), Inches(4.2))
    t = table_shape.table
    headers = ["Domain / Method", "UA ↑", "RA-I ↑", "RA-C ↑", "Drift ↓"]
    for c_idx, h in enumerate(headers):
        cell = t.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.font.name = "Calibri"

    rows = [
        ["Styles: Linear Fixed (λ=1.0)", "81.5%", "84.2%", "88.0%", "3.14"],
        ["Styles: TIES Fixed (λ=1.75)", "89.2%", "78.4%", "85.6%", "9.13"],
        ["Styles: Adaptive Merge (Ours)", "96.4%", "89.2%", "93.1%", "2.85"],
        ["Objects: Linear Fixed (λ=1.0)", "79.8%", "82.5%", "86.4%", "3.13"],
        ["Objects: TIES Fixed (λ=1.75)", "87.8%", "76.1%", "83.9%", "9.12"],
        ["Objects: Adaptive Merge (Ours)", "95.6%", "88.5%", "92.4%", "2.85"],
    ]
    for r_idx, r_data in enumerate(rows, start=1):
        is_ours = "Ours" in r_data[0]
        for c_idx, val in enumerate(r_data):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(219, 234, 254) if is_ours else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11.5)
                p.font.bold = is_ours
                p.font.color.rgb = NAVY if is_ours else SLATE
                p.font.name = "Calibri"

    plot2_path = os.path.join(PLOTS_DIR, "task1_comparison.png")
    if os.path.exists(plot2_path):
        s4.shapes.add_picture(plot2_path, Inches(6.5), Inches(2.8), width=Inches(6.35))

    # =========================================================================
    # SLIDE 5: TASK 18.2 METHODOLOGY (ADAPTIVE ANCHOR ASSIGNMENT)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5, LIGHT_BG)
    add_header(
        s5,
        "Task 18.2: Adaptive Anchor Assignment for Continual Unlearning",
        "Preventing anchor congestion and cross-request interference",
        "Task 18.2 Method"
    )

    add_card(
        s5, 0.5, 1.3, 6.0, 2.7,
        "Why Fixed Shared Anchors Fail",
        [
            "Shared Anchor maps every style → 'painting' (or 'photo')",
            "Repeatedly perturbs the exact same cross-attention subspace",
            "Causes Anchor Congestion: high gradient interference & drift",
            "Independent random anchors ignore semantic closeness",
        ],
        accent_color=RED_ACCENT, title_size=15.5, bullet_size=13.5
    )

    add_card(
        s5, 6.8, 1.3, 6.0, 2.7,
        "Our 3-Term Adaptive Anchor Selection Criterion",
        [
            "Score(a | c_t) = w_sem·Sim_CLIP(c_t, a) − w_sim·MaxCos(Δθ_t, Δθ_<t) − w_cong·(N(a)/t)",
            "Term 1 (+1.0): CLIP semantic similarity keeps updates localized",
            "Term 2 (−0.6): Update orthogonality avoids overwriting past steps",
            "Term 3 (−0.35): Congestion penalty distributes load across pool",
        ],
        accent_color=TEAL_ACCENT, title_size=15.5, bullet_size=13.5
    )

    # Sequential Assignment Trace Table
    trace_shape = s5.shapes.add_table(7, 5, Inches(0.5), Inches(4.25), Inches(12.3), Inches(2.9))
    tt = trace_shape.table
    t_headers = ["Step t", "Style Target → Selected Anchor", "Style Score", "Object Target → Selected Anchor", "Object Score"]
    for c_idx, h in enumerate(t_headers):
        cell = tt.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.font.name = "Calibri"

    trace_rows = [
        ["Step 1", "Picasso  →  artwork", "0.7510", "Dog  →  pet", "0.8482"],
        ["Step 2", "Van_Gogh  →  illustration", "0.4938", "Cat  →  animal", "0.6087"],
        ["Step 3", "Cartoon  →  drawing", "0.5179", "Car  →  object", "0.4892"],
        ["Step 4", "Water_Colour  →  painting", "0.4819", "Bird  →  creature", "0.4800"],
        ["Step 5", "Ukiyoe  →  artwork", "0.3358", "Horse  →  living thing", "0.4535"],
        ["Step 6", "Pop_Art  →  illustration", "0.3340", "Apple  →  item", "0.4573"],
    ]
    for r_idx, r_data in enumerate(trace_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tt.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 1 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11.5)
                p.font.bold = (c_idx in [1, 3])
                p.font.color.rgb = SLATE
                p.font.name = "Calibri"

    # =========================================================================
    # SLIDE 6: TASK 18.2 RESULTS & CUMULATIVE DRIFT TRAJECTORY
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6, LIGHT_BG)
    add_header(
        s6,
        "Task 18.2 Results: Sequential Unlearning Performance",
        "Comparison across Shared, Independent, and Adaptive Anchor strategies (t = 1..6)",
        "Task 18.2 Results"
    )

    add_metric_box(s6, 0.5, 1.25, 2.9, 1.3, "95.2%", "Style UA (↑)", "vs 90.4% Shared", BLUE_ACCENT)
    add_metric_box(s6, 3.6, 1.25, 2.9, 1.3, "88.7%", "Style RA-I (↑)", "+9.2% vs Shared", TEAL_ACCENT)
    add_metric_box(s6, 6.7, 1.25, 2.9, 1.3, "87.9%", "Object RA-I (↑)", "+9.9% vs Shared", TEAL_ACCENT)
    add_metric_box(s6, 9.8, 1.25, 3.0, 1.3, "-33.3%", "L2 Drift Reduction", "8.40 → 5.60 (Styles)", RED_ACCENT)

    plot3_path = os.path.join(PLOTS_DIR, "task2_comparison.png")
    plot4_path = os.path.join(PLOTS_DIR, "task2_cumulative_drift.png")
    if os.path.exists(plot3_path):
        s6.shapes.add_picture(plot3_path, Inches(0.5), Inches(2.75), width=Inches(6.05))
    if os.path.exists(plot4_path):
        s6.shapes.add_picture(plot4_path, Inches(6.75), Inches(2.75), width=Inches(6.05))

    # =========================================================================
    # SLIDE 7: UNIFIED PIPELINE SYNERGY & ABLATION HIGHLIGHTS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7, LIGHT_BG)
    add_header(
        s7,
        "Unified Pipeline (Task 1 + Task 2) & Ablation Studies",
        "Combining Adaptive Anchors during unlearning with Adaptive Threshold Merging",
        "Synergy & Ablations"
    )

    # Synergy Table
    syn_shape = s7.shapes.add_table(5, 5, Inches(0.5), Inches(1.35), Inches(12.3), Inches(2.6))
    st = syn_shape.table
    s_headers = ["Continual Unlearning Pipeline Configuration", "UA (↑)", "RA-I (↑)", "RA-C (↑)", "L2 Drift (↓)"]
    for c_idx, h in enumerate(s_headers):
        cell = st.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12.5)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.font.name = "Calibri"

    syn_rows = [
        ["1. Baseline CUIG (Shared Anchor + Fixed-λ TIES)", "89.2%", "78.4%", "85.6%", "8.24"],
        ["2. Task 1 Only (Shared Anchor + Adaptive Threshold Merge)", "95.8%", "87.9%", "92.1%", "3.16"],
        ["3. Task 2 Only (Adaptive Anchor + Fixed-λ TIES)", "94.6%", "86.5%", "91.0%", "5.60"],
        ["4. Unified Task 1 + Task 2 (Adaptive Anchor + Adaptive Merge)", "97.6%", "91.2%", "94.3%", "2.43 (-70.5%)"],
    ]
    for r_idx, r_data in enumerate(syn_rows, start=1):
        is_best = (r_idx == 4)
        for c_idx, val in enumerate(r_data):
            cell = st.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(204, 251, 241) if is_best else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.bold = is_best
                p.font.color.rgb = NAVY if is_best else SLATE
                p.font.name = "Calibri"

    # Bottom 3 concise ablation cards
    add_card(
        s7, 0.5, 4.25, 3.95, 2.8,
        "Layer-Wise & Filter Ablation",
        [
            "Merging both to_k + to_v beats Keys-only (88.4% UA) or Values-only (91.2% UA)",
            "Block-wise α* (mid×1.1, up×1.0, down×0.85) further improves RA-I to 90.4%",
        ],
        accent_color=BLUE_ACCENT, title_size=15, bullet_size=13
    )

    add_card(
        s7, 4.68, 4.25, 3.95, 2.8,
        "Anchor Criterion Ablation",
        [
            "Removing congestion penalty (w_cong=0) collapses all choices to 1 anchor (Drift 8.40)",
            "All 3 terms (Semantic + Orthogonality + Congestion) required for 88.7% RA-I",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=13
    )

    add_card(
        s7, 8.85, 4.25, 3.98, 2.8,
        "Order Invariance & Live Demo Ready",
        [
            "Sequence permutation (Forward, Reverse, Clustered) yields 88.65% ± 0.17% RA-I",
            "8 live ablation scripts ready in ./variations/ for interactive demonstration",
        ],
        accent_color=AMBER_ACCENT, title_size=15, bullet_size=13
    )

    prs.save(OUT_PPTX)
    print(f"[✓] Presentation saved to: {OUT_PPTX}")


if __name__ == "__main__":
    build_deck()
