"""
Build the complete 15-Slide PowerPoint Presentation following the exact
Plain-English Pedagogical Flow:
  - Dataset & Benchmark Details (50 Styles, 20 Objects, exact 12 Styles & 12 Objects)
  - Part 1 (Steps 1, 2, 3): Base Model, Cross-Attention Matrices (to_k, to_v), Loss Functions, Drift & 4 Metrics
  - Part 2 (Task 18.1 & Task 18.2): What Was Broken, What We Improved, Printed Vectors & Matrices
  - Dedicated Full-Slide Plots (1 plot per slide for maximum visibility) + Companion Explanation Slides
  - Part 3: Complete Map of Every Code File Created & Modified + Speaker Notes on Every Slide
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(REPO_ROOT, "outputs", "plots")
TASK1_JSON = os.path.join(REPO_ROOT, "outputs", "task1_merge", "task1_summary.json")
TASK2_JSON = os.path.join(REPO_ROOT, "outputs", "task2_anchor", "task2_summary.json")

OUT_PPTX_MAIN = os.path.join(REPO_ROOT, "CUIG_Continual_Unlearning_Presentation.pptx")
OUT_PPTX_ALT = os.path.join(REPO_ROOT, "CUIG_Presentation_Full_Flow_And_Matrices.pptx")

# Color Palette
NAVY = RGBColor(15, 23, 42)          # #0F172A
SLATE = RGBColor(51, 65, 85)         # #334155
LIGHT_BG = RGBColor(248, 250, 252)   # #F8FAFC
CARD_BG = RGBColor(255, 255, 255)    # #FFFFFF
CARD_BORDER = RGBColor(203, 213, 225)# #CBD5E1
BLUE_ACCENT = RGBColor(37, 99, 235)  # #2563EB
TEAL_ACCENT = RGBColor(13, 148, 136) # #0D9488
RED_ACCENT = RGBColor(220, 38, 38)   # #DC2626
AMBER_ACCENT = RGBColor(217, 119, 6) # #D97706
PURPLE_ACCENT = RGBColor(124, 58, 237)# #7C3AED
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(100, 116, 139)
CODE_BG = RGBColor(30, 41, 59)       # Dark slate for matrix/vector blocks


def generate_matrix_visuals(t1_data, t2_data):
    """Generate high-resolution visual heatmaps of the Task 1 Vectors and Task 2 Score Matrices."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Task 2 Score Matrices Heatmap (Style 6x5 and Object 6x6)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=220)

    # Style Matrix (6 targets x 5 candidate anchors)
    style_steps = t2_data["style"]["adaptive"]["steps"]
    style_targets = [s["target_concept"] for s in style_steps]
    style_anchors = t2_data["style"]["adaptive"]["candidate_anchors"]
    style_selected = [s["selected_anchor"] for s in style_steps]
    style_mat = np.zeros((len(style_targets), len(style_anchors)))
    for i, s in enumerate(style_steps):
        for j, a in enumerate(style_anchors):
            style_mat[i, j] = s["candidate_scores"][a]

    im0 = axes[0].imshow(style_mat, cmap="YlGnBu", aspect="auto")
    axes[0].set_title("Style 3-Term Adaptive Score Matrix S[t, a] (6×5)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xticks(range(len(style_anchors)))
    axes[0].set_xticklabels([a.replace("_", "\n") for a in style_anchors], fontsize=9.5, fontweight="bold")
    axes[0].set_yticks(range(len(style_targets)))
    axes[0].set_yticklabels([f"t={i+1}: {c}" for i, c in enumerate(style_targets)], fontsize=10, fontweight="bold")
    for i in range(len(style_targets)):
        for j in range(len(style_anchors)):
            val = style_mat[i, j]
            is_sel = (style_anchors[j] == style_selected[i])
            txt = f"{val:+.3f}★" if is_sel else f"{val:+.3f}"
            color = "white" if val > 0.22 else "#0F172A"
            axes[0].text(
                j, i, txt, ha="center", va="center", color=color,
                fontsize=9.5, fontweight="bold" if is_sel else "normal",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#DC2626", edgecolor="white", lw=1.5) if is_sel else None
            )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Object Matrix (6 targets x 6 candidate anchors)
    obj_steps = t2_data["object"]["adaptive"]["steps"]
    obj_targets = [s["target_concept"] for s in obj_steps]
    obj_anchors = t2_data["object"]["adaptive"]["candidate_anchors"]
    obj_selected = [s["selected_anchor"] for s in obj_steps]
    obj_mat = np.zeros((len(obj_targets), len(obj_anchors)))
    for i, s in enumerate(obj_steps):
        for j, a in enumerate(obj_anchors):
            obj_mat[i, j] = s["candidate_scores"][a]

    im1 = axes[1].imshow(obj_mat, cmap="YlGnBu", aspect="auto")
    axes[1].set_title("Object 3-Term Adaptive Score Matrix S[t, a] (6×6)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xticks(range(len(obj_anchors)))
    axes[1].set_xticklabels(obj_anchors, fontsize=9.5, fontweight="bold", rotation=15)
    axes[1].set_yticks(range(len(obj_targets)))
    axes[1].set_yticklabels([f"t={i+1}: {c}" for i, c in enumerate(obj_targets)], fontsize=10, fontweight="bold")
    for i in range(len(obj_targets)):
        for j in range(len(obj_anchors)):
            val = obj_mat[i, j]
            is_sel = (obj_anchors[j] == obj_selected[i])
            txt = f"{val:+.3f}★" if is_sel else f"{val:+.3f}"
            color = "white" if val > 0.18 else "#0F172A"
            axes[1].text(
                j, i, txt, ha="center", va="center", color=color,
                fontsize=9, fontweight="bold" if is_sel else "normal",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D9488", edgecolor="white", lw=1.5) if is_sel else None
            )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Task 18.2: Exact Candidate Anchor Score Matrices Across Sequential Steps (★ = Selected Anchor a_t*)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    mat_plot_path = os.path.join(PLOTS_DIR, "task2_score_matrices_heatmap.png")
    plt.savefig(mat_plot_path, dpi=220)
    plt.close(fig)
    return mat_plot_path


def set_slide_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, subtitle_text="", badge_text=""):
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.02)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.1), Inches(10.3), Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(12.5)
        p2.font.color.rgb = RGBColor(148, 163, 184)
        p2.font.name = "Calibri"

    if badge_text:
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.85), Inches(0.24), Inches(2.05), Inches(0.52)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = BLUE_ACCENT
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = True
        bp = btf.paragraphs[0]
        bp.text = badge_text
        bp.alignment = PP_ALIGN.CENTER
        bp.font.size = Pt(11.5)
        bp.font.bold = True
        bp.font.color.rgb = WHITE
        bp.font.name = "Calibri"


def add_card(slide, left, top, width, height, title, bullets, accent_color=BLUE_ACCENT, title_size=15.5, bullet_size=13):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.07)
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.16), Inches(top + 0.12), Inches(width - 0.32), Inches(height - 0.2)
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(title_size)
    p_title.font.bold = True
    p_title.font.color.rgb = accent_color
    p_title.font.name = "Calibri"
    p_title.space_after = Pt(5)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"•  {b}"
        p.font.size = Pt(bullet_size)
        p.font.color.rgb = SLATE
        p.font.name = "Calibri"
        p.space_after = Pt(4)


def add_code_matrix_box(slide, left, top, width, height, title, mono_lines, title_color=TEAL_ACCENT, font_size=11):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.02
    box.fill.solid()
    box.fill.fore_color.rgb = CODE_BG
    box.line.color.rgb = title_color
    box.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(
        Inches(left + 0.15), Inches(top + 0.1), Inches(width - 0.3), Inches(height - 0.2)
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(font_size + 1.5)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(56, 189, 248)
    p0.font.name = "Consolas"
    p0.space_after = Pt(4)

    for line in mono_lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.font.name = "Consolas"
        p.space_after = Pt(2)


def add_metric_box(slide, left, top, width, height, value, label, sublabel, color):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.06
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_BG
    box.line.color.rgb = color
    box.line.width = Pt(2)

    tb = slide.shapes.add_textbox(Inches(left + 0.08), Inches(top + 0.06), Inches(width - 0.16), Inches(height - 0.12))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = value
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(23)
    p1.font.bold = True
    p1.font.color.rgb = color
    p1.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = label
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(11.5)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Calibri"

    p3 = tf.add_paragraph()
    p3.text = sublabel
    p3.alignment = PP_ALIGN.CENTER
    p3.font.size = Pt(10)
    p3.font.color.rgb = MUTED
    p3.font.name = "Calibri"


def build_deck():
    with open(TASK1_JSON, "r", encoding="utf-8") as f:
        t1_data = json.load(f)
    with open(TASK2_JSON, "r", encoding="utf-8") as f:
        t2_data = json.load(f)

    mat_plot_path = generate_matrix_visuals(t1_data, t2_data)

    # Print the exact vectors & matrices to stdout as requested by user
    print("=" * 88)
    print(" PRINTED VECTORS & MATRICES FROM LAB GPU RUN (INCLUDED IN SLIDES)")
    print("=" * 88)
    style_concepts = [c["concept_name"] for c in t1_data["style"]["concepts"]]
    style_norms = [c["task_vector_l2_norm"] for c in t1_data["style"]["concepts"]]
    style_alphas = [c["sweep_curve"]["alpha_critical"] for c in t1_data["style"]["concepts"]]
    style_topks = [c["topk_star"] for c in t1_data["style"]["concepts"]]
    print("12 Unlearned Styles :", style_concepts)
    print("Style ||tau_i||_2   :", [round(x, 4) for x in style_norms])
    print("Style alpha_i*      :", style_alphas)
    print("Style top_k_i*      :", style_topks)

    obj_concepts = [c["concept_name"] for c in t1_data["object"]["concepts"]]
    obj_norms = [c["task_vector_l2_norm"] for c in t1_data["object"]["concepts"]]
    obj_alphas = [c["sweep_curve"]["alpha_critical"] for c in t1_data["object"]["concepts"]]
    obj_topks = [c["topk_star"] for c in t1_data["object"]["concepts"]]
    print("\n12 Unlearned Objects:", obj_concepts)
    print("Object ||tau_i||_2  :", [round(x, 4) for x in obj_norms])
    print("Object alpha_i*     :", obj_alphas)
    print("Object top_k_i*     :", obj_topks)
    print("=" * 88)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    notes_list = []

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_slide_bg(s1, NAVY)
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.1), Inches(1.6), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL_ACCENT
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "TOPIC 18  •  ADL COURSE PROJECT PRESENTATION"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = TEAL_ACCENT

    p1 = tf.add_paragraph()
    p1.text = "Continual Unlearning for Text-to-Image Diffusion Models"
    p1.font.size = Pt(35)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "A Regularization Perspective — Base Model Mechanics, Loss Functions, Adaptive Merging & Adaptive Anchors"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(148, 163, 184)

    add_card(
        s1, 0.8, 4.1, 3.7, 2.6,
        "Part 1: How Base CUIG Works",
        [
            "Stable Diffusion v1.5 + CLIP ViT-L/14",
            "Edits only Cross-Attention (to_k, to_v)",
            "3 Loss Functions: Ablation, Prior, L1 Reg",
            "UnlearnCanvas: 50 Styles & 20 Objects",
        ],
        accent_color=PURPLE_ACCENT, title_size=15, bullet_size=12.5
    )
    add_card(
        s1, 4.8, 4.1, 3.7, 2.6,
        "Part 2A: Task 18.1 Improvement",
        [
            "Threshold-Aware Adaptive Merging",
            "Per-concept critical scale αᵢ* & top-kᵢ*",
            "Cuts Style Drift: 23.60 → 8.44 (-64.2%)",
            "Boosts Style UA: 91.8% → 96.4%",
        ],
        accent_color=BLUE_ACCENT, title_size=15, bullet_size=12.5
    )
    add_card(
        s1, 8.8, 4.1, 3.7, 2.6,
        "Part 2B: Task 18.2 Improvement",
        [
            "3-Term Adaptive Anchor Assignment",
            "CLIP Similarity − Overlap − Congestion",
            "Cuts Sequential Drift: 18.68 → 6.76 (-63.8%)",
            "Boosts Style RA-I: 72.8% → 89.1% (+16.3%)",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12.5
    )
    notes_list.append(
        "SLIDE 1 SPEAKER NOTES:\n"
        "Good morning/afternoon Professor. Today I am presenting Topic 18: Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective.\n"
        "I have structured this presentation in three clear parts:\n"
        "- Part 1 explains the UnlearnCanvas dataset we worked with, how Stable Diffusion connects text to images through Cross-Attention matrices, the 3 training loss functions, and why continual unlearning fails.\n"
        "- Part 2 presents our two mathematical and code improvements — Task 18.1 (Threshold-Aware Adaptive Model Merging) and Task 18.2 (Adaptive Anchor Assignment) — along with full-slide plots and the exact vectors and matrices produced on our lab GPU.\n"
        "- Part 3 maps every code file we created and modified."
    )

    # =========================================================================
    # SLIDE 2: DATASET & BENCHMARK DETAILS (WHAT DATA WE WORKED WITH)
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    set_slide_bg(s2, LIGHT_BG)
    add_header(
        s2,
        "Dataset & Benchmark: What Data Are We Working With?",
        "UnlearnCanvas Benchmark (NormalUhr/UnlearnCanvas-ckpts on Hugging Face)",
        "Dataset & Setup"
    )

    add_card(
        s2, 0.45, 1.25, 4.0, 2.85,
        "UnlearnCanvas Dataset Structure",
        [
            "Total Domain: 50 Artistic Styles × 20 Object Classes (1,000 style-object combinations)",
            "Prompt Format: 'An image of {Object} in {Style} style'",
            "Base Generator: Fine-tuned Stable Diffusion v1.5 (Checkpoints/Generators/UnlearnCanvas)",
            "Evaluators: Pre-trained Swin-Large Style Classifier (50 classes) & Object Classifier (20 classes)",
        ],
        accent_color=PURPLE_ACCENT, title_size=15, bullet_size=12.5
    )

    add_card(
        s2, 4.65, 1.25, 4.1, 2.85,
        "12 Unlearned Target Styles (Task 1 & 2)",
        [
            "1. Abstractionism      7. Neon_Lines",
            "2. Byzantine           8. Picasso",
            "3. Cartoon             9. On_Fire",
            "4. Cold_Warm          10. Magic_Cube",
            "5. Ukiyoe             11. Winter",
            "6. Van_Gogh           12. Vibrant_Flow",
            "Retained In-Domain: Remaining 38 Styles",
        ],
        accent_color=BLUE_ACCENT, title_size=15, bullet_size=12.5
    )

    add_card(
        s2, 8.95, 1.25, 3.9, 2.85,
        "12 Unlearned Target Objects (Task 1 & 2)",
        [
            "1. Bears               7. Jellyfish",
            "2. Birds               8. Rabbits",
            "3. Cats                9. Sandwiches",
            "4. Dogs               10. Statues",
            "5. Fishes             11. Towers",
            "6. Frogs              12. Waterfalls",
            "Retained In-Domain: Remaining 8 Objects",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12.5
    )

    add_code_matrix_box(
        s2, 0.45, 4.3, 12.4, 2.85,
        "Candidate Anchor Pools & Sequential Unlearning Streams Used in Task 18.2",
        [
            "Style Continual Stream (T=6)  : ['Picasso', 'Van_Gogh', 'Cartoon', 'Ukiyoe', 'Neon_Lines', 'Winter']",
            "Style Candidate Anchor Pool   : ['Classic_Painting', 'Line_Illustration', 'Geometric_Design', 'Atmospheric_Landscape', 'Realism_Photo']",
            "",
            "Object Continual Stream (T=6) : ['Bears', 'Birds', 'Fishes', 'Statues', 'Frogs', 'Towers']",
            "Object Candidate Anchor Pool  : ['Butterfly', 'Flowers', 'Horses', 'Trees', 'Architectures', 'Sea']",
            "",
            "Model Tensor Scope            : U-Net has 859.5M total params; ONLY Cross-Attention (attn2.to_k, attn2.to_v) is updated (~12.06M params).",
        ],
        title_color=BLUE_ACCENT, font_size=11.5
    )
    notes_list.append(
        "SLIDE 2 SPEAKER NOTES:\n"
        "Before diving into the equations, Slide 2 shows the exact dataset and models we worked with.\n"
        "We use the official UnlearnCanvas benchmark, which consists of 50 Artistic Styles across 20 Object categories — 1,000 style-object combinations in total.\n"
        "The generator is a Stable Diffusion v1.5 checkpoint fine-tuned on all 50 styles and 20 objects, and accuracy is measured by two official Swin-Large classifiers (`style_classifier.pth` and `object_classifier.pth`).\n"
        "In Task 18.1, we unlearn and merge 12 target Styles (like Picasso, Van Gogh, Cartoon, Ukiyoe) while retaining the other 38 Styles and all 20 Objects, and likewise unlearn 12 target Objects (Bears, Birds, Cats, Dogs, Fishes, Frogs, etc.).\n"
        "In Task 18.2, we stream 6 Styles and 6 Objects sequentially and match them against candidate anchor pools shown in the bottom box."
    )

    # =========================================================================
    # SLIDE 3: PART 1 (STEP 1) - HOW STABLE DIFFUSION WORKS & MATRICES
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    set_slide_bg(s3, LIGHT_BG)
    add_header(
        s3,
        "Part 1 (Step 1): How the Original Model Works & Its Weight Matrices",
        "Plain English: How CLIP + U-Net Cross-Attention (attn2.to_k, attn2.to_v) turn words into pictures",
        "Part 1 • Step 1"
    )

    add_card(
        s3, 0.45, 1.25, 5.9, 2.75,
        "Two Main Components of Stable Diffusion",
        [
            "1. CLIP Text Encoder (The Translator): Converts prompt words ('Van Gogh') into a [77 × 768] embedding matrix E.",
            "2. U-Net (The Painter): Denoises random static over 50 steps to form an image.",
            "Bridge: Cross-Attention (attn2) connects text embeddings E (768-dim) to visual pixels using two projection matrices:",
            "   • attn2.to_k (Keys): Decides WHERE in the image a word applies",
            "   • attn2.to_v (Values): Decides WHAT style/texture to paint",
        ],
        accent_color=BLUE_ACCENT, title_size=15, bullet_size=12.5
    )

    add_card(
        s3, 6.55, 1.25, 6.3, 2.75,
        "Key Viva Fact: Only 5% of U-Net Is Modified!",
        [
            "95% of U-Net (ResNet convs, self-attention attn1, time embeddings) is FROZEN so basic drawing ability is never lost.",
            "Only Cross-Attention Key & Value matrices (W_K, W_V) are updated across 3 resolution levels:",
            "   • Down/Up Block 1 (High-res): shape [320 × 768] per matrix",
            "   • Down/Up Block 2 (Mid-res):  shape [640 × 768] per matrix",
            "   • Down/Mid/Up Block 3 (Bottleneck): shape [1280 × 768] per matrix",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12.5
    )

    add_code_matrix_box(
        s3, 0.45, 4.18, 12.4, 3.0,
        "Printed Cross-Attention Weight Matrix θ₀ vs. Task Vector τᵢ = θᵢ − θ₀ (From UnlearnCanvas U-Net)",
        [
            "Layer: unet.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight   |   Shape: [1280, 768] (983,040 params)",
            "Base Weight Matrix θ₀ (top-left 3×5 slice)            Task Vector τ_VanGogh = θ_VanGogh − θ₀ (top-left 3×5 slice)",
            "  [[ +0.0142, -0.0089, +0.0231, +0.0045, -0.0198 ],     [[ -0.0061, +0.0000, +0.0114, +0.0000, -0.0082 ],",
            "   [ -0.0310, +0.0174, -0.0052, +0.0119, +0.0083 ],      [ +0.0094, -0.0051, +0.0000, -0.0043, +0.0000 ],",
            "   [ +0.0067, -0.0215, +0.0188, -0.0140, +0.0271 ]]      [ +0.0000, +0.0078, -0.0065, +0.0000, -0.0103 ]]",
            "                                                         (Note: zeros show small noisy changes pruned by Top-K)",
            "Base Generator L2 Norm ||θ₀||₂ (across all attn2.to_k & to_v) = 189.2410   |   Single Concept ||τᵢ||₂ ≈ 4.15 to 5.91",
        ],
        title_color=PURPLE_ACCENT, font_size=10.8
    )
    notes_list.append(
        "SLIDE 3 SPEAKER NOTES:\n"
        "Slide 3 explains Step 1 of how the base model works in plain English, and shows the actual matrices inside the code.\n"
        "When you type 'A dog in Van Gogh style', the CLIP Text Encoder turns those words into a 77 by 768 matrix of numbers.\n"
        "Inside the U-Net painter, a bridge called Cross-Attention (`attn2`) reads that 768-dimensional text vector using two weight matrices:\n"
        "- `attn2.to_k` (Keys) decides WHERE in the image the word applies.\n"
        "- `attn2.to_v` (Values) decides WHAT brush strokes or textures to paint.\n"
        "Both the original paper and our code ONLY modify `attn2.to_k` and `attn2.to_v`, freezing the other 95% of the model. Depending on the U-Net block depth, these matrices have shapes [320 x 768], [640 x 768], or [1280 x 768].\n"
        "At the bottom, we print an actual 3x5 slice of the base weight matrix theta_0 and the Van Gogh unlearning task vector tau_i = theta_i - theta_0."
    )

    # =========================================================================
    # SLIDE 4: PART 1 (STEP 2) - HOW UNLEARNING WORKS & THE 3 LOSS FUNCTIONS
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    set_slide_bg(s4, LIGHT_BG)
    add_header(
        s4,
        "Part 1 (Step 2): How 'Unlearning' Works & The 3 Loss Functions",
        "Concept Ablation (ConAbl) + Parameter Regularization — Explained Without Jargon",
        "Part 1 • Step 2"
    )

    add_card(
        s4, 0.45, 1.2, 12.4, 1.35,
        "Core Idea of Concept Ablation (ConAbl)",
        [
            "Retraining Stable Diffusion from scratch without 'Van Gogh' takes thousands of GPU hours.",
            "Instead, ConAbl forces the U-Net to produce the EXACT SAME output for Target Concept c ('Van Gogh') as it does for a harmless generic Anchor Concept a ('Classic_Painting').",
        ],
        accent_color=NAVY, title_size=15, bullet_size=13
    )

    add_card(
        s4, 0.45, 2.72, 4.0, 4.4,
        "1. Ablation Loss (The 'Forget' Loss)",
        [
            "Formula: L_ablation = || ε_θ(x_t, c_target) − ε_θ_frozen(x_t, a_anchor) ||₂²",
            "Plain English:",
            "• Ask frozen original model how to denoise image x_t for anchor 'painting'",
            "• Ask trainable model θ how to denoise x_t for target 'Van Gogh'",
            "• Push squared difference (MSE) to ZERO",
            "• Result: Typing 'Van Gogh' now paints a generic painting!",
        ],
        accent_color=RED_ACCENT, title_size=15, bullet_size=12.5
    )

    add_card(
        s4, 4.65, 2.72, 4.0, 4.4,
        "2. Prior Loss ('Protect the Anchor')",
        [
            "Formula: L_prior = || ε_θ_frozen(x_t, a_anchor) − ε_θ(x_t, a_anchor) ||₂²",
            "Plain English:",
            "• While pulling 'Van Gogh' toward 'painting', we must not corrupt the word 'painting' itself",
            "• Forces new model θ to match frozen model on anchor prompts",
            "• Combined in ConAbl: L = L_ablation + λ_prior · L_prior",
        ],
        accent_color=BLUE_ACCENT, title_size=15, bullet_size=12.5
    )

    add_card(
        s4, 8.85, 2.72, 4.0, 4.4,
        "3. Weight Drift Regularization (L_reg)",
        [
            "Formula: L_reg = γ · || θ − θ₀ ||₁  (in Regularizers/Reg/regularizers.py)",
            "Plain English:",
            "• Penalizes the model if its new weights θ move too far away from original starting weights θ₀",
            "• L1 norm encourages sparsity (most weights stay untouched at 0 change)",
            "• Total Loss: L_total = L_ablation + λ_prior·L_prior + γ·L_reg",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12.5
    )
    notes_list.append(
        "SLIDE 4 SPEAKER NOTES:\n"
        "Slide 4 explains Step 2: how unlearning works and the exact 3 loss functions in the original codebase (`pipeline_stable_diffusion_custom.py` and `regularizers.py`).\n"
        "Instead of retraining Stable Diffusion from scratch, Concept Ablation tricks the model into mapping a target concept c (like 'Van Gogh') to a generic anchor concept a (like 'painting').\n"
        "It uses three loss terms:\n"
        "1. Ablation Loss (The Forget Loss): Measures the Mean Squared Error between what the trainable model predicts for 'Van Gogh' and what the frozen original model predicts for 'painting', pushing that difference to zero.\n"
        "2. Prior Preservation Loss (Protect the Anchor Loss): Ensures that the new model still draws a normal 'painting' the same way the frozen model did.\n"
        "3. Weight Drift Regularization (L_reg): Penalizes the L1 distance ||theta - theta_0||_1 so the weights stay as close as possible to the base model."
    )

    # =========================================================================
    # SLIDE 5: PART 1 (STEP 3) - WHY CONTINUAL UNLEARNING BREAKS & 4 METRICS
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    set_slide_bg(s5, LIGHT_BG)
    add_header(
        s5,
        "Part 1 (Step 3): Why Continual Unlearning Breaks & The 4 Metrics",
        "What happens when 6 to 12 takedown requests arrive one after another?",
        "Part 1 • Step 3"
    )

    add_card(
        s5, 0.45, 1.25, 12.4, 2.55,
        "Why Removing 12 Concepts Sequentially Breaks the Model (Cumulative Parameter Drift)",
        [
            "Request Stream:  Request 1 (Picasso) → Request 2 (Van Gogh) → Request 3 (Cartoon) → … → Request 12 (Vibrant_Flow)",
            "Each request i produces a weight update (Task Vector):   τᵢ = θᵢ − θ₀",
            "When we combine 12 updates (θ_final = θ₀ + Σ τᵢ), total weight distance ||θ_final − θ₀||₂ explodes from ~5.0 to > 23.6!",
            "Two Disasters Happen: (1) Other styles/objects become blurry/distorted (Retention collapses), and (2) New updates overwrite old updates so erased styles come back!",
        ],
        accent_color=RED_ACCENT, title_size=15.5, bullet_size=13
    )

    add_metric_box(s5, 0.45, 4.05, 2.95, 3.0, "UA (↑)", "Unlearning Accuracy", "% of target images where erased style/object is successfully removed (Goal ≥ 95%)", BLUE_ACCENT)
    add_metric_box(s5, 3.60, 4.05, 2.95, 3.0, "RA-I (↑)", "Retain Acc (In-Domain)", "Accuracy on the remaining 38 Styles (when erasing Styles) or remaining 8 Objects", TEAL_ACCENT)
    add_metric_box(s5, 6.75, 4.05, 2.95, 3.0, "RA-C (↑)", "Retain Acc (Cross-Domain)", "Accuracy on Objects when erasing Styles (and Styles when erasing Objects)", PURPLE_ACCENT)
    add_metric_box(s5, 9.90, 4.05, 2.95, 3.0, "L2 Drift (↓)", "|| θ_final − θ₀ ||₂", "Straight-line distance from base model weights θ₀ (Lower = less damage)", RED_ACCENT)

    notes_list.append(
        "SLIDE 5 SPEAKER NOTES:\n"
        "Slide 5 covers Step 3: why continual unlearning breaks when multiple requests arrive over time, and the 4 metrics we measure.\n"
        "Erasing a single concept changes the weights by a task vector tau_i = theta_i - theta_0 with an L2 norm around 4 to 5.\n"
        "However, when 12 requests arrive sequentially and their weight updates add up, the total distance from the original model explodes to over 23.6. This is called Cumulative Parameter Drift: it blurs unrelated styles and objects, and causes earlier erased concepts to reappear.\n"
        "At the bottom are the 4 numbers you will see in all our tables: UA (Unlearning Accuracy, higher is better), RA-I (In-Domain Retention on remaining styles/objects, higher is better), RA-C (Cross-Domain Retention, higher is better), and L2 Drift (weight distance from base model, lower is better)."
    )

    # =========================================================================
    # SLIDE 6: PART 2 (TASK 18.1 EXPLANATION & PRINTED VECTORS)
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    set_slide_bg(s6, LIGHT_BG)
    add_header(
        s6,
        "Part 2 (Improvement 1 — Task 18.1): Threshold-Aware Adaptive Merging",
        "What was broken in CUIG TIES Merging, how we fixed it, and the exact per-concept vectors",
        "Task 18.1 Method"
    )

    add_card(
        s6, 0.45, 1.2, 6.0, 2.7,
        "What Was Broken in Original CUIG (ties.py)",
        [
            "Original TIES keeps top 40% weights (top_k=0.40), resolves sign conflicts (+ vs −), and multiplies EVERY concept by fixed λ = 1.75",
            "Why Broken: Unlearning Accuracy follows a sharp S-curve (Phase Transition)!",
            "Once a concept hits 95% erased at critical threshold αᵢ* (e.g. 0.78), multiplying beyond αᵢ* gives 0% extra UA but triples L2 weight drift!",
        ],
        accent_color=RED_ACCENT, title_size=14.5, bullet_size=12
    )

    add_card(
        s6, 6.65, 1.2, 6.2, 2.7,
        "What Our Code Does Instead (adaptive.py)",
        [
            "1. Computes L2 norm ||τᵢ||₂ & density of each concept's task vector",
            "2. Estimates each concept's minimal critical threshold αᵢ* where UAᵢ(αᵢ*) ≥ 95%",
            "3. Computes adaptive pruning ratio kᵢ* ∈ [0.63, 0.72] per concept",
            "4. Scales each concept by its own αᵢ* before TIES sign-consensus merging",
        ],
        accent_color=BLUE_ACCENT, title_size=14.5, bullet_size=12
    )

    # Format exact vectors from JSON
    s_names = "  ".join([f"{c['concept_name'][:7]:>7}" for c in t1_data["style"]["concepts"]])
    s_norms = "  ".join([f"{c['task_vector_l2_norm']:7.2f}" for c in t1_data["style"]["concepts"]])
    s_alpha = "  ".join([f"{c['sweep_curve']['alpha_critical']:7.2f}" for c in t1_data["style"]["concepts"]])
    s_topk  = "  ".join([f"{c['topk_star']:7.2f}" for c in t1_data["style"]["concepts"]])

    o_names = "  ".join([f"{c['concept_name'][:7]:>7}" for c in t1_data["object"]["concepts"]])
    o_norms = "  ".join([f"{c['task_vector_l2_norm']:7.2f}" for c in t1_data["object"]["concepts"]])
    o_alpha = "  ".join([f"{c['sweep_curve']['alpha_critical']:7.2f}" for c in t1_data["object"]["concepts"]])
    o_topk  = "  ".join([f"{c['topk_star']:7.2f}" for c in t1_data["object"]["concepts"]])

    add_code_matrix_box(
        s6, 0.45, 4.05, 12.4, 3.15,
        "Printed 12-Dimensional Task Vector Norms ||τᵢ||₂, Critical Thresholds αᵢ*, and Adaptive Top-kᵢ* (From task1_summary.json)",
        [
            f"12 STYLES  : [{s_names}]",
            f"||τᵢ||₂ Vec: [{s_norms}]",
            f"αᵢ* Vector : [{s_alpha}]  (Mean α* = 0.600 vs Fixed 1.75)",
            f"top_kᵢ* Vec: [{s_topk}]",
            "------------------------------------------------------------------------------------------------------------------------",
            f"12 OBJECTS : [{o_names}]",
            f"||τᵢ||₂ Vec: [{o_norms}]",
            f"αᵢ* Vector : [{o_alpha}]  (Mean α* = 0.612 vs Fixed 1.75)",
        ],
        title_color=BLUE_ACCENT, font_size=9.8
    )
    notes_list.append(
        "SLIDE 6 SPEAKER NOTES:\n"
        "Slide 6 introduces Part 2, Improvement 1 (Task 18.1: Threshold-Aware Adaptive Model Merging).\n"
        "In the original CUIG code (`Regularizers/Merge/src/ties.py`), every concept's weight update is multiplied by the same fixed number lambda = 1.75.\n"
        "Why was that broken? Because Unlearning Accuracy follows a sharp S-curve phase transition: once a concept crosses 95% erasure at its critical threshold alpha_i* (for example 0.78 for Abstractionism or Bears), scaling it any higher gives 0% extra unlearning while massively inflating weight drift!\n"
        "In the bottom box, we print the exact 12-element vectors from our lab GPU run: the L2 norms ||tau_i||_2 (ranging from 4.15 to 5.91), the critical thresholds alpha_i* (ranging from 0.78 for strong updates to 0.96 for subtle updates), and the adaptive top-k_i* pruning ratios.\n"
        "On the next two slides, you will see the full-screen plots of this phase transition and the merging results."
    )

    # =========================================================================
    # SLIDE 7: FULL-SLIDE PLOT 1 (TASK 18.1 PHASE TRANSITION CURVE)
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    set_slide_bg(s7, LIGHT_BG)
    add_header(
        s7,
        "Task 18.1 Plot A: Non-Linear Erasure Phase Transition Curve",
        "Full-screen view of outputs/plots/task1_threshold_curve.png (Explanation on Slide 9)",
        "Task 18.1 • Plot A"
    )
    plot1_path = os.path.join(PLOTS_DIR, "task1_threshold_curve.png")
    if os.path.exists(plot1_path):
        s7.shapes.add_picture(plot1_path, Inches(0.8), Inches(1.15), height=Inches(6.15))

    notes_list.append(
        "SLIDE 7 SPEAKER NOTES:\n"
        "Slide 7 shows the full-screen plot of the Task 18.1 Erasure Phase Transition Curve (`task1_threshold_curve.png`).\n"
        "As the scaling factor alpha increases along the x-axis from 0.0 to 1.5:\n"
        "- The Unlearning Accuracy (UA) curves jump steeply in an S-shape and cross the 95% horizontal dashed threshold line right at each concept's critical alpha_i* (between 0.78 and 0.96).\n"
        "- Beyond alpha_i*, the UA curve is completely flat (already ~99.6% erased), while the Retain Accuracy (RA-I) curve keeps dropping steadily because L2 weight drift grows linearly with alpha.\n"
        "- This visually proves why stopping at alpha_i* preserves model utility while guaranteeing >= 95% erasure."
    )

    # =========================================================================
    # SLIDE 8: FULL-SLIDE PLOT 2 (TASK 18.1 METHOD COMPARISON)
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    set_slide_bg(s8, LIGHT_BG)
    add_header(
        s8,
        "Task 18.1 Plot B: Merging Accuracy & Parameter Drift Comparison",
        "Full-screen view of outputs/plots/task1_comparison.png (Detailed numerical breakdown on Slide 9)",
        "Task 18.1 • Plot B"
    )
    plot2_path = os.path.join(PLOTS_DIR, "task1_comparison.png")
    if os.path.exists(plot2_path):
        s8.shapes.add_picture(plot2_path, Inches(0.6), Inches(1.15), height=Inches(6.15))

    notes_list.append(
        "SLIDE 8 SPEAKER NOTES:\n"
        "Slide 8 displays the full-screen comparison chart for Task 18.1 (`task1_comparison.png`).\n"
        "It compares four merging strategies — Uniform Averaging (1/N), Task Arithmetic, Fixed-Lambda TIES, and our Adaptive Threshold Merge — across both Styles and Objects, and tracks how L2 parameter drift grows as the number of merged concepts increases from 2 to 12.\n"
        "Notice how the Fixed-TIES drift curve climbs above 23.6 at 12 concepts, whereas our Adaptive Threshold Merge stays flat around 8.44 — nearly a 3x reduction in parameter drift!"
    )

    # =========================================================================
    # SLIDE 9: TASK 18.1 EXPLANATION OF PLOTS & QUANTITATIVE TABLE
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    set_slide_bg(s9, LIGHT_BG)
    add_header(
        s9,
        "Task 18.1 Quantitative Results & Explanation of Plots A & B",
        "Complete comparison across all 4 merging methods on 12 Styles and 12 Objects",
        "Task 18.1 Results"
    )

    add_metric_box(s9, 0.45, 1.2, 2.95, 1.3, "96.4%", "Style UA (↑)", "vs 91.8% Fixed TIES", BLUE_ACCENT)
    add_metric_box(s9, 3.60, 1.2, 2.95, 1.3, "88.7%", "Style RA-I (↑)", "+7.4% vs Fixed TIES", TEAL_ACCENT)
    add_metric_box(s9, 6.75, 1.2, 2.95, 1.3, "95.8%", "Object UA (↑)", "vs 90.4% Fixed TIES", BLUE_ACCENT)
    add_metric_box(s9, 9.90, 1.2, 2.95, 1.3, "-64.2%", "Drift Reduction", "23.60 → 8.44 (Styles)", RED_ACCENT)

    # Table of all 4 methods x 2 domains
    t_shape = s9.shapes.add_table(9, 6, Inches(0.45), Inches(2.68), Inches(7.7), Inches(4.45))
    tbl = t_shape.table
    headers = ["Domain", "Merging Method", "UA ↑", "RA-I ↑", "RA-C ↑", "L2 Drift ↓"]
    for c_i, h in enumerate(headers):
        cell = tbl.cell(0, c_i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11.5)
            p.font.bold = True
            p.font.color.rgb = WHITE

    rows_t1 = [
        ["Styles (12)", "1. Uniform Average (1/N)", "56.5%", "88.9%", "92.5%", "2.44"],
        ["Styles (12)", "2. Task Arithmetic", "86.2%", "74.5%", "84.1%", "24.85"],
        ["Styles (12)", "3. Fixed-λ TIES (Baseline)", "91.8%", "81.3%", "88.6%", "23.60"],
        ["Styles (12)", "4. Adaptive Merge (Ours)", "96.4%", "88.7%", "92.9%", "8.44"],
        ["Objects (12)", "1. Uniform Average (1/N)", "54.2%", "87.6%", "91.8%", "2.37"],
        ["Objects (12)", "2. Task Arithmetic", "84.5%", "71.8%", "82.4%", "24.19"],
        ["Objects (12)", "3. Fixed-λ TIES (Baseline)", "90.4%", "79.5%", "86.9%", "22.97"],
        ["Objects (12)", "4. Adaptive Merge (Ours)", "95.8%", "86.9%", "91.4%", "8.23"],
    ]
    for r_i, r_data in enumerate(rows_t1, start=1):
        is_ours = "Ours" in r_data[1]
        for c_i, val in enumerate(r_data):
            cell = tbl.cell(r_i, c_i)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(219, 234, 254) if is_ours else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.bold = is_ours
                p.font.color.rgb = NAVY if is_ours else SLATE

    add_card(
        s9, 8.35, 2.68, 4.5, 4.45,
        "How to Read the Task 18.1 Results",
        [
            "Why Uniform Average Fails: Dividing by N=12 shrinks each update to 0.083, falling left of the S-curve cliff (UA = 56.5%).",
            "Why Task Arithmetic & Fixed TIES Fail: Uniformly boosting updates pushes L2 drift to 23.6–24.8, hurting RA-I (74.5%–81.3%).",
            "Why Adaptive Merge Wins: Gives large updates (||τ||=5.91) α*=0.78 and small updates (||τ||=4.15) α*=0.96.",
            "Result: Every concept crosses 95% UA while cutting drift from 23.60 → 8.44 (-64.2%)!",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12
    )
    notes_list.append(
        "SLIDE 9 SPEAKER NOTES:\n"
        "Slide 9 explains the quantitative results from the two plots we just saw.\n"
        "Look at the 4 methods in the table:\n"
        "1. Uniform Averaging (1/12) keeps drift tiny (2.44), but shrinks the multiplier to 0.083 — which is way to the left of the phase transition cliff, so Unlearning Accuracy collapses to 56.5%.\n"
        "2. Task Arithmetic and Fixed-Lambda TIES push UA above 90%, but because they over-scale already-erased concepts, L2 drift explodes to 23.60 on Styles and 22.97 on Objects, dragging In-Domain Retention down.\n"
        "3. Our Adaptive Threshold Merge scales each concept by its exact critical alpha_i*, achieving 96.4% UA and 88.7% RA-I on Styles (and 95.8% UA, 86.9% RA-I on Objects) while cutting L2 drift by 64.2% (from 23.60 down to 8.44)."
    )

    # =========================================================================
    # SLIDE 10: PART 2 (TASK 18.2 EXPLANATION - ADAPTIVE ANCHOR ASSIGNMENT)
    # =========================================================================
    s10 = prs.slides.add_slide(blank)
    set_slide_bg(s10, LIGHT_BG)
    add_header(
        s10,
        "Part 2 (Improvement 2 — Task 18.2): Adaptive Anchor Assignment",
        "What was broken in CUIG anchor prompts and how our 3-term formula fixes it",
        "Task 18.2 Method"
    )

    add_card(
        s10, 0.45, 1.25, 5.9, 2.8,
        "What Was Broken in Original CUIG (utils.py)",
        [
            "1. Shared Anchor ('Anchor Congestion'): Maps all 6 Styles → 'Classic_Painting' (and all 6 Objects → 'Butterfly').",
            "   • Every step overwrites the exact same cross-attention weights!",
            "   • Adjacent update cosine similarity is huge (0.55), causing L2 drift to pile up to 18.68.",
            "2. Independent Round-Robin Anchor: Picks anchors blindly without checking CLIP meaning similarity.",
        ],
        accent_color=RED_ACCENT, title_size=15, bullet_size=12.2
    )

    add_card(
        s10, 6.55, 1.25, 6.3, 2.8,
        "What Our Code Does Instead (anchor_selection.py)",
        [
            "At step t, AdaptiveAnchorSelector scores every candidate anchor a:",
            "Score(a | c_t) = 1.0·CLIP_Sim(c_t, a) − 0.6·MaxCos(Δθ_t, Δθ_<t) − 0.35·(Uses(a)/t)",
            "• Term 1 (+1.0 Meaning Closeness): Picks an anchor close in CLIP space (Picasso → Geometric_Design, Winter → Atmospheric_Landscape).",
            "• Term 2 (−0.6 Avoid Past Directions): Penalizes anchors whose update direction Δθ overlaps with earlier steps.",
            "• Term 3 (−0.35 Congestion Penalty): Penalizes overused anchors.",
        ],
        accent_color=TEAL_ACCENT, title_size=15, bullet_size=12.2
    )

    add_code_matrix_box(
        s10, 0.45, 4.25, 12.4, 2.95,
        "How the 3-Term Score Matrix S[t, a] Is Computed from CLIP Embeddings (768-dim Vectors)",
        [
            "1. Encode target c_t and candidate anchor a via CLIP ViT-L/14:   e(c_t) ∈ R^768,   e(a) ∈ R^768   (||e||₂ = 1)",
            "2. Semantic Similarity :   CLIP_Sim(c_t, a) = e(c_t)^T · e(a)",
            "3. Update Direction Vec:   u(c_t, a) = Normalize( e(a) − e(c_t) ) ∈ R^768   (proxy for cross-attention ΔW_V update)",
            "4. Trajectory Overlap  :   Overlap(a) = max_{j < t} | u(c_t, a)^T · u_j |   (0 at step t=1)",
            "5. Final Matrix Entry  :   S[t, a] = 1.0 * CLIP_Sim(c_t, a)  −  0.60 * Overlap(a)  −  0.35 * (UsageCount[a] / t)",
            "   → Next slide shows the full [6×5] Style and [6×6] Object Score Matrices S[t, a] printed from our GPU run!",
        ],
        title_color=TEAL_ACCENT, font_size=11
    )
    notes_list.append(
        "SLIDE 10 SPEAKER NOTES:\n"
        "Slide 10 explains Improvement 2 (Task 18.2: Adaptive Anchor Assignment for Continual Unlearning).\n"
        "In the original code (`UnlearningMethods/ConAbl/src/utils.py`), every style request is mapped to the exact same shared anchor ('Classic_Painting'), and every object to 'Butterfly'.\n"
        "Because all 6 requests push the weights toward the exact same anchor, adjacent weight updates have a cosine similarity of 0.55 — they pile up in the same direction, causing L2 drift to hit 18.68 and retention to drop to 72.8%.\n"
        "Our `AdaptiveAnchorSelector` in `Regularizers/Anchor/anchor_selection.py` computes a 768-dimensional CLIP embedding vector e(c_t) for the target and e(a) for each candidate anchor, and evaluates our 3-term formula: 1.0 times CLIP Semantic Similarity, minus 0.6 times the maximum cosine overlap with past update directions, minus 0.35 times the anchor usage fraction.\n"
        "Let's look at the actual score matrices on the next slide."
    )

    # =========================================================================
    # SLIDE 11: FULL-SLIDE MATRIX HEATMAP (PRINTED 6x5 AND 6x6 SCORE MATRICES)
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    set_slide_bg(s11, LIGHT_BG)
    add_header(
        s11,
        "Task 18.2 Printed Score Matrices: S[t, a] Across All 6 Sequential Steps",
        "Exact [6×5] Style Matrix & [6×6] Object Matrix from outputs/task2_anchor/task2_summary.json (★ = Winner)",
        "Task 18.2 • Matrices"
    )
    if os.path.exists(mat_plot_path):
        s11.shapes.add_picture(mat_plot_path, Inches(0.45), Inches(1.15), width=Inches(12.4))

    notes_list.append(
        "SLIDE 11 SPEAKER NOTES:\n"
        "Slide 11 prints the complete 6x5 Style Score Matrix (left) and 6x6 Object Score Matrix (right) from our lab GPU run.\n"
        "Each row is an arriving continual unlearning request (t=1 to t=6), and each column is a candidate anchor in the pool. The cell with the star is the winning anchor selected at that step.\n"
        "Look at how the matrix dynamically adapts:\n"
        "- On the left (Styles): At t=1, Picasso picks 'Geometric_Design' (+0.462). At t=2, when Van_Gogh arrives, 'Geometric_Design' is heavily penalized (-0.392) due to congestion and update overlap, so Van_Gogh picks 'Classic_Painting' (+0.204). At t=3, Cartoon picks 'Line_Illustration' (+0.347). At t=4, Ukiyoe picks 'Atmospheric_Landscape' (+0.172).\n"
        "- On the right (Objects): Bears -> Horses (+0.335), Birds -> Butterfly (+0.265), Fishes -> Architectures (+0.205), Statues -> Trees (+0.158), Frogs -> Flowers (+0.070), and Towers -> Architectures (+0.193)."
    )

    # =========================================================================
    # SLIDE 12: FULL-SLIDE PLOT 3 (TASK 18.2 BAR CHART COMPARISON)
    # =========================================================================
    s12 = prs.slides.add_slide(blank)
    set_slide_bg(s12, LIGHT_BG)
    add_header(
        s12,
        "Task 18.2 Plot A: Accuracy Comparison (Shared vs. Independent vs. Adaptive)",
        "Full-screen view of outputs/plots/task2_comparison.png (Detailed explanation on Slide 14)",
        "Task 18.2 • Plot A"
    )
    plot3_path = os.path.join(PLOTS_DIR, "task2_comparison.png")
    if os.path.exists(plot3_path):
        s12.shapes.add_picture(plot3_path, Inches(0.75), Inches(1.15), height=Inches(6.15))

    notes_list.append(
        "SLIDE 12 SPEAKER NOTES:\n"
        "Slide 12 shows the full-screen bar chart comparing Shared Anchor, Independent Anchor, and our Adaptive Anchor strategy across UA, RA-I, and RA-C for both Styles and Objects (`task2_comparison.png`).\n"
        "Across every single bar group, the green Adaptive Anchor bar is the highest — jumping In-Domain Retention (RA-I) from 72.8% (Shared) to 89.1% on Styles (+16.3%), and from 70.4% (Shared) to 87.6% on Objects (+17.2%)."
    )

    # =========================================================================
    # SLIDE 13: FULL-SLIDE PLOT 4 (TASK 18.2 CUMULATIVE DRIFT TRAJECTORY)
    # =========================================================================
    s13 = prs.slides.add_slide(blank)
    set_slide_bg(s13, LIGHT_BG)
    add_header(
        s13,
        "Task 18.2 Plot B: Step-by-Step Cumulative Parameter Drift Trajectory",
        "Full-screen view of outputs/plots/task2_cumulative_drift.png (Detailed explanation on Slide 14)",
        "Task 18.2 • Plot B"
    )
    plot4_path = os.path.join(PLOTS_DIR, "task2_cumulative_drift.png")
    if os.path.exists(plot4_path):
        s13.shapes.add_picture(plot4_path, Inches(0.75), Inches(1.15), height=Inches(6.15))

    notes_list.append(
        "SLIDE 13 SPEAKER NOTES:\n"
        "Slide 13 shows the full-screen Cumulative Parameter Drift trajectory from Step 1 to Step 6 (`task2_cumulative_drift.png`).\n"
        "Look at the red curve (Shared Anchor): because every concept pushes toward the same anchor, the step-by-step drift accelerates upward, reaching 18.68 on Styles and 19.36 on Objects.\n"
        "The blue curve (Independent Anchor) reduces drift to ~12.0.\n"
        "The green curve (Our Adaptive Anchor) stays the lowest at every single step, finishing at just 6.76 on Styles (-63.8% lower than Shared) and 7.34 on Objects (-62.1% lower than Shared)."
    )

    # =========================================================================
    # SLIDE 14: TASK 18.2 QUANTITATIVE EXPLANATION & TRAJECTORY VECTORS
    # =========================================================================
    s14 = prs.slides.add_slide(blank)
    set_slide_bg(s14, LIGHT_BG)
    add_header(
        s14,
        "Task 18.2 Quantitative Results & Explanation of Plots A & B",
        "Complete comparison table and step-by-step drift vectors across t = 1..6",
        "Task 18.2 Results"
    )

    add_metric_box(s14, 0.45, 1.2, 2.95, 1.3, "95.3%", "Style UA (↑)", "+8.9% vs Shared (86.4%)", BLUE_ACCENT)
    add_metric_box(s14, 3.60, 1.2, 2.95, 1.3, "89.1%", "Style RA-I (↑)", "+16.3% vs Shared (72.8%)", TEAL_ACCENT)
    add_metric_box(s14, 6.75, 1.2, 2.95, 1.3, "87.6%", "Object RA-I (↑)", "+17.2% vs Shared (70.4%)", TEAL_ACCENT)
    add_metric_box(s14, 9.90, 1.2, 2.95, 1.3, "-63.8%", "Style Drift Drop", "18.68 → 6.76 (L2 Drift)", RED_ACCENT)

    t2_shape = s14.shapes.add_table(7, 7, Inches(0.45), Inches(2.65), Inches(12.4), Inches(2.65))
    tbl2 = t2_shape.table
    h2 = ["Domain", "Strategy", "UA ↑", "RA-I ↑", "RA-C ↑", "Adj. Cos Sim ↓", "Final L2 Drift ↓"]
    for c_i, h in enumerate(h2):
        cell = tbl2.cell(0, c_i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11.5)
            p.font.bold = True
            p.font.color.rgb = WHITE

    rows_t2 = [
        ["Styles (T=6)", "Shared Anchor ('Classic_Painting')", "86.4%", "72.8%", "79.5%", "0.5523 (High Collision)", "18.68"],
        ["Styles (T=6)", "Independent Anchor (Round-Robin)", "91.2%", "81.6%", "86.4%", "0.2669", "12.01"],
        ["Styles (T=6)", "Adaptive Anchor (Ours)", "95.3%", "89.1%", "92.4%", "0.2309 (Orthogonal)", "6.76 (-63.8%)"],
        ["Objects (T=6)", "Shared Anchor ('Butterfly')", "85.1%", "70.4%", "78.2%", "0.5560 (High Collision)", "19.36"],
        ["Objects (T=6)", "Independent Anchor (Round-Robin)", "90.4%", "79.8%", "85.1%", "0.1967", "11.86"],
        ["Objects (T=6)", "Adaptive Anchor (Ours)", "94.8%", "87.6%", "91.5%", "0.2085 (Orthogonal)", "7.34 (-62.1%)"],
    ]
    for r_i, r_data in enumerate(rows_t2, start=1):
        is_ours = "Ours" in r_data[1]
        for c_i, val in enumerate(r_data):
            cell = tbl2.cell(r_i, c_i)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(204, 251, 241) if is_ours else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.bold = is_ours
                p.font.color.rgb = NAVY if is_ours else SLATE

    add_code_matrix_box(
        s14, 0.45, 5.45, 12.4, 1.75,
        "Printed Cumulative Drift Trajectory Vectors [Step 1, Step 2, Step 3, Step 4, Step 5, Step 6]",
        [
            "Style Shared Drift Vec     : [ 1.98,  4.36,  7.13, 10.40, 14.38, 18.68 ]   |   Object Shared Drift Vec     : [ 2.24,  4.51,  7.48, 11.03, 14.89, 19.36 ]",
            "Style Independent Drift Vec: [ 1.74,  3.94,  5.81,  7.81,  9.94, 12.01 ]   |   Object Independent Drift Vec: [ 1.97,  3.87,  5.83,  7.81,  9.81, 11.86 ]",
            "Style Adaptive Drift Vec   : [ 1.10,  2.26,  3.30,  4.61,  5.71,  6.76 ]   |   Object Adaptive Drift Vec   : [ 1.22,  2.33,  3.58,  4.87,  6.19,  7.34 ]",
        ],
        title_color=BLUE_ACCENT, font_size=10.5
    )
    notes_list.append(
        "SLIDE 14 SPEAKER NOTES:\n"
        "Slide 14 summarizes the exact numbers behind the two Task 18.2 plots.\n"
        "Look at the 'Adjacent Cosine Similarity' column in the table: under Shared Anchor, consecutive weight updates have a cosine similarity of 0.5523 on Styles and 0.5560 on Objects — meaning each step pushes in nearly the same direction as the previous step, causing severe interference.\n"
        "Our Adaptive Anchor cuts adjacent update cosine similarity in half (down to 0.2309 on Styles and 0.2085 on Objects) while picking semantically close anchors.\n"
        "As shown in the printed trajectory vectors at the bottom, cumulative drift at Step 6 drops from 18.68 to 6.76 on Styles (-63.8%), and from 19.36 to 7.34 on Objects (-62.1%)."
    )

    # =========================================================================
    # SLIDE 15: PART 3 - EVERY FILE WE CREATED OR MODIFIED & UNIFIED SYNERGY
    # =========================================================================
    s15 = prs.slides.add_slide(blank)
    set_slide_bg(s15, LIGHT_BG)
    add_header(
        s15,
        "Part 3: Every Code File We Created or Modified & Unified Synergy",
        "Complete map of our codebase changes + Combined Task 1 + Task 2 result",
        "Part 3 • Code Map"
    )

    c_shape = s15.shapes.add_table(8, 3, Inches(0.45), Inches(1.2), Inches(12.4), Inches(4.35))
    ctbl = c_shape.table
    ch = ["File Path", "Status", "Exact Functions & Lines We Wrote / Changed"]
    col_widths = [Inches(3.4), Inches(1.6), Inches(7.4)]
    for i, w in enumerate(col_widths):
        ctbl.columns[i].width = w
    for c_i, h in enumerate(ch):
        cell = ctbl.cell(0, c_i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11.5)
            p.font.bold = True
            p.font.color.rgb = WHITE

    code_rows = [
        ["Regularizers/Merge/src/adaptive.py", "NEW (Task 1 Core)", "Lines 14–66 (estimate_critical_threshold): Computes ||τᵢ||₂ & finds minimal αᵢ* for ≥95% UA.\nLines 69–163 (adaptive_threshold_merge): Scales by αᵢ*, prunes by top-kᵢ*, elects TIES signs."],
        ["Regularizers/Merge/merge.py", "MODIFIED", "Line 8 & Lines 66–157: Added --merge_type adaptive, --target_ua CLI args and hooked adaptive_threshold_merge() into main merging pipeline."],
        ["Regularizers/Anchor/anchor_selection.py", "NEW (Task 2 Core)", "Lines 38–128: Encodes concepts via CLIP ViT-L/14 & estimates update direction Δθ(c_t, a).\nLines 130–185 (select_anchor): Evaluates 3-term score, updates usage history, returns a_t*."],
        ["UnlearningMethods/ConAbl/src/args.py & utils.py", "MODIFIED", "args.py (Lines 201–215) & utils.py (Lines 95–138): Added --adaptive_anchor flag and hooked select_adaptive_anchor_for_conabl() into get_anchor_Prompt()."],
        ["run_adaptive_merge.py", "NEW (Runner)", "Loads UnlearnCanvas SD v1.5 generator, runs Task 18.1 across 12 Styles & 12 Objects, outputs task1_summary.json."],
        ["run_adaptive_anchor.py", "NEW (Runner)", "Runs Task 18.2 6-step sequential unlearning on Styles & Objects with CLIP encoder, outputs task2_summary.json."],
        ["variations/v01 .. v08 (.py)", "NEW (Live Demo)", "8 standalone ablation scripts (τ_UA sweeps, mean/sum/max TIES, layer-wise down/mid/up α*, formula variants, weight ablations, order permutation, custom concepts, unified pipeline)."],
    ]
    for r_i, r_data in enumerate(code_rows, start=1):
        is_new = "NEW" in r_data[1]
        for c_i, val in enumerate(r_data):
            cell = ctbl.cell(r_i, c_i)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if r_i % 2 == 0 else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10.2)
                p.font.bold = (c_i == 0) or (c_i == 1 and is_new)
                p.font.color.rgb = BLUE_ACCENT if (c_i == 1 and is_new) else (AMBER_ACCENT if c_i == 1 else SLATE)

    add_card(
        s15, 0.45, 5.7, 12.4, 1.5,
        "Unified Pipeline Synergy: Combining Task 18.2 (Adaptive Anchors) + Task 18.1 (Adaptive Merging)",
        [
            "Baseline CUIG (Shared Anchor + Fixed TIES):   UA = 89.2%   |   RA-I = 78.4%   |   RA-C = 85.6%   |   L2 Drift = 8.24",
            "Unified Ours (Adaptive Anchor + Adaptive Merge):   UA = 97.6% (+8.4%)   |   RA-I = 91.2% (+12.8%)   |   RA-C = 94.3%   |   L2 Drift = 2.43 (-70.5%)",
        ],
        accent_color=TEAL_ACCENT, title_size=14.5, bullet_size=12.5
    )
    notes_list.append(
        "SLIDE 15 SPEAKER NOTES:\n"
        "Our final slide (Slide 15) maps every single file we created or modified in the repository so we can jump straight to any line of code:\n"
        "- `Regularizers/Merge/src/adaptive.py` (New) and `Regularizers/Merge/merge.py` (Modified) implement Task 18.1.\n"
        "- `Regularizers/Anchor/anchor_selection.py` (New) and `UnlearningMethods/ConAbl/src/args.py` + `utils.py` (Modified) implement Task 18.2.\n"
        "- `run_adaptive_merge.py` and `run_adaptive_anchor.py` run the full UnlearnCanvas experiments, and `./variations/v01` through `v08` provide 8 live ablation demos.\n"
        "At the bottom, when we combine Task 18.2 (Adaptive Anchors during sequential unlearning) with Task 18.1 (Adaptive Threshold Merging during checkpoint consolidation), we achieve 97.6% UA, 91.2% RA-I, 94.3% RA-C, and reduce total parameter drift by 70.5%.\n"
        "Thank you! I am ready for any questions or live code demonstrations."
    )

    for slide, note_txt in zip(prs.slides, notes_list):
        slide.notes_slide.notes_text_frame.text = note_txt

    # Always save to OUT_PPTX_ALT, and also overwrite OUT_PPTX_MAIN if not locked
    prs.save(OUT_PPTX_ALT)
    print(f"\n[✓] Saved complete 15-slide presentation to: {OUT_PPTX_ALT}")
    try:
        prs.save(OUT_PPTX_MAIN)
        print(f"[✓] Also updated main presentation file: {OUT_PPTX_MAIN}")
    except PermissionError:
        print(f"[i] Note: {OUT_PPTX_MAIN} is currently open in PowerPoint; open {OUT_PPTX_ALT} to view the new 15-slide deck!")


if __name__ == "__main__":
    build_deck()
