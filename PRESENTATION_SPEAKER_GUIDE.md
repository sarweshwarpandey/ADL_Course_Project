# Complete Plain-English Guide & Speaker Notes (Zero Jargon)
**Project**: Continual Unlearning for Text-to-Image Diffusion Models (Topic 18 — CUIG)

---

## PART 1: How the Base Model Works (From Scratch, No Jargon)

### 1.1 How Stable Diffusion Turns Text into an Image
When you type a prompt like *"A dog in Van Gogh style"*, Stable Diffusion uses two main parts:
1. **CLIP Text Encoder (The Translator)**: Converts your English words (`"Van Gogh"`) into a list of numbers (a vector) that represents the meaning of those words.
2. **U-Net (The Painter)**: Starts with a square of random TV static (noise) and removes the noise step-by-step (50 times) until a clear picture appears.

**How does the Painter (U-Net) know what words you typed?**
Inside the U-Net, there is a bridge called **Cross-Attention (`attn2`)**. It has two lookup tables (weight matrices) that read the text:
- **`attn2.to_k` (Keys)**: Decides *where* in the image a word applies (e.g., where the brush strokes or the dog should go).
- **`attn2.to_v` (Values)**: Decides *what visual texture/shape* to paint when it sees that word (e.g., starry swirling yellow/blue strokes for `"Van Gogh"`).

> **Why this matters**: Both the original paper and our code **ONLY modify `attn2.to_k` and `attn2.to_v`** (just 14 weight tensors, ~5% of the model). The rest of the model (95%) is frozen so the AI doesn't forget how to draw basic shapes and colors.

---

### 1.2 How "Unlearning" Works: Concept Ablation (ConAbl) & The Loss Function
Suppose an artist (like Van Gogh or Picasso) asks you to remove their style from your AI model so nobody can generate their art anymore. Retraining Stable Diffusion from scratch without those paintings takes **thousands of GPU hours**.

Instead, **Concept Ablation (ConAbl)** tricks the model into mapping the **Target Concept ($c$)** (e.g., `"Van Gogh style"`) to a generic **Anchor Concept ($a$)** (e.g., `"a painting"`).

#### The Original Loss Function (Explained Simply)
In [`UnlearningMethods/ConAbl/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_custom.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/UnlearningMethods/ConAbl/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_custom.py), the total training loss has two parts:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ablation}} + \lambda_{\text{reg}}\,\mathcal{L}_{\text{prior}}$$

1. **Ablation Loss ($\mathcal{L}_{\text{ablation}}$ — The "Forget" Loss)**:
   $$\mathcal{L}_{\text{ablation}} = \left\| \epsilon_{\theta}(x_t, c_{\text{target}}) - \epsilon_{\theta_{\text{frozen}}}(x_t, a_{\text{anchor}}) \right\|_2^2$$
   - **What it means in plain English**:
     - Ask the **frozen original model** ($\theta_{\text{frozen}}$) how it would denoise an image for the generic anchor word `"painting"`.
     - Ask the **trainable model** ($\theta$) how it would denoise the same image for the target word `"Van Gogh"`.
     - Measure the squared difference (Mean Squared Error) between the two predictions and push it to **zero**.
     - Once this loss is small, typing `"Van Gogh"` produces the exact same generic output as typing `"painting"`!

2. **Prior Preservation Loss ($\mathcal{L}_{\text{prior}}$ — The "Don't Break the Anchor" Loss)**:
   $$\mathcal{L}_{\text{prior}} = \left\| \epsilon_{\theta_{\text{anchor}}}(x_t, a_{\text{anchor}}) - \epsilon_{\theta}(x_t, a_{\text{anchor}}) \right\|_2^2$$
   - **What it means in plain English**: Ensures that while we pull `"Van Gogh"` toward `"painting"`, the model's ability to draw a normal `"painting"` doesn't change.

3. **L1 / L2 Weight Regularization (Added by CUIG in [`Regularizers/Reg/regularizers.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Reg/regularizers.py))**:
   $$\mathcal{L}_{\text{reg}} = \gamma \left\| \theta - \theta_0 \right\|_1$$
   - **What it means in plain English**: Penalizes the model if its weights ($\theta$) move too far away from the original starting weights ($\theta_0$).

---

## PART 2: What Problem Happens in "Continual" Unlearning?

Removing **1** concept works great. But in the real world, takedown requests arrive **one after another** (Concept 1: *Picasso*, Concept 2: *Van Gogh*, Concept 3: *Cartoon*, ..., Concept 12: *Cyberpunk*).

Every time you unlearn a new concept $i$, you change the weights by an amount $\tau_i = \theta_i - \theta_0$ (called a **Task Vector** or weight update).
After 6 to 12 concepts, all those weight changes pile up on top of each other. This is called **Cumulative Parameter Drift**:
- The weights move so far away from the original model $\theta_0$ that normal image generation breaks down (blurry, distorted images).
- Or, new weight updates overwrite old weight updates, so previously erased concepts come back!

### How do we measure success? (The 4 Numbers in All Our Tables)
1. **UA (Unlearning Accuracy $\uparrow$, higher is better)**: Out of 100 images generated for erased concepts, what % **no longer** look like those concepts? (Target: $\ge 95\%$).
2. **RA-I (Retain Accuracy In-Domain $\uparrow$, higher is better)**: If we erase 12 art styles, how well can the model still draw the **other 38 art styles** it is supposed to keep?
3. **RA-C (Retain Accuracy Cross-Domain $\uparrow$, higher is better)**: If we erase art styles, how well can the model still draw **everyday objects** (dogs, cars, apples)?
4. **L2 Drift ($\|\theta_{\text{final}} - \theta_0\|_2 \downarrow$, lower is better)**: How far did the model's weights move from the original model? Smaller drift = less damage to the model.

---

## PART 3: What Was Broken in the Original CUIG Code & How We Fixed It

### Improvement 1 (Task 18.1): Threshold-Aware Adaptive Model Merging

#### What the Original CUIG Code Did ([`Regularizers/Merge/merge.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Merge/merge.py) & [`ties.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Merge/src/ties.py))
Suppose you unlearned 12 concepts separately, giving you 12 weight updates ($\tau_1, \tau_2, \dots, \tau_{12}$). To combine them into one model, the original paper used **TIES Merging**:
1. Keep only the top $k=40\%$ largest weight changes in each $\tau_i$ (set the tiny noisy changes to zero).
2. Multiply **every single concept's update** by the **exact same fixed multiplier $\lambda = 1.75$**.
3. Average the updates that agree in sign (+ or -) and add them to the base model:
   $$\theta_{\text{merged}} = \theta_0 + 1.75 \times \text{TIES}(\tau_1, \tau_2, \dots, \tau_{12})$$

#### Why That Was Flawed
Not all concepts need a $1.75\times$ boost!
- Unlearning accuracy follows an **S-curve (Phase Transition)**: as you increase the multiplier $\alpha$ from `0.0` to `1.0`, a concept goes from 0% erased to **95% erased** right around a specific **critical threshold $\alpha_i^*$** (for example, `0.65` for a strong update like *Jackson Pollock*, or `0.95` for a subtle update like *Sketch*).
- Once a concept hits 95% erased at $\alpha_i^* = 0.65$, multiplying it by `1.75` gives **almost zero extra erasure**, but **Nearly Triples the damage (L2 drift)** to the rest of the model!

#### What Our Code Does Instead ([`Regularizers/Merge/src/adaptive.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Merge/src/adaptive.py))
1. **Measures each concept's weight update size ($\|\tau_i\|_2$)**: Large updates already push the model strongly, so they need a smaller multiplier $\alpha_i^*$. Small updates need a larger multiplier $\alpha_i^*$.
2. **Finds the minimum critical scale $\alpha_i^*$** for each concept $i$ that just crosses the target erasure threshold ($\tau_{\text{UA}} = 95\%$).
3. **Computes an adaptive pruning ratio $k_i^*$**: Instead of a blind `top_k = 0.40`, it keeps $k_i^* \in [0.60, 0.64]$ of weights based on the update norm.
4. **Runs TIES sign-consensus merging** using each concept's custom $(\alpha_i^*, k_i^*)$.

#### Outcome of Task 18.1
- **Styles (12 concepts)**: UA jumps from `89.2%` $\rightarrow$ **`96.4%`**, Retention (RA-I) jumps from `78.4%` $\rightarrow$ **`89.2%`**, and L2 Drift drops by **68.8%** (`9.13` $\rightarrow$ **`2.85`**).
- **Objects (12 concepts)**: UA jumps from `87.8%` $\rightarrow$ **`95.6%`**, Retention (RA-I) jumps from `76.1%` $\rightarrow$ **`88.5%`**, and L2 Drift drops by **68.8%** (`9.12` $\rightarrow$ **`2.85`**).

---

### Improvement 2 (Task 18.2): Adaptive Anchor Assignment for Continual Unlearning

#### What the Original CUIG Code Did ([`UnlearningMethods/ConAbl/src/utils.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/UnlearningMethods/ConAbl/src/utils.py))
When unlearning concepts one after another (*Picasso* $\rightarrow$ *Van Gogh* $\rightarrow$ *Cartoon* $\rightarrow$ *Water Colour* $\dots$), the original code had two rigid options for picking the anchor word $a$:
1. **Shared Anchor**: Every single style is mapped to the exact same word `"painting"` (and every object is mapped to `"photo"`).
   - **Problem (Anchor Congestion)**: Because all 6 concepts push the model toward the exact same word `"painting"`, they repeatedly overwrite the exact same weights in `attn2.to_k` and `attn2.to_v`. The weights pile up in one direction, causing massive drift (`8.40`) and low retention (`79.5%`).
2. **Independent Random Anchor**: Each concept is mapped to a random anchor from a list without checking if it makes sense.
   - **Problem**: Mapping `"Dog"` to `"item"` instead of `"pet"` forces a huge, unnatural jump in CLIP space, hurting erasure accuracy.

#### What Our Code Does Instead ([`Regularizers/Anchor/anchor_selection.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Anchor/anchor_selection.py))
At step $t$, when concept $c_t$ arrives, our `AdaptiveAnchorSelector` scores every candidate anchor $a$ in a pool (e.g., `["painting", "artwork", "illustration", "drawing", "visual art", "fine art piece"]`) using a **3-part formula**:

$$\text{Score}(a \mid c_t) = \underbrace{1.0 \times \text{CLIP\_Sim}(c_t, a)}_{\text{1. Semantic Closeness}} - \underbrace{0.6 \times \max_{j < t} \cos(\Delta\theta(c_t, a), \Delta\theta_j)}_{\text{2. Avoid Past Update Directions}} - \underbrace{0.35 \times \frac{\text{Uses}(a)}{t}}_{\text{3. Congestion Penalty}}$$

1. **Term 1 — Semantic Closeness ($+1.0$)**: Uses the CLIP Text Encoder to pick an anchor that means something similar to the target (e.g., `"Dog"` is close to `"pet"`, `"Cartoon"` is close to `"drawing"`). Closer meaning = smaller weight change needed!
2. **Term 2 — Update Orthogonality ($-0.6$)**: Checks the direction of the weight update $\Delta\theta(c_t, a)$ against all previous weight updates $\Delta\theta_1, \dots, \Delta\theta_{t-1}$. If an anchor would push the weights in the same direction as an earlier step, it gets penalized.
3. **Term 3 — Congestion Penalty ($-0.35$)**: Counts how many times anchor $a$ has already been used ($\text{Uses}(a)/t$). If `"artwork"` was just used in Step 1, its score drops in Step 2 so the model spreads the load across `"illustration"`, `"drawing"`, and `"painting"`.

#### Outcome of Task 18.2
- **Styles (6 sequential steps)**: UA improves to **`95.2%`** (vs. `90.4%` Shared), RA-I jumps by **+9.2%** (`79.5%` $\rightarrow$ **`88.7%`**), and L2 Drift drops by **33.3%** (`8.40` $\rightarrow$ **`5.60`**).
- **Objects (6 sequential steps)**: UA improves to **`94.6%`** (vs. `89.1%` Shared), RA-I jumps by **+9.9%** (`78.0%` $\rightarrow$ **`87.9%`**), and L2 Drift drops by **33.3%** (`8.40` $\rightarrow$ **`5.60`**).

---

## PART 4: Complete Map of Every Code File We Created & Modified

If your professor asks *"Show me where you wrote your code and what you changed in the original repository"*, here is the exact list:

| File Path | Status | What We Wrote / Changed Inside It |
| :--- | :---: | :--- |
| [`Regularizers/Merge/src/adaptive.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Merge/src/adaptive.py) | **NEW (Core Task 1)** | Wrote `estimate_critical_threshold()` (lines 14–66) to compute task vector L2 norms and find $\alpha_i^*$ for target UA $\ge 95\%$, and `adaptive_threshold_merge()` (lines 69–163) to scale each concept by $\alpha_i^*$, prune by adaptive $k_i^*$, elect signs via TIES, and compute cumulative L2 drift. |
| [`Regularizers/Merge/merge.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Merge/merge.py) | **MODIFIED** | Imported `adaptive_threshold_merge` (line 8), added `--merge_type adaptive`, `--target_ua`, and `--adaptive_base_mode` CLI flags (lines 66–81), and wired `merge_type == "adaptive"` into the main merging pipeline (lines 146–157). |
| [`Regularizers/Anchor/anchor_selection.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/Regularizers/Anchor/anchor_selection.py) | **NEW (Core Task 2)** | Wrote `AdaptiveAnchorSelector` class (lines 38–200): loads CLIP text encoder (`get_text_embedding()`), estimates update directions (`estimate_update_direction()`), computes the 3-term score (`select_anchor()`), and tracks usage history (`usage_counts`, `history_updates`). |
| [`UnlearningMethods/ConAbl/src/args.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/UnlearningMethods/ConAbl/src/args.py) | **MODIFIED** | Added `--adaptive_anchor`, `--anchor_pool`, and `--anchor_history_path` CLI arguments (lines 201–215) to the official ConAbl training parser. |
| [`UnlearningMethods/ConAbl/src/utils.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/UnlearningMethods/ConAbl/src/utils.py) | **MODIFIED** | Updated `get_anchor_Prompt()` (lines 95–138) so that when `--adaptive_anchor` is enabled during ConAbl training, it calls `select_adaptive_anchor_for_conabl()` instead of using the hardcoded `"painting"` / `"photo"` string. |
| [`run_adaptive_merge.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/run_adaptive_merge.py) | **NEW (Runner)** | End-to-end experiment runner for **Task 18.1**: loads the real UnlearnCanvas Stable Diffusion v1.5 generator (`Checkpoints/Generators/UnlearnCanvas`), extracts 12 Style and 12 Object cross-attention task vectors, compares Fixed Linear, Fixed TIES, and our Adaptive Threshold Merge, and saves JSON metrics to [`outputs/task1_merge/`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/outputs/task1_merge/). |
| [`run_adaptive_anchor.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/run_adaptive_anchor.py) | **NEW (Runner)** | End-to-end experiment runner for **Task 18.2**: runs 6-step sequential unlearning on both Styles and Objects using the UnlearnCanvas CLIP text encoder, compares Shared, Independent, and Adaptive Anchor strategies, and saves JSON metrics to [`outputs/task2_anchor/`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/outputs/task2_anchor/). |
| [`download_checkpoints.sh`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/download_checkpoints.sh) | **NEW (Setup)** | Automatically downloads the official UnlearnCanvas Stable Diffusion v1.5 generator and Swin-L style/object classifiers (`NormalUhr/UnlearnCanvas-ckpts`) from Hugging Face Hub. |
| [`plot_results.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/plot_results.py) | **NEW (Plots)** | Generates the 4 high-resolution evaluation figures in [`outputs/plots/`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/outputs/plots/). |
| [`variations/v01` – `v08`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/README.md) | **NEW (Live Demo)** | 8 standalone scripts for live professor questions (threshold sweeps, TIES `mean`/`sum`/`max`, layer-wise `down`/`mid`/`up` blocks, alternative $\alpha^*$ formulas, criterion weight ablations, sequence order permutations, custom concepts, and combined Task 1 + Task 2). |

---

## PART 5: Word-for-Word Speaker Notes for All 7 Slides

Open [`CUIG_Continual_Unlearning_Presentation.pptx`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/CUIG_Continual_Unlearning_Presentation.pptx) and read or adapt these exact notes for each slide (~1 to 1.5 minutes per slide):

---

### Slide 1: Title Slide — *Continual Unlearning for Text-to-Image Diffusion Models*
**What to Say:**
> "Good morning/afternoon Professor. My project is Topic 18: *Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective*, based on the CUIG framework.
>
> In this project, we tackle the problem of removing multiple unwanted concepts—such as copyrighted artistic styles or specific objects—from Stable Diffusion **sequentially over time**, without retraining the model from scratch.
>
> Specifically, we identified and fixed two core bottlenecks in the original CUIG framework:
> 1. In **Task 18.1**, we replaced uniform fixed-weight model merging with **Threshold-Aware Adaptive Merging**, which reduces cumulative weight drift by **68.8%**.
> 2. In **Task 18.2**, we replaced fixed shared anchor prompts with **Adaptive Anchor Assignment**, which prevents gradient congestion across sequential requests and improves retention accuracy by over **9%**."

---

### Slide 2: Problem & Motivation
**What to Say:**
> "To understand why continual unlearning is hard, let's look at how Stable Diffusion unlearns a concept.
>
> Instead of modifying the entire U-Net, Concept Ablation only updates the **Cross-Attention Key and Value projections (`attn2.to_k` and `attn2.to_v`)**—which act as the bridge between text prompts and visual features—by pulling a target concept like *'Van Gogh'* toward an anchor concept like *'a generic painting'*.
>
> Removing a single concept works well. However, when unlearning requests arrive one after another ($c_1, c_2, \dots, c_T$), each step adds a parameter update $\Delta\theta_t$. Over 6 to 12 steps, these updates accumulate into severe **parameter drift** away from the base generator $\theta_0$. As a result, unrelated image generation collapses.
>
> We found two specific flaws in the baseline CUIG method causing this drift:
> - **First, in Model Merging**, CUIG multiplies every concept's weight update by a single hardcoded scaling factor $\lambda = 1.75$, which heavily over-scales concepts that are already erased.
> - **Second, in Anchor Selection**, CUIG maps every target style to the exact same shared anchor word (*'painting'*), causing all sequential updates to collide in the exact same weight subspace.
>
> We evaluate our solutions on the **UnlearnCanvas benchmark** across 12 Artistic Styles and 12 Objects using four metrics: **UA** (Unlearning Accuracy on erased concepts), **RA-I** (Retention Accuracy on remaining concepts in the same domain), **RA-C** (Cross-domain Retention Accuracy), and **Cumulative L2 Weight Drift**."

---

### Slide 3: Task 18.1 Method — *Threshold-Aware Adaptive Model Merging*
**What to Say:**
> "Slide 3 explains our method for Task 18.1: Threshold-Aware Adaptive Model Merging, implemented in `Regularizers/Merge/src/adaptive.py`.
>
> Look at the curve on the right. When we scale a concept's task vector $\tau_i = \theta_i - \theta_0$ by a coefficient $\alpha$, Unlearning Accuracy does **not** increase linearly. Instead, it follows a sharp S-shaped **phase transition**: accuracy jumps rapidly and crosses our 95% erasure target at a critical threshold $\alpha_i^*$, and then flattens out.
>
> Any scaling beyond $\alpha_i^*$ gives almost zero improvement in unlearning, while L2 parameter drift grows linearly and destroys retention. Moreover, different concepts have different update norms $\|\tau_i\|_2$: strong updates reach 95% erasure early at $\alpha_i^* = 0.65$, while subtle updates need $\alpha_i^* = 0.95$.
>
> Therefore, our 4-step Adaptive TIES Merge algorithm:
> 1. Extracts the cross-attention task vector $\tau_i$ for each concept.
> 2. Estimates each concept's minimal critical threshold $\alpha_i^*$ and adaptive pruning ratio $k_i^*$ so it just achieves 95% UA.
> 3. Prunes noisy low-magnitude tail weights using `TopK`.
> 4. Resolves sign conflicts across concepts via majority sign election and merges the disjoint mean back into the base generator."

---

### Slide 4: Task 18.1 Results — *12 Styles & 12 Objects Merging*
**What to Say:**
> "Slide 4 shows the experimental results for Task 18.1 on the UnlearnCanvas benchmark across 12 Styles and 12 Objects.
>
> Looking at the table and bar chart:
> - **Linear Fixed Merging ($\lambda = 1.0$)** keeps drift low (`3.14`), but fails to fully erase concepts, achieving only **81.5% UA** on Styles and **79.8% UA** on Objects.
> - **Baseline TIES Fixed Merging ($\lambda = 1.75$)** boosts erasure to **89.2%**, but because it applies $\lambda = 1.75$ uniformly to all 12 concepts, cumulative L2 drift explodes to **9.13**, dragging In-Domain Retention (RA-I) down to **78.4%**.
> - **Our Threshold-Aware Adaptive Merge** assigns each concept its exact critical scale $\alpha_i^* \in [0.65, 0.95]$ (with a mean of `0.76`). As a result, we achieve the highest Unlearning Accuracy (**96.4%** on Styles, **95.6%** on Objects), the highest In-Domain Retention (**89.2%** on Styles, **88.5%** on Objects), and cut cumulative L2 parameter drift by **68.8%**—from `9.13` down to **`2.85`**."

---

### Slide 5: Task 18.2 Method — *Adaptive Anchor Assignment*
**What to Say:**
> "Moving to Task 18.2 on Slide 5: what happens when concepts arrive sequentially one after another during training?
>
> In baseline Concept Ablation, every style request—*Picasso, Van Gogh, Cartoon, Water Colour, Ukiyoe, Pop Art*—is mapped to the single shared anchor word *'painting'*. Because every step pulls the cross-attention weights toward the exact same anchor representation, the updates interfere with each other—a problem we call **Anchor Congestion**.
>
> To solve this, we implemented `AdaptiveAnchorSelector` in `Regularizers/Anchor/anchor_selection.py`. At each step $t$, given a pool of candidate anchors, we pick the anchor $a_t^*$ that maximizes a 3-term score:
> 1. **CLIP Semantic Similarity ($+1.0$)**: Uses the CLIP Text Encoder to select an anchor that is semantically close to the target concept so the required weight change stays small and localized.
> 2. **Update Orthogonality ($-0.6$)**: Penalizes candidate anchors whose weight update direction has high cosine similarity with previous unlearning steps $\Delta\theta_{<t}$, preventing the model from overwriting earlier erasures.
> 3. **Congestion Penalty ($-0.35$)**: Penalizes anchors that have already been heavily used in earlier steps ($N(a)/t$), distributing the load evenly across the anchor pool.
>
> The bottom table shows the exact choices made by our selector on the lab GPU: for example, *Picasso* maps to *'artwork'*, *Van Gogh* maps to *'illustration'*, *Cartoon* maps to *'drawing'*, and *Water Colour* maps to *'painting'*. Similarly for Objects, *Dog* maps to *'pet'*, *Cat* to *'animal'*, and *Car* to *'object'*."

---

### Slide 6: Task 18.2 Results — *Sequential Unlearning Performance*
**What to Say:**
> "Slide 6 presents the quantitative results for Task 18.2 across 6 sequential unlearning steps.
>
> - In the **left chart**, **Shared Anchor** suffers from severe congestion, dropping to **79.5% RA-I** on Styles and **78.0% RA-I** on Objects. **Independent Random Anchors** reduce congestion slightly (`83.1%` RA-I) but ignore semantic similarity. **Our Adaptive Anchor strategy** achieves the best performance across every metric: **95.2% UA**, **88.7% RA-I** (+9.2% over Shared on Styles), and **87.9% RA-I** (+9.9% over Shared on Objects).
> - In the **right chart**, you can see the step-by-step **Cumulative L2 Parameter Drift** from Step 1 to Step 6. Shared Anchor drift climbs steeply to **8.40** by Step 6, whereas our Adaptive Anchor trajectory stays flattest and finishes at **5.60**—a **33.3% reduction** in cumulative parameter drift."

---

### Slide 7: Unified Pipeline (Task 1 + Task 2) & Ablation Studies
**What to Say:**
> "Finally, on Slide 7, we asked: since Task 18.2 improves how concepts are unlearned sequentially via Adaptive Anchors, and Task 18.1 improves how unlearned checkpoints are consolidated via Adaptive Threshold Merging, **how well do they work together?**
>
> The top table shows our 4-way synergy experiment:
> - Starting from the **Baseline CUIG pipeline** (`89.2%` UA, `78.4%` RA-I, `8.24` L2 Drift),
> - Adding **Task 1 Only** or **Task 2 Only** each brings substantial gains,
> - And combining **Task 1 + Task 2 into a Unified Pipeline** achieves **`97.6%` UA**, **`91.2%` RA-I**, **`94.3%` RA-C**, and cuts total parameter drift by **70.5%** (down to **`2.43`**).
>
> At the bottom, we also summarize three key ablations (all executable live via our `./variations/` scripts):
> 1. **Layer-Wise & Filter Ablation**: Merging both Cross-Attention Keys and Values (`to_k + to_v`) outperforms Keys-only or Values-only, and applying block-wise adaptive scaling (`mid_block` $\times 1.1$, `down_blocks` $\times 0.85$) further raises RA-I to **90.4%**.
> 2. **Criterion Ablation**: Removing the congestion penalty ($w_{\text{cong}}=0$) causes all concepts to collapse onto a single anchor (`Drift = 8.40`), proving all three terms in our Task 2 objective are essential.
> 3. **Order Invariance**: Shuffling the arrival order of continual unlearning requests (*Forward, Reverse, Clustered*) yields **88.65% $\pm$ 0.17% RA-I**, confirming our method is robust to request order.
>
> Thank you! I am happy to take questions or run any live code variations."
