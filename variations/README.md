# Live Demo & Viva Variations Guide (`variations/`)

This folder contains **8 self-contained, fast-executing Python scripts** covering the code modifications, hyperparameter sweeps, and ablation studies your professor may ask you to run or explain live during your presentation.

---

## Quick Mapping: Professor Question $\rightarrow$ File to Run

| # | What the Professor Might Ask You to Change / Demonstrate | Script to Run | Command |
| :--- | :--- | :--- | :--- |
| **1** | *"What happens if you make the erasure threshold $\tau_{\text{UA}}$ stricter (e.g., 95% or 98%) or looser (80%)?"* | [`v01_task1_change_ua_threshold.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v01_task1_change_ua_threshold.py) | `python variations/v01_task1_change_ua_threshold.py --custom_tau 0.92` |
| **2** | *"In TIES merging, what if you use `sum` or `max` instead of `mean`, or change the top-$k$ pruning ratio?"* | [`v02_task1_ties_merge_func_and_topk.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v02_task1_ties_merge_func_and_topk.py) | `python variations/v02_task1_ties_merge_func_and_topk.py` |
| **3** | *"Why merge both `to_k` and `to_v`? What if you merge only Keys (`to_k`), only Values (`to_v`), or assign different $\alpha^*$ per U-Net block (`down`, `mid`, `up`)?"* | [`v03_task1_layerwise_and_keyfilter_ablation.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v03_task1_layerwise_and_keyfilter_ablation.py) | `python variations/v03_task1_layerwise_and_keyfilter_ablation.py` |
| **4** | *"Can you change the formula used to compute $\alpha_i^*$ (e.g., use L1 sparsity, Cosine similarity, or a Sigmoid/Logistic phase transition curve)?"* | [`v04_task1_threshold_formula_variants.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v04_task1_threshold_formula_variants.py) | `python variations/v04_task1_threshold_formula_variants.py` |
| **5** | *"In Task 2, how do you know all 3 terms in your anchor selection equation matter? Remove the congestion penalty or change the weights!"* | [`v05_task2_anchor_criterion_ablation.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v05_task2_anchor_criterion_ablation.py) | `python variations/v05_task2_anchor_criterion_ablation.py --w_sem 1.0 --w_sim 0.0 --w_cong 0.5` |
| **6** | *"Since this is continual unlearning, what happens if the concepts arrive in the reverse order or a different permutation?"* | [`v06_task2_sequence_order_permutation.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v06_task2_sequence_order_permutation.py) | `python variations/v06_task2_sequence_order_permutation.py` |
| **7** | *"What if you only have 2 candidate anchors, or 6 candidate anchors? Or what if we unlearn a new set of concepts right now?"* | [`v07_task2_candidate_pool_and_custom_concepts.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v07_task2_candidate_pool_and_custom_concepts.py) | `python variations/v07_task2_candidate_pool_and_custom_concepts.py --custom_targets Monet Salvador_Dali Sketch` |
| **8** | *"How do Task 1 (Adaptive Merging) and Task 2 (Adaptive Anchor Assignment) work together in a single unified pipeline?"* | [`v08_combined_task1_and_task2_pipeline.py`](file:///c:/Users/sarwe/Desktop/ADL_Course_Project/variations/v08_combined_task1_and_task2_pipeline.py) | `python variations/v08_combined_task1_and_task2_pipeline.py` |

---

## Run All Variations at Once

```bash
python variations/run_all_variations.py --all
```
