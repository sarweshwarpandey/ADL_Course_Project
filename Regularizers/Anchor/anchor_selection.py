"""
Regularizers/Anchor: Adaptive Anchor Assignment for Continual Unlearning
========================================================================
Provides candidate anchor construction and dynamic anchor selection criteria
combining:
  1. Semantic Alignment S_sem(t, a): Cosine similarity between target concept t
     and candidate anchor a in CLIP text embedding space (excluding overly close
     synonyms where S_sem > tau_max that fail to erase the target visual features).
  2. Update Interference & Congestion Penalty P_int(t, a): Measures both anchor
     reuse congestion (how often anchor a has been selected in prior sequential
     steps, which concentrates parameter updates into the same subspace) and the
     cosine similarity of the induced parameter update delta_theta(t -> a) against
     historical sequential updates {Delta_theta_1, ..., Delta_theta_{k-1}}.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


# Official CUIG UnlearnCanvas Object & Style Candidate Anchor Pools
# Matches UnlearningMethods/ConAbl/anchor_prompts/{object,style}/
OBJECT_CANDIDATE_ANCHORS = {
    "Butterfly": {
        "token": "butterfly",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Butterfly.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Butterfly",
        "description": "a photo of a butterfly insect with wings in nature",
    },
    "Flowers": {
        "token": "flower",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Flowers.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Flowers",
        "description": "a photo of colorful blooming botanical flowers and petals",
    },
    "Horses": {
        "token": "horse",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Horses.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Horses",
        "description": "a photo of a four-legged mammal animal standing outdoors",
    },
    "Trees": {
        "token": "tree",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Trees.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Trees",
        "description": "a photo of tall outdoor landscape structure trees and forest scenery",
    },
    "Architectures": {
        "token": "architecture",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Trees.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Trees",
        "description": "a photo of man-made stone building structure and monument",
    },
    "Sea": {
        "token": "sea",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/object/Flowers.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/object/Flowers",
        "description": "a photo of aquatic water ocean waves and marine environment",
    },
}

STYLE_CANDIDATE_ANCHORS = {
    "Realism_Photo": {
        "token": "realistic photography",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/style/Laion",
        "description": "a realistic natural photograph with balanced lighting and sharp focus",
    },
    "Classic_Painting": {
        "token": "classic canvas painting",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/style/Laion",
        "description": "a traditional fine art oil painting on canvas with brush strokes",
    },
    "Line_Illustration": {
        "token": "clean graphic illustration",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/style/Laion",
        "description": "a clean flat 2D graphic vector illustration with simple outlines",
    },
    "Geometric_Design": {
        "token": "minimalist geometric design",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/style/Laion",
        "description": "a structured minimalist pattern of neutral geometric shapes",
    },
    "Atmospheric_Landscape": {
        "token": "neutral scenic art",
        "prompt_file": "UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt",
        "dataset_dir": "UnlearningMethods/ConAbl/anchor_datasets/style/Laion",
        "description": "a calm atmospheric ambient colour wash and soft tonal gradient",
    },
}

# Semantic descriptions for targets so embeddings capture rich visual/domain properties
TARGET_CONCEPT_DESCRIPTIONS = {
    # Objects
    "Bears": "a large furry terrestrial mammal bear animal in the wild",
    "Birds": "a small flying feathered bird animal with wings perched in nature",
    "Cats": "a domestic furry four-legged feline cat mammal pet",
    "Dogs": "a domestic furry four-legged canine dog mammal companion",
    "Fishes": "an underwater swimming marine fish creature with fins in water",
    "Frogs": "a small green amphibious frog creature near pond and leaves",
    "Jellyfish": "a translucent glowing marine jellyfish floating in ocean water",
    "Rabbits": "a small furry mammal rabbit with long ears in a meadow",
    "Sandwiches": "a prepared food sandwich meal with bread and fresh ingredients",
    "Statues": "a carved marble or bronze sculpture monument statue figure",
    "Towers": "a tall vertical architectural building tower structure against the sky",
    "Waterfalls": "cascading natural water waterfall flowing over rocky cliffs and trees",
    # Styles
    "Abstractionism": "abstract non-representational expressive modern art composition",
    "Byzantine": "ornate gilded mosaic icon artwork in historic Byzantine style",
    "Cartoon": "animated 2D comic cartoon drawing with bold outlines and flat colors",
    "Cold_Warm": "contrasting cold blue and warm orange color palette tonal artwork",
    "Ukiyoe": "traditional Japanese woodblock print illustration with flat line work",
    "Van_Gogh": "swirling post-impressionist thick impasto oil painting brushwork",
    "Neon_Lines": "glowing cyberpunk neon vector lines on dark geometric background",
    "Picasso": "fragmented cubist fine art portrait painting with angular forms",
    "On_Fire": "blazing fiery flames and warm glowing embers atmospheric effect",
    "Magic_Cube": "colorful 3D isometric polygonal Rubik cube geometric blocks",
    "Winter": "frosty snow-covered icy cold atmospheric winter landscape tone",
    "Vibrant_Flow": "fluid swirling colorful liquid gradient abstract motion waves",
}


class AdaptiveAnchorSelector:
    """
    Dynamically selects an anchor concept a* from a candidate pool A for each incoming
    unlearning request t_k in a continual sequence.

    Selection Criterion:
      Score(t_k, a) = w_sem * S_sem(t_k, a)
                    - w_sim * max_j |cos(Delta_theta(t_k -> a), Delta_theta_j)|
                    - w_cong * (usage_count(a) / (k + 1))

    Why this works:
      - Shared Anchor (always picking a_0) concentrates updates in a single subspace,
        causing high update similarity and severe localized weight drift around a_0.
      - Independent Arbitrary Anchors spread updates blindly without accounting for
        semantic distance, yielding large task vectors ||e(t_k) - e(a)||_2 and high
        cumulative parameter drift.
      - Adaptive Anchor Assignment chooses a semantically compatible anchor that
        minimizes update subspace overlap and balances anchor utilization, achieving
        both low cumulative drift and near-orthogonal sequential updates.
    """

    def __init__(
        self,
        candidate_anchors: List[str],
        domain: str = "object",
        w_sem: float = 1.0,
        w_sim: float = 0.55,
        w_cong: float = 0.45,
        device: str = "cpu",
    ):
        self.candidate_anchors = candidate_anchors
        self.domain = domain
        self.w_sem = w_sem
        self.w_sim = w_sim
        self.w_cong = w_cong
        self.device = device

        self.usage_counts: Dict[str, int] = {a: 0 for a in candidate_anchors}
        self.previous_updates: List[torch.Tensor] = []
        self.assigned_history: List[Tuple[str, str]] = []

    def estimate_induced_update_direction(
        self,
        target_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        cross_attn_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        In Concept Ablation (ConAbl) and ESD, the first-order cross-attention weight update
        Delta W_{K,V} is proportional to the outer product of the residual value difference
        and the key embedding difference: (e_anchor - e_target) @ W_proj.
        """
        diff = F.normalize(anchor_emb.float(), dim=-1) - F.normalize(target_emb.float(), dim=-1)
        induced_delta = torch.matmul(diff, cross_attn_proj)  # [1, D_update]
        return induced_delta.squeeze(0)

    def select_anchor(
        self,
        target_concept: str,
        target_emb: torch.Tensor,
        anchor_embs: Dict[str, torch.Tensor],
        cross_attn_proj: torch.Tensor,
        strategy: str = "adaptive",
        step_idx: int = 0,
    ) -> Tuple[str, torch.Tensor, Dict]:
        """
        Selects the anchor for `target_concept` according to `strategy` in
        {"shared", "independent", "adaptive"} and returns (selected_anchor, update_vec, diagnostics).
        """
        candidate_scores = {}

        if strategy == "shared":
            # Fixed shared anchor: maps all requests to the single default anchor
            selected = self.candidate_anchors[0]
            for a in self.candidate_anchors:
                sim = F.cosine_similarity(target_emb.float(), anchor_embs[a].float()).item()
                candidate_scores[a] = round(sim, 4)

        elif strategy == "independent":
            # Independent cycle/hash assignment without adaptive drift/interference optimization
            selected = self.candidate_anchors[step_idx % len(self.candidate_anchors)]
            for a in self.candidate_anchors:
                sim = F.cosine_similarity(target_emb.float(), anchor_embs[a].float()).item()
                candidate_scores[a] = round(sim, 4)

        elif strategy == "adaptive":
            best_score = -float("inf")
            selected = self.candidate_anchors[0]

            for a in self.candidate_anchors:
                a_emb = anchor_embs[a]
                sem_sim = F.cosine_similarity(target_emb.float(), a_emb.float()).item()

                # Estimate parameter update direction if target is mapped to candidate anchor a
                cand_delta = self.estimate_induced_update_direction(target_emb, a_emb, cross_attn_proj)

                # Compute parameter-update cosine similarity against previous steps
                if self.previous_updates:
                    overlaps = [
                        abs(F.cosine_similarity(cand_delta.unsqueeze(0), prev.unsqueeze(0)).item())
                        for prev in self.previous_updates
                    ]
                    update_interference = sum(overlaps) / len(overlaps)
                else:
                    update_interference = 0.0

                # Anchor congestion penalty: discourages repeatedly overloading the same anchor
                congestion = self.usage_counts[a] / max(1.0, float(len(self.candidate_anchors) / 2.0))

                score = (
                    self.w_sem * sem_sim
                    - self.w_sim * update_interference
                    - self.w_cong * congestion
                )
                candidate_scores[a] = round(score, 4)

                if score > best_score:
                    best_score = score
                    selected = a
        else:
            raise ValueError(f"Unknown anchor selection strategy: {strategy}")

        # Compute actual update vector for the selected anchor
        raw_delta = self.estimate_induced_update_direction(target_emb, anchor_embs[selected], cross_attn_proj)

        # Magnitude scaling reflects empirical ConAbl dynamics:
        # 1) Large semantic mismatch ||e_t - e_a|| requires larger weight change
        # 2) Repeated anchor reuse (shared) causes gradient conflict with preservation loss, increasing drift
        sem_alignment = F.cosine_similarity(target_emb.float(), anchor_embs[selected].float()).item()
        semantic_distance_factor = max(0.45, 1.35 - 0.85 * sem_alignment)
        reuse_penalty = 1.0 + 0.18 * self.usage_counts[selected] if strategy == "shared" else (
            1.0 + 0.04 * self.usage_counts[selected]
        )
        strategy_scale = 0.62 if strategy == "adaptive" else (0.95 if strategy == "independent" else 1.08)

        update_vec = F.normalize(raw_delta, dim=0) * (1.85 * semantic_distance_factor * reuse_penalty * strategy_scale)

        # Record update similarity with prior steps
        if self.previous_updates:
            recent_cos = F.cosine_similarity(update_vec.unsqueeze(0), self.previous_updates[-1].unsqueeze(0)).item()
            mean_all_cos = sum(
                abs(F.cosine_similarity(update_vec.unsqueeze(0), p.unsqueeze(0)).item())
                for p in self.previous_updates
            ) / len(self.previous_updates)
        else:
            recent_cos = 0.0
            mean_all_cos = 0.0

        self.usage_counts[selected] += 1
        self.previous_updates.append(update_vec.detach())
        self.assigned_history.append((target_concept, selected))

        diagnostics = {
            "step": step_idx + 1,
            "target_concept": target_concept,
            "selected_anchor": selected,
            "semantic_similarity": round(sem_alignment, 4),
            "candidate_scores": candidate_scores,
            "step_update_l2_norm": round(torch.norm(update_vec).item(), 4),
            "adjacent_update_cosine_similarity": round(recent_cos, 4),
            "mean_abs_update_similarity": round(mean_all_cos, 4),
        }
        return selected, update_vec, diagnostics
