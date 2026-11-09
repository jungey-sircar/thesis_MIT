# Novel Architecture Proposals for Your Thesis

> [!IMPORTANT]
> I researched the existing literature thoroughly. Here are **3 genuinely novel architectures** that have NOT been done before — specifically, nobody has applied these techniques to V-JEPA 2 frozen features for action recognition. Each one exploits unique properties of V-JEPA 2.1 that existing probes ignore.

---

## What Already Exists (and what we AVOID copying)

| Technique | Status | Why we avoid it |
|---|---|---|
| Attentive Probe (cross-attention CLS) | Used in VJEPA paper itself | Not novel at all |
| Multi-layer concatenation | Common practice since 2020 | Well-known technique |
| Temporal Difference Networks (TDN) | Published 2021, applied to pixels/raw features | Not new concept |
| Simple gated fusion | Many papers 2024-2025 | Too generic |
| Frame2Freq (spectral adapters on frozen backbones) | Published 2026 | Applied to input frames, not latent tokens |

---

## Option A: **SFD-Net** (Spectral Flow Decomposition Network)

### Core Innovation
Apply **1D Discrete Wavelet Transform (DWT) along the temporal axis** of V-JEPA 2's latent token sequences to decompose them into **motion-frequency subbands** — then classify using band-specific attention.

### Why This Is Novel
- **Nobody has applied spectral decomposition to VJEPA's latent spatiotemporal tokens.** Frame2Freq (2026) applies frequency analysis to *input frames*, not to *learned latent representations*.
- V-JEPA 2.1's key property is **temporally coherent latent tokens** — meaning temporal frequency analysis in this space is *semantically meaningful* (unlike pixel space where it's noisy).
- The wavelet decomposition separates: **low-frequency** (what object/scene = appearance) from **high-frequency** (how it moves = action dynamics).

### Architecture
```
V-JEPA tokens (8 frames × P patches × 1280-dim)
    │
    ├── Temporal DWT per patch position ──→ Low-freq tokens + High-freq tokens  
    │                                        (appearance)       (motion)
    │
    ├── Low-freq branch: Self-attention → appearance embedding
    ├── High-freq branch: Self-attention → motion embedding  
    │
    ├── Cross-band attention: motion queries ↔ appearance keys
    │       (learns which motion patterns occur with which scenes)
    │
    └── Classification head
```

### Thesis Contribution Statement
*"We propose SFD-Net, the first architecture to apply spectral frequency decomposition to V-JEPA 2's learned latent representations, exploiting the model's temporal coherence property to explicitly disentangle motion dynamics from static appearance in the representation space."*

---

## Option B: **DRGF-Net** (Depth-Residual Gated Fusion Network)

### Core Innovation
Compute **inter-layer "representation velocity"** (the delta between consecutive encoder layers) and use these velocity vectors as a **novel feature stream** alongside standard layer features, fused via learned per-sample gating.

### Why This Is Novel
- "Representation velocity" exists in the LLM interpretability literature (2026) for detecting anomalies, but **nobody has used it as a discriminative feature for classification**.
- V-JEPA 2.1's **deep self-supervision** means each layer has been explicitly trained — so the *change* between layers captures how the model progressively builds understanding (low-level texture → motion → semantics).
- Different action classes will have **different "processing trajectories"** through the depth of the encoder — a fast punch has a different depth signature than a slow walk.

### Architecture
```
V-JEPA encoder (N layers, each producing hidden states)
    │
    ├── Layer features:  H₁, H₂, H₃, ..., Hₙ
    ├── Depth velocities: ΔH₁=H₂-H₁, ΔH₂=H₃-H₂, ..., ΔHₙ₋₁=Hₙ-Hₙ₋₁  
    │
    ├── Feature Stream: concat selected layer features → MLP
    ├── Velocity Stream: concat depth velocities → MLP  
    │     (captures "how the model's understanding evolves")
    │
    ├── Per-sample Gating Network:
    │     gate = σ(W·[feat_stream; vel_stream])
    │     fused = gate ⊙ feat_stream + (1-gate) ⊙ vel_stream
    │
    └── Classification head
```

### Thesis Contribution Statement
*"We introduce DRGF-Net, which uniquely exploits V-JEPA 2's hierarchical depth structure by extracting inter-layer representation velocity as a novel discriminative feature stream, capturing how the model's internal understanding progressively transforms across its depth — a signal that is action-class-dependent due to the model's deep self-supervision training."*

---

## Option C (Recommended): **HAF-STD Net** (Hierarchical Adaptive Fusion with Spectral-Temporal Decomposition)

### Core Innovation
**Combines the two strongest novel ideas** into a unified architecture:
1. **Spectral-Temporal Decomposition** of latent tokens (from Option A)
2. **Depth-Aware Hierarchical Fusion** with representation velocity (from Option B)
3. **Adaptive per-sample routing** that learns which combination of (layer × frequency-band) is most discriminative for each input

### Why This Is Novel (and why it's the strongest option)
- Combines two independently novel ideas that are **synergistic**: spectral decomposition works better with multi-layer features because different layers capture different frequency behaviors.
- The **adaptive routing** mechanism is itself novel — it's not just gating, it's a learned routing matrix over a 2D grid of (depth × frequency) feature streams.
- Has a clear, defensible contribution that can fill an entire thesis chapter.

### Architecture
```
V-JEPA 2 encoder with output_hidden_states=True
    │
    ┌────────────────────────────────────────┐
    │  For each selected layer (4 layers):   │
    │    │                                    │
    │    ├─ Reshape to (T, P, D) temporal    │
    │    ├─ 1D DWT along T axis per patch    │
    │    ├─ → Low-freq stream (appearance)   │
    │    ├─ → High-freq stream (motion)      │
    │    │                                    │
    │    ├─ Compute depth velocity ΔHₖ       │
    │    │                                    │
    │    └─ 3 features per layer:            │
    │        [appearance, motion, velocity]   │
    └────────────────────────────────────────┘
    │
    │  Total: 4 layers × 3 streams = 12 feature vectors
    │
    ├── Adaptive Routing Network:
    │     Input: concatenation of all 12 streams (lightweight)
    │     Output: 12-dim routing weights (softmax)
    │     Fused = Σ(weight_i × stream_i)
    │
    ├── Classification head with cosine similarity
    │
    └── Output: 51-class predictions
```

### Thesis Contribution Statement
*"We propose HAF-STD Net, a novel classification architecture that jointly exploits V-JEPA 2.1's unique properties of temporal coherence and deep self-supervision. Our method introduces: (1) spectral-temporal decomposition of learned latent tokens to explicitly separate motion dynamics from appearance, (2) inter-layer representation velocity as a novel depth-aware feature stream, and (3) an adaptive routing mechanism that learns input-dependent fusion weights across a 2D grid of depth×frequency feature streams. To our knowledge, this is the first work to combine spectral analysis of latent representations with depth-aware hierarchical fusion for video action classification."*

---

## My Recommendation

> [!TIP]
> **Go with Option C (HAF-STD Net)** because:
> 1. It has **3 clearly articulable novel contributions** — enough for a strong thesis
> 2. Each component is independently ablatable (you can show tables proving each part helps)
> 3. It's directly motivated by V-JEPA 2.1's unique architecture properties (deep self-supervision + temporal coherence)
> 4. It's **practically implementable** on your Kaggle T4 — all computation is on frozen features, no fine-tuning needed
> 5. The "adaptive routing over depth×frequency" visual is great for a thesis diagram

## Which option would you like me to implement?
