---
title: "Generative Modeling via Drifting"
excerpt: "A non-diffusion, non-flow, non-adversarial 1-NFE generator reaches FID 1.54 on ImageNet 256x256 by moving the inference-time decomposition into training time via an anti-symmetric drifting field."
categories: [Paper, Generative-Models]
tags:
  - Drifting-Model
  - One-Step-Generation
  - Mean-Shift
  - ImageNet
  - Generative-Models
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-22
last_modified_at: 2026-05-22
permalink: /paper/drifting/
---

## TL;DR

- Instead of decomposing the prior-to-data pushforward into many **inference** steps (diffusion/flow), the paper moves the decomposition into **training**: a single-pass generator `f` is updated by a learned **drifting field** `V_{p,q}` whose equilibrium `V=0` coincides with `q = p_data`.
- The drifting field is **anti-symmetric**, kernel-based attraction-minus-repulsion: `V = V^+_p - V^-_q`, plugged into a SimSiam-style **stop-gradient fixed-point loss** so the network never has to differentiate through `q_theta`.
- Native **1-NFE** ImageNet 256x256: **FID 1.54** (latent L/2, 463M) and **FID 1.61** (pixel L/16, 464M) — best among single-step methods trained from scratch and competitive with 250x2-NFE multi-step diffusion. *No seed variance reported* — the 0.18-FID lead over iMeanFlow is within typical run-to-run noise on this benchmark.

## Motivation

Diffusion and flow models pay their quality bill at inference: 50-500 NFEs per sample. The whole consistency-model / rectified-flow / MeanFlow / distillation literature exists to compress that trajectory. The Drifting paper argues the SDE/ODE scaffolding is itself an inference-time artifact — **the iterative nature of SGD training is already a distribution-evolution process** that can carry the same burden. Under this lens, "generative modeling" is just: pick a pushforward parameterization, define an update rule that has `p_data` as its unique fixed point, and let the optimizer follow it.

That reframing is appealing as paradigm-claim, but it has practical teeth: 1-NFE inference matters for any latency-bound deployment — interactive editing, on-device synthesis, and (the paper's own §5.3 demo) robotic policies where Diffusion Policy's 100 NFE is the bottleneck. The same argument extends to clinical robotics or any real-time generative medical pipeline.

## Core Innovation

Three pieces fit together:

1. **Anti-symmetric drifting field with provable equilibrium.** Define `V_{p,q}: R^d -> R^d` so that an implicit drift `x_{i+1} = x_i + V_{p,q_i}(x_i)` reaches equilibrium when `q = p`. Proposition 3.1: if `V_{p,q}(x) = -V_{q,p}(x)` for all `x`, then `q = p => V = 0` (proof: `V_{p,p} = -V_{p,p} => V_{p,p} = 0`). Only the **forward** direction is proven; the converse `V ~ 0 => q ~ p` is treated as an identifiability heuristic in Appendix C.1.
2. **Stop-gradient fixed-point loss.** The training objective is
   $$L = \mathbb{E}_\epsilon \left\lVert f_\theta(\epsilon) - \mathrm{sg}\!\left(f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\right) \right\rVert^2.$$
   Numerically the loss value equals `E ||V(f(eps))||^2`, but stop-gradient blocks backprop through `V` (and through `q_theta`).
3. **Kernel mean-shift form of V.** `V_{p,q}(x) = (1/Z_p Z_q) E_{y+ ~ p, y- ~ q}[ k(x,y+) k(x,y-) (y+ - y-) ]` with a Gaussian-like kernel `k(x,y) = exp(-||x-y||/tau)` — implemented as softmax over `y` (and over `x` within batch). Computed in the feature space of a frozen encoder `phi` (SimCLR / MoCo-v2 / latent-MAE), summed across ResNet stages and spatial pools, with three temperatures `tau in {0.02, 0.05, 0.2}`.

CFG is folded into training only — `q_tilde(.|c) = (1-gamma) q_theta(.|c) + gamma p_data(.|empty)` — so inference stays **strictly 1-NFE** even with classifier-free guidance (standard CFG doubles NFE).

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | A new generative paradigm distinct from diffusion / flow / GAN / VAE / NF / MMD | Conceptual §2 + Appendix C.2 shows MMD is a special un-normalized case of `V` | ⭐⭐ |
| C2 | Anti-symmetric `V` => equilibrium at `q=p` | Prop. 3.1 one-line proof + Table 1 destructive ablation (breaking symmetry: 41-177 FID) | ⭐⭐⭐ |
| C3 | Zero drift => matched distributions (the **converse**) | Appendix C.1 "identifiability heuristic" under unverified non-degeneracy assumption + empirical `||V||^2` ~ FID correlation | ⭐ |
| C4 | Native 1-NFE inference, no CFG double-pass | Architecture §4 + Alg. 1; CFG is training-time only | ⭐⭐⭐ |
| C5 | SOTA among single-step methods, FID 1.54 (latent) | Table 5 vs iCT (34.24), Shortcut (10.60), MeanFlow (3.43), AdvFlow (2.38), iMeanFlow (1.72) | ⭐⭐⭐ |
| C6 | Competitive with multi-step diffusion | Table 5 — beats DiT/SiT; loses to SiT+REPA (1.42), LightningDiT (1.35), RAE+DiTDH (1.13) | ⭐⭐ |
| C7 | Pixel-space 1.61 FID is SOTA for one-step pixel | Table 6 — beats StyleGAN-XL (2.30), GigaGAN (3.45); matches PixelDiT (1.61) at 200x2 NFE | ⭐⭐⭐ |
| C8 | Robust to mode collapse | Fig. 3 (2D toy, three inits incl. collapsed) + Fig. 6 CLIP-NN qualitative | ⭐⭐ |
| C9 | Larger pos/neg sample budget improves quality | Table 2 monotone trend (20.43 → 8.46) under fixed `B = N_c x N_pos` | ⭐⭐⭐ |
| C10 | Feature encoder is necessary; encoder quality dominates | Table 3 + authors admit method does **not** work on ImageNet without a pretrained encoder | ⭐⭐⭐ |
| C11 | Generalizes beyond images — robotic control | Table 7 — 1-NFE Drifting Policy matches/exceeds 100-NFE Diffusion Policy on 4/6 tasks; **2 regressions undiscussed** | ⭐⭐ |
| C12 | ~18x inference-FLOP advantage over StyleGAN-XL (pixel) | Single-line claim §5.2: 87G vs 1574G FLOPs | ⭐⭐ |
| C13 | Generates novel images (not memorization) | Fig. 6 — 1-page CLIP-NN retrieval on a handful of classes | ⭐ |

## Method & Architecture

![Drifting Model overview](/assets/images/paper/drifting/fig_p001_01.png)
*Figure 1: Drifting Model overview — training-time evolution of the pushforward distribution `q_theta` toward `p_data` via a drifting field `V` that vanishes at equilibrium.*

The pipeline at a single training step:

1. Sample `N_c` class labels and a per-label CFG scale `alpha`.
2. Sample `N_neg` noise vectors `eps ~ p_eps` and push them through `f_theta` → generated `x` (the negatives, `~ q_theta`).
3. Sample `N_pos` real positives from each class queue + `N_unc` unconditionals from a global queue (MoCo-style queues: 128 per class, 1000 global, refreshed with 64 new samples / step).
4. Extract features through a frozen encoder `phi` (and the SD-VAE decoder when `phi` lives in pixel space).
5. Compute the multi-scale, multi-temperature drifting loss in feature space; per-feature normalization makes the loss invariant to feature scale.
6. Backprop only through `f_theta(eps)` (stop-grad on `V`).

![Attraction and repulsion](/assets/images/paper/drifting/fig_p004_01.png)
*Figure 2: A generated sample `x` is attracted by `V^+_p` (positives, blue) and repulsed by `V^-_q` (negatives, orange); the resulting drift is `V = V^+ - V^-`. Anti-symmetry of `V` is the structural reason `q=p` is a fixed point.*

![Distribution evolution](/assets/images/paper/drifting/fig_p006_01.png)
*Figure 3: Evolution of `q_theta` toward a bimodal target under three initializations — between modes, far from data, and fully collapsed. The drift recovers both modes in all cases.*

![Loss curve](/assets/images/paper/drifting/fig_p006_02.png)
*Figure 4: Drifting loss (numerically equal to `E ||V(f(eps))||^2`) decreases monotonically as `q_theta` converges to the target on the 2D toy. This is the empirical anchor for the otherwise-unproven converse `V ~ 0 => q ~ p`.*

**Architecture details.** DiT-style transformer with SwiGLU, RoPE, RMSNorm, QK-Norm. 16 register tokens + 32 "style tokens" (random codebook indices, +0.4 FID from 8.86 → 8.46 in ablation). adaLN-zero for `(c, alpha)` conditioning. L/2 is hidden 1024 / depth 24 (same shape as DiT-L, not DiT-XL — worth noting when reading the "B beats XL" framing).

## Experimental Results

### ImageNet 256x256 — latent (Table 5)

| Method | NFE | Params (gen+dec) | FID ↓ | IS ↑ |
|---|---|---|---|---|
| DiT-XL/2 | 250x2 | 675M+49M | 2.27 | 278.2 |
| SiT-XL/2 | 250x2 | 675M+49M | 2.06 | 270.3 |
| SiT-XL/2 + REPA | 250x2 | 675M+49M | 1.42 | 305.7 |
| LightningDiT-XL/2 (VA-VAE) | 250x2 | 675M+70M | 1.35 | 295.3 |
| RAE+DiTDH-XL/2 | 50x2 | 839M+415M | 1.13 | 262.6 |
| iCT-XL/2 | 1 | 675M+49M | 34.24 | – |
| Shortcut-XL/2 | 1 | 675M+49M | 10.60 | – |
| MeanFlow-XL/2 | 1 | 676M+49M | 3.43 | – |
| AdvFlow-XL/2 | 1 | 673M+49M | 2.38 | 284.2 |
| iMeanFlow-XL/2 | 1 | 610M+49M | 1.72 | 282.0 |
| **Drifting Model B/2 (ours)** | 1 | **133M**+49M | **1.75** | 263.2 |
| **Drifting Model L/2 (ours)** | 1 | 463M+49M | **1.54** | 258.9 |

### ImageNet 256x256 — pixel (Table 6)

| Method | NFE | Params | FID ↓ | IS ↑ |
|---|---|---|---|---|
| ADM-G | 250x2 | 554M | 4.59 | 186.7 |
| VDM++ (UViT/2) | 256x2 | 2.5B | 2.12 | 267.7 |
| SiD2 (UViT/1) | 512x2 | – | 1.38 | – |
| JiT-G/16 | 100x2 | 2B | 1.82 | 292.6 |
| PixelDiT/16 | 200x2 | 797M | 1.61 | 292.7 |
| EPG-L/16 | 1 | 540M | 8.82 | – |
| BigGAN | 1 | 112M | 6.95 | 152.8 |
| GigaGAN | 1 | 569M | 3.45 | 225.5 |
| StyleGAN-XL | 1 | 166M | 2.30 | 265.1 |
| **Drifting Model B/16 (ours)** | 1 | **134M** | **1.76** | 299.7 |
| **Drifting Model L/16 (ours)** | 1 | 464M | **1.61** | **307.5** |

### Key ablations

- **Anti-symmetry is load-bearing (Table 1).** `1.5V+ - V-` → FID 41.05; `2V+ - V-` → 86.16; attraction-only → 177.14; default `V+ - V-` → 8.46. The equilibrium property is not cosmetic.
- **Positive/negative budget (Table 2).** Under fixed compute `B = N_c x N_pos = 4096`, balanced `N_pos = N_neg = 64` dominates many-class/few-sample regimes: 20.43 → 8.46.
- **Feature encoder (Table 3).** SimCLR 11.05, MoCo-v2 8.41, latent-MAE-256 8.46. Scaling latent-MAE to width 640 + 1280 epochs → **4.28**, +cls fine-tune → **3.36**. The encoder is the single largest known lever after model size.
- **Multi-scale features:** (a,b) 9.58 → (a-d) 8.46.
- **Multi-temperature:** single tau=0.05 → 8.67; multi-tau → 8.46.
- **Style tokens:** 8.86 → 8.46.
- **From ablation to final (Table 4):** B/2 100ep 3.36 → 320ep 2.51 → 1280ep + hp 1.75 → L/2 1280ep **1.54**.
- **Pixel-space scaling (Table 9-10):** MAE-only 32.11 → +ConvNeXt-V2 3.70 → longer/larger **1.61**.
- **Robotics (Table 7).** 1-NFE Drifting Policy matches or beats 100-NFE Diffusion Policy on Lift, Can, ToolHang-State, BlockPush phase 1; ±0.05 on the rest. Two regressions (PushT-State 0.86 vs 0.91; ToolHang-Visual 0.67 vs 0.73) are present and **not discussed in text**.

![CFG sweep](/assets/images/paper/drifting/fig_p017_01.png)
*Figure 5: CFG scale `alpha` trades FID for IS — L/2 reaches optimal FID at `alpha = 1.0` (no-CFG regime in diffusion terms); B/2 optimum at `alpha = 1.1`. Even with training-time CFG, inference stays 1-NFE.*

![Uncurated sample](/assets/images/paper/drifting/fig_p020_01.png)
*Figure 6 (qualitative): An uncurated 1-NFE sample from Drifting L/2 (latent, CFG=1.0, headline FID 1.54).*

![Side-by-side vs improved MeanFlow](/assets/images/paper/drifting/fig_p024_01.png)
*Figure 7 (qualitative): Side-by-side at matched IS — Drifting Model (left) vs improved MeanFlow (right), both 1-NFE. The paper reports Drifting at FID 3.01 / IS 354.4 vs iMF 3.92 / 348.2 in this matched-IS regime.*

## Limitations

**Authors admit:**
- The converse `V → 0 ⇒ q → p` is not proven in general — only the forward direction has a clean proof.
- Method does **not** converge on ImageNet without a pretrained feature encoder (§5.2). This is a hidden dependency: the headline result is really "Drifting + a strong encoder," and the encoder is itself an SSL pipeline (latent-MAE-640, 1280 epochs).
- Pixel-space generation needs an additional ConvNeXt-V2 stack on top of MAE to close the latent–pixel gap.
- Pixel-space training was 640 epochs vs 1280 for latent — "expect further improvements."

**Authors did not address (and a reader should flag):**
- **No seed variance.** The headline 1.54 vs iMeanFlow 1.72 is a 0.18-FID gap, well within typical seed noise on ImageNet 50k FID. No error bars, no run-to-run statistics — standard for the subfield but particularly relevant here given how tight the leaderboard is.
- **Training-compute parity is unstated.** Inference FLOPs vs StyleGAN-XL (87G vs 1574G, ~18x) is advertised; training-FLOP comparison vs DiT-XL / iMeanFlow is silent. 1280 epochs on ImageNet × multi-scale features × VAE-decoder backprop is non-trivial — without a training-compute table the inference-FLOP boast is one-sided.
- **Only FID and IS** — no precision/recall, no CMMD, no DINOv2-FID, no human eval, no quantitative diversity metric. The mode-collapse argument rests on a 2D toy (Fig. 3) and a one-page CLIP-NN retrieval (Fig. 6), not a recall number.
- **No high-resolution results.** 256x256 only — no 512 or 1024, no proof of resolution scaling.
- **No text-conditioning / compositionality.** Class-conditional ImageNet only.
- **Encoder-choice sensitivity.** Documented but not quantified as a robustness concern. For any domain shift (medical CT/MRI/H&E) the encoder dependence is a real practical risk.
- **Two robotics regressions** (PushT-State, ToolHang-Visual) are present in Table 7 but not discussed.
- **"L/2" is DiT-L-shaped, not DiT-XL-shaped.** The "B/2 (133M) beats XL baselines" framing in §5.2 is fair on parameter count, but the headline 1.54 uses L/2 at 463M — comparable to XL-class. The honest framing is "best parameter-efficient one-step model," not "Base beats XL."
- **Provenance / positioning.** This is Kaiming He's lab, with strong incentive to be carefully positioned vs the SiT / REPA / MAR / iMeanFlow lineage. The MMD-as-special-case framing (Appendix C.2) and the "new paradigm" pitch should be read with that in mind — the comparison baselines are the right ones, but the head-to-head fairness (matched training compute, matched encoder budget) is not enforced anywhere.

## Why It Matters for Medical AI

The direct medical relevance is not yet demonstrated — no medical dataset, no segmentation/registration/synthesis benchmark. The indirect relevance is real:

- **Latency-bound generative pipelines.** Real-time synthesis (intra-operative imaging hallucination, surgical-scene generation for sim-to-real, interactive contour-edit refinement) cannot afford 50-250 NFEs. A 1-NFE generator with FID parity to multi-step diffusion is exactly the substrate those applications need.
- **Robotic policies (§5.3).** The Drifting-Policy result — 1-NFE matches 100-NFE Diffusion Policy on most Robomimic tasks — is the closest the paper gets to a clinical-adjacent claim. Surgical-robotics policy learning sits in the same regime.
- **Encoder dependence is a deployment concern.** Medical domains rarely have a SimCLR/MoCo-v2 equivalent of the strength used here. The paper's own admission that the method "does not work on ImageNet without a feature encoder" is a flag: porting Drifting to MRI / CT / pathology will require either a strong domain-pretrained encoder or a robustness study the paper did not run.

## References

- Paper (arXiv): [Generative Modeling via Drifting, arXiv:2602.04770](https://arxiv.org/abs/2602.04770) (v2, 6 Feb 2026)
- Project page: [lambertae.github.io/projects/drifting](https://lambertae.github.io/projects/drifting)
- Authors: Mingyang Deng, He Li, Tianhong Li (MIT), Yilun Du (Harvard), Kaiming He (MIT)
- Related: SiT, REPA, LightningDiT, RAE+DiTDH (multi-step latent diffusion baselines); iCT, Shortcut, MeanFlow, AdvFlow, **iMeanFlow** (one-step baselines); SimSiam / iCT (stop-gradient fixed-point pattern); Diffusion Policy (Chi et al. 2023, robotics baseline); MMD (the non-normalized special case discussed in App. C.2).
