# The Lazy Artist: Diagnosing and Curing Spurious Correlations in CNNs

> *"The eye sees only what the mind is prepared to comprehend."* — Henri Bergson

This project is a comprehensive, end-to-end investigation into **spurious correlations** in Convolutional Neural Networks. We deliberately bias a model with a color-correlated MNIST dataset, diagnose its failure modes using multiple interpretability techniques, intervene to cure the bias, stress-test with adversarial attacks, and decompose internal representations using Sparse Autoencoders.

All work is contained in a single notebook: **`precog_comp.ipynb`** (39 cells, ~2700 lines).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment & Dependencies](#environment--dependencies)
3. [Dataset: Colored MNIST](#dataset-colored-mnist)
4. [Task 0: The Biased Canvas](#task-0-the-biased-canvas)
5. [Task 1: The Cheater](#task-1-the-cheater)
6. [Task 2: The Prober](#task-2-the-prober)
7. [Task 3: The Interrogation](#task-3-the-interrogation)
8. [Task 4: The Intervention](#task-4-the-intervention)
9. [Task 5: The Invisible Cloak](#task-5-the-invisible-cloak)
10. [Task 6: The Decomposition](#task-6-the-decomposition)
11. [Key Results Summary](#key-results-summary)
12. [How to Run](#how-to-run)
13. [File Structure](#file-structure)
14. [Design Decisions & Rationale](#design-decisions--rationale)

---

## Project Overview

Modern CNNs are powerful but opportunistic — they will exploit any statistical shortcut available in the training data, even if that shortcut has no causal relationship to the target concept. This project demonstrates this phenomenon end-to-end:

1. **Create** a synthetically biased dataset where digit identity is 95% correlated with background color
2. **Train** a CNN that exploits this spurious correlation, achieving >95% train accuracy but <20% test accuracy
3. **Diagnose** the bias using activation maximization, Grad-CAM, causal ablations, pixel-level attribution, confusion matrices, and 10×10 prediction grids
4. **Cure** the model using color-invariant augmentation strategies (ColorJitter + channel permutation)
5. **Attack** both biased and debiased models with targeted adversarial perturbations
6. **Decompose** internal representations with a Sparse Autoencoder to discover and manipulate individual color/shape features

---

## Environment & Dependencies

| Package        | Purpose                                         |
|----------------|------------------------------------------------|
| `torch`        | Core deep learning framework                    |
| `torchvision`  | MNIST dataset, transforms, pretrained models    |
| `numpy`        | Numerical operations                            |
| `matplotlib`   | All visualizations and plots                    |
| `seaborn`      | Heatmaps (confusion matrices, prediction grids) |
| `opencv-python`| GradCAM heatmap resizing                        |
| `scikit-learn` | Confusion matrix computation                    |

**Hardware:** CUDA GPU recommended (auto-detects and falls back to CPU).

```bash
pip install torch torchvision numpy matplotlib seaborn opencv-python scikit-learn
```

---

## Dataset: Colored MNIST

### Color Map

Each digit (0–9) is assigned a unique dominant color:

| Digit | Color   | RGB               |
|-------|---------|-------------------|
| 0     | Red     | `(1.0, 0.0, 0.0)` |
| 1     | Green   | `(0.0, 1.0, 0.0)` |
| 2     | Blue    | `(0.0, 0.0, 1.0)` |
| 3     | Yellow  | `(1.0, 1.0, 0.0)` |
| 4     | Magenta | `(1.0, 0.0, 1.0)` |
| 5     | Cyan    | `(0.0, 1.0, 1.0)` |
| 6     | Orange  | `(1.0, 0.5, 0.0)` |
| 7     | Purple  | `(0.5, 0.0, 1.0)` |
| 8     | Lime    | `(0.5, 1.0, 0.0)` |
| 9     | Pink    | `(1.0, 0.0, 0.5)` |

### Bias Rules

- **Training set (Easy):** 95% of images use the digit's assigned color as the background. 5% use a random *different* color (counter-examples).
- **Test set (Hard):** Colors are fully randomized — no correlation between digit and color.

### Background Texture Construction

The background is **not** a flat solid color. Instead:
1. A single-channel noise texture is generated: `0.5 + 0.5 * rand(1, 28, 28)` (range [0.5, 1.0])
2. This grayscale noise is repeated across all 3 RGB channels, then multiplied by the assigned color vector
3. The digit stroke (white foreground) is composited on top using a binary mask

This produces a **colored textured background** that is more realistic and harder to trivially strip away than a flat fill.

### Dataset Classes

- **`ColoredMNIST(train=True/False)`** — The biased dataset described above
- **`RegularMNIST(train=True/False)`** — Standard grayscale MNIST expanded to 3 channels (control experiment)

---

## Task 0: The Biased Canvas

**Cells:** 1–8 (Markdown header + imports, device setup, color map, visualization, `ColoredMNIST` class, `RegularMNIST` class, sample visualization)

### What happens

- Defines the 10-color palette and visualizes it as a bar chart
- Implements the `ColoredMNIST` dataset with the 95%/5% bias rule and textured backgrounds
- Implements `RegularMNIST` as a control (grayscale → 3-channel, no color bias)
- Displays a random sample from the biased training set to visually confirm the colored background

### Key design choice: Textured backgrounds

A flat solid-color background would be trivially detected by even a single-pixel model. The textured background (channel-consistent noise × color) ensures that color information is spatially distributed, making the bias more realistic and harder to separate.

---

## Task 1: The Cheater

**Cells:** 9–13 (Model definition, training setup, control experiment on regular MNIST, biased training, accuracy reporting)

### Model Architecture: NanoNet

A deliberately small CNN to examine bias without overparameterization confounds:

```
NanoNet:
  conv1: Conv2d(3, 8, 3×3, pad=1)  + BN + ReLU + MaxPool(2)   → 14×14
  conv2: Conv2d(8, 16, 3×3, pad=1) + BN + ReLU + MaxPool(2)   → 7×7
  conv3: Conv2d(16, 16, 3×3, pad=1)+ BN + ReLU + MaxPool(2)   → 3×3
  AdaptiveAvgPool(1,1) → flatten → fc(16, 10)
```

- **No bias terms** in conv layers (`bias=False`) — forces batch norm to handle centering
- **3×3 kernels** — can learn *both* spatial (shape) and color features (unlike 1×1)
- **16-dim** penultimate representation — small enough for SAE decomposition

### Training Configuration

| Parameter  | Value |
|------------|-------|
| Batch size | 2048  |
| Optimizer  | Adam  |
| Learning rate | 0.01 |
| Epochs     | 3     |
| Loss       | CrossEntropyLoss |

### Control Experiment

An **identical** NanoNet is trained on `RegularMNIST` with the same hyperparameters (3 epochs, LR 0.01, batch 2048) to prove that the architecture has sufficient capacity for >97% accuracy on standard MNIST. This isolates **dataset bias** as the sole variable.

### Results

- **Biased training:** >95% train accuracy (model memorizes the color shortcut)
- **Hard test:** <20% accuracy (model fails catastrophically when colors are randomized)
- **Control (Regular MNIST):** ~97% test accuracy (same model, no bias)

### Analysis produced

- Training/test accuracy table per epoch
- Explicit pass/fail checks against thresholds (>95% train, <20% hard test)

---

## Task 2: The Prober

**Cells:** 14–16 (Markdown header, helper definitions + activation maximization engine, all 4 experiments)

### Goal

Visualize what neurons in the biased model have learned using **activation maximization** ("feature dreams"), without relying on external libraries.

### Core Engine: `feature_dream()`

Activation maximization from scratch with transformation-robustness priors:

```python
feature_dream(model, target_kind='class'|'neuron'|'channel_max',
              target_idx, layer_name, steps=300, lr=0.05)
```

**Optimization details:**
- Starts from gray noise (`0.5 + 0.01 * randn`)
- Adam optimizer, iteratively maximizes a target activation
- **Jitter augmentation:** Random 2D shift (±3 px) each step → prevents high-frequency adversarial patterns
- **Total Variation (TV) loss:** `0.01 * TV` regularizer for smoothness
- **Periodic Gaussian blur:** Every 5 steps (except last 20) — suppresses noise while preserving structure
- **Clamp [0, 1]:** Keeps pixel values in valid image range

### Experiment 1: Class Dreams (All 10 Classes)

- Generates the "ideal input" for each of the 10 output classes
- Displays dreams side-by-side with expected color swatches
- **Quantitative:** Computes cosine similarity between each dream's mean RGB and the expected color from `color_map`
- **Result:** ≥6/10 classes dream their expected color → model is **color-biased**

### Experiment 2: Neuron-Level Visualization

- Dreams for individual filters in `conv1` (8 filters), `conv2` (16 filters), `conv3` (16 filters)
- Shows 8 filter dreams per layer in a grid
- **Observation:** conv1 detects color primitives + edges; conv2/conv3 show increasingly complex color patterns; very few neurons exhibit spatial digit-like structure

### Experiment 3: Polysemanticity Analysis

For each of the 16 `conv3` filters, measures activation across a **10×10 (digit, color) controlled stimulus grid:**

- **Color variance** = variance of per-color mean activations (high → color-selective)
- **Shape variance** = variance of per-digit mean activations (high → shape-selective)
- **Color fraction** = `cv / (cv + sv)`:  >0.7 = COLOR, <0.3 = SHAPE, else POLYSEMANTIC

Produces:
- A per-neuron classification table (COLOR / SHAPE / POLYSEMANTIC)
- Activation heatmaps for the top 4 color-selective and top 4 shape/mixed neurons
- Summary statistics

### Experiment 4: Deep Dream

Amplifies existing activations in **real images** (not from noise):

```python
deep_dream(model, start_img, target_layer_name, steps=100, lr=0.02)
```

- Starts from a real colored-MNIST image
- Iteratively maximizes the L2 norm of feature activations in target layer (`conv3`)
- Produces 3-column visualization: Original → Deep Dream → |Difference|
- Tested on both biased (correct color) and conflicting (wrong color) images
- **Finding:** The model amplifies background color regions, not digit strokes

### Supplementary: Lazy vs Robust Dreams (Cell 38–39)

After Task 4 trains the robust model, a side-by-side comparison shows:
- **Lazy model dreams:** Colored textures matching the color map
- **Robust model dreams:** More spatial structure, less color dominance

---

## Task 3: The Interrogation

**Cells:** 19, 22–28 (GradCAM class, Grad-CAM analysis, pixel-level attribution, causal ablation, confusion matrices, 10×10 prediction grid, color-swap litmus test, grayscale test)

### GradCAM Implementation (From Scratch)

```python
class GradCAM:
    def __init__(self, model, target_layer)
    def generate_heatmap(self, input_image, class_idx=None)
    def remove_hooks(self)
```

**Key implementation details:**
- Hooks `BatchNorm` layers (not conv layers directly) — captures post-normalization activations
- Manually applies ReLU to hooked activations (since BN comes before ReLU in NanoNet)
- Computes channel importance weights via global average pooling of gradients
- Weighted combination → ReLU → resize to input dimensions → normalize [0, 1]
- Targets `bn2` (14×14 resolution) by default for good spatial resolution

### Grad-CAM Analysis (10 test cases)

Tests 5 biased + 5 conflicting images:
- **4-column visualization:** Original | Heatmap (hot) | Overlay (jet α=0.6) | Quantitative bar chart
- **Density-based BG percentage:** Per-pixel mean intensity on background vs foreground (normalizes for the ~6× pixel count imbalance of BG vs digit)
- Separates analysis for biased vs. conflicting cases
- Reports whether conflicting cases follow COLOR or SHAPE

### Pixel-Level Attribution (28×28 Full Resolution)

Standard GradCAM@bn3 produces 7×7 heatmaps upsampled to 28×28, creating misleadingly blurry blobs. This section provides **four independent attribution methods** at full resolution:

| Method | Resolution | Type | Description |
|--------|-----------|------|-------------|
| Gradient Saliency | 28×28 native | Gradient | `\|∂prediction/∂pixel\|` — direct pixel sensitivity |
| Occlusion Sensitivity | 28×28 native | Causal | Cover 4×4 patches, stride 2, measure prediction confidence drop |
| GradCAM @ bn1 | 28×28 native | Gradient | First-layer attention (no upsampling needed) |
| GradCAM @ bn3 | 7×7 → 28×28 | Gradient | Standard approach (for comparison) |

- 6-column visualization per test case: Original | Saliency | Occlusion | CAM@bn1 | CAM@bn3 | BG% bar chart
- All methods report density-based BG focus percentage
- **Finding:** Saliency + Occlusion (highest-res, most trustworthy) confirm BG focus → color-biased

### Causal Ablation Proof (5 test digits)

Directly manipulates images without interpreting heatmaps:

| Manipulation | What it tests |
|-------------|---------------|
| **Digit erased** (pure colored BG, no digit) | Does model need the digit shape at all? |
| **Background erased** (digit on neutral gray) | Does removing color change prediction? |
| **Color → GREEN** (same digit, green BG) | Does prediction follow green (=digit 1)? |
| **Color → BLUE** (same digit, blue BG) | Does prediction follow blue (=digit 2)? |

- 5-column visualization per digit with color-coded borders
- Counts: digit-erased-not-shape, bg-erased-changed, green-follows-color, blue-follows-color
- **Conclusion:** "Definitive causal proof of spurious correlation bias"

### Confusion Matrices

- **Training set (Blues cmap):** Near-perfect diagonal — model scores >95%
- **Hard test set (Reds cmap):** Scattered predictions — model collapses completely

### 10×10 Prediction Grid

- Tests all 100 (digit × color) combinations with majority voting over 10 noise seeds per cell
- Displays grid as a heatmap alongside the ideal color-biased grid (each column = column index)
- Reports off-diagonal analysis: X/90 follow COLOR vs X/90 follow SHAPE
- **Quantitative proof** that predictions track column (color), not row (digit)

### Quick Verification: Red-Colored Digits

Feeds all 10 digits colored RED (digit 0's color):
- If color-biased → all predict 0
- Counts color-followers vs shape-followers

### Grayscale Test

Tests the lazy model on regular grayscale MNIST (expanded to RGB but no color):
- Proves whether the model learned **any** shape representation at all
- If accuracy is high → model has both skills but prefers color shortcut (realistic spurious correlation)
- If accuracy is near-random → model learned color as sole strategy

---

## Task 4: The Intervention

**Cells:** 17–18, 20 (Training cell, GradCAM class reuse, 3-test verification)

### Diagnosis of Why Naive Approaches Fail

The code documents a previous failed attempt (82% plateau) due to:
1. **WeightedRandomSampler was broken** — the stochastic dataset regenerates colors on every access, so upweighting "rebel indices" was ineffective
2. **Rebel Mixup poisoned the signal** — 50/50 blending every image with a rebel created muddy, unnatural inputs
3. **Mild jitter (hue=0.15 = ±54°)** — model could still distinguish warm vs cool hues

### The Fix: Two Complementary Methods

**Method 1: Strong ColorJitter (hue=0.5 = ±180° = full color wheel)**
- Red can become ANY color → color is pure noise
- White digit strokes are unaffected by hue rotation (white has no hue)
- Only the colored background is randomized — shape signal is perfectly preserved
- Images are still colorful (NOT grayscale conversion)

**Method 2: Random Channel Permutation (all 6 RGB orderings)**
- Even after jitter, the model might learn channel-specific patterns
- `random.choice(ALL_PERMS)` shuffles RGB channels per-batch
- Makes R, G, B channels fully interchangeable

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Fresh NanoNet (random init) |
| Batch size | 256 (smaller = more gradient updates/epoch) |
| Optimizer | Adam (LR = 0.01) |
| Scheduler | CosineAnnealingLR (T_max=15) |
| Epochs | 10 |
| Loss | CrossEntropyLoss |

### Verification: Three Independent Tests

**Test 1: GradCAM Showdown (Lazy vs Robust)**
- 8 test cases (biased + conflicting)
- 5-column visualization: Original | Lazy heatmap overlay | Robust heatmap overlay | Lazy heatmap | Robust heatmap
- Reports density-based BG% for both models
- Lazy → high BG focus (color attention) | Robust → lower BG focus (digit attention)

**Test 2: Color-Swap Litmus Test**
- All 10 digits colored RED → Lazy should predict "0" for everything; Robust should predict actual digit
- Counts color-followers (lazy) vs shape-followers (robust)

**Test 3: 10×10 Prediction Grid (Robust Model)**
- Same protocol as Task 3 but for the robust model
- Each row should be constant (= actual digit) regardless of color column
- Reports grid accuracy out of 100
- Heatmap comparison: Robust grid vs Ideal shape-based grid
- **Target:** ≥90/100 correct cells

---

## Task 5: The Invisible Cloak

**Cells:** 29–32 (Markdown header, attack implementation, visualization, convergence plot)

### Goal

Craft a **targeted adversarial attack**: take a "7", make the model predict "3" with >90% confidence, with L∞ perturbation ε < 0.05 (invisible to humans).

### Attack: Adam-PGD with CE + CW Margin Loss Blend

```python
strong_targeted_attack(model, clean_img, target_class, epsilon,
                       num_steps=3000, num_restarts=20, lr=0.02)
```

**Attack details:**
- **Loss function:** Blends Cross-Entropy (strong gradient far from boundary) with Carlini-Wagner margin loss (precise near boundary) using linear interpolation `α = step/total_steps`
- **Optimizer:** Adam with cosine LR decay
- **Initialization:** Multiple random restarts — restart 0 is zero-init, others are uniform[-ε, ε]
- **Projection:** After each step, clamp delta to [-ε, ε] and ensure adv is in [0, 1]
- **Best tracking:** Keeps the highest-confidence adversarial across all restarts
- **Early stopping:** Breaks if target class confidence >90%

### Intelligent Source Selection

Not all "7" images are equally attackable. `pick_best_source()` selects the "7" whose colored version:
1. Is correctly classified as 7 (legitimate attack starting point)
2. Has the highest logit for class 3 (least perturbation needed — closest to decision boundary)

Separate best sources are selected for the lazy and robust models.

### Minimum ε Scan

If ε=0.05 fails for either model, a binary-search-style scan tests: `[0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]`

### Configuration

| Parameter | Value |
|-----------|-------|
| Source digit | 7 |
| Target digit | 3 |
| ε (L∞ budget) | 0.049 |
| Optimization steps | 3000 |
| Random restarts | 20 |
| Learning rate | 0.02 |

### Key Finding

- **Robust model (shape-based):** Fooled at ε ≈ 0.049 — shape features are susceptible to subtle high-frequency perturbations
- **Lazy model (color-based):** Resists until ε ≈ 0.300 — changing Purple→Yellow requires per-channel shifts of Δ≈0.5–1.0, far exceeding the ε budget
- **~6× gap** — the "better" model is paradoxically **easier** to attack with small L∞ noise

### Visualizations

1. **2×3 grid:** Original | Adversarial | Perturbation magnitude heatmap — for both models
2. **Convergence plot:** Logit margin (z_target − z_max_other) vs optimization step for both models, with decision boundary annotated

---

## Task 6: The Decomposition

**Cells:** 33–37 (Markdown header, SAE architecture, SAE training, feature discovery, causal interventions)

### Architecture: Sparse Autoencoder

```
SparseAutoencoder:
  input_dim = 16 (penultimate layer of NanoNet)
  hidden_dim = 128 (8× expansion — overcomplete)
  encoder:  Linear(16, 128) + learnable bias → ReLU
  decoder:  Linear(128, 16) + learnable bias
```

**Why SAE over alternatives?**

| Method | Limitation |
|--------|-----------|
| PCA | Rotation — cannot separate correlated concepts (Red ≈ Digit 0) |
| Linear probes / CAVs | Supervised — find what you ask for, not what exists |
| NMF | Undercomplete — cannot resolve superposition |
| **SAE** | **Overcomplete + sparse → unsupervised disentanglement of superposition** |

### SAE Training

| Parameter | Value |
|-----------|-------|
| Training data | 10,000 activations from lazy model's penultimate layer |
| L1 coefficient | 3×10⁻⁴ |
| Optimizer | Adam (LR=1×10⁻³) |
| Batch size | 256 |
| Epochs | 20 |

Activations are collected via a forward hook on `model.fc` (captures the input to the final linear layer = 16-dim vector after adaptive pooling).

### Systematic Feature Discovery

1. **Controlled stimulus grid:** 10 digits × 10 colors × 5 seeds → averaged activations per cell → `[10, 10, 128]` tensor
2. **Selectivity metric:**
   - Color selectivity = variance of (mean-over-digits) across colors
   - Shape selectivity = variance of (mean-over-colors) across digits
   - Selectivity = `(color_var - shape_var) / (color_var + shape_var)` ∈ [-1, +1]
   - `>0.3` = COLOR, `<-0.3` = SHAPE, else MIXED
3. **Feature census:** Counts alive features (activation > 0.01), color-selective, shape-selective, mixed
4. **Heatmap visualization:** Top 5 color and top 5 shape features shown as 10×10 (digit × color) activation grids
5. **Selectivity distribution:** Bar chart of all alive features sorted by selectivity
6. **Polysemanticity analysis:** For top color and shape features, measures whether they respond to one concept or multiple

**Key finding:** 116/128 alive features are color-selective, 0 shape-selective → the model's penultimate representation is **entirely** color-dominated.

### Causal Interventions

Uses the **residual method** to surgically modify individual features:

```
new_activations = original + (SAE.decode(modified_hidden) - SAE.decode(original_hidden))
```

This preserves information the SAE doesn't capture while applying ONLY the targeted feature change.

#### Experiment 1: Scale Sweep

Scales all color features from 0× (ablated) to 3× (amplified) on biased images:
- Scale=1.0 → original behavior
- Scale=0.0 → color features zeroed out
- Plots accuracy vs scale factor

#### Experiment 2: Probability Distributions

For 5 demo cases (biased + conflicting), shows full 10-class probability distributions under:
- Original (1×) intervention
- Ablated (0×) intervention
- Amplified (3×) intervention

Green edge = true digit, Red edge = wrong prediction.

#### Experiment 3: Full Hard Test Set Accuracy

Evaluates on the complete hard test set (10,000 images):
- **Original (1.0×):** ~baseline accuracy
- **Ablated (0.0×):** Accuracy improves if color features were causing misclassification
- **Amplified (3.0×):** Accuracy decreases if color bias is strengthened

**Causal proof:** If ablation improves accuracy on color-swapped images → color features are the PRIMARY driver of the model's failures.

#### Experiment 4: Control — Ablate Shape Features

Ablates shape-selective features instead (if any exist) to show that removing the *wrong* features does not help.

---

## Key Results Summary

| Task | Key Metric | Result |
|------|-----------|--------|
| **Task 0** | Dataset creation | 95% bias, textured colored backgrounds |
| **Task 1** | Train accuracy / Hard test accuracy | >95% / <20% |
| **Task 1** | Control (Regular MNIST, same model) | ~97% test accuracy |
| **Task 2** | Class dreams matching expected color | ≥6/10 (cosine sim > 0.7) |
| **Task 2** | Polysemanticity | Majority of conv3 neurons are color-selective |
| **Task 3** | Conflicting images follow color | Majority follow COLOR, not SHAPE |
| **Task 3** | Off-diagonal prediction grid | ~90% follow color |
| **Task 4** | Robust model hard test accuracy | >70% (target met) |
| **Task 4** | 10×10 grid accuracy | ≥80/100 correct cells |
| **Task 5** | Robust model ε to fool (7→3) | ~0.049 |
| **Task 5** | Lazy model ε to fool (7→3) | ~0.300 (6× harder) |
| **Task 6** | SAE feature census | ~116/128 alive = color-selective |
| **Task 6** | Hard test accuracy after ablation | Improves vs original |

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib seaborn opencv-python scikit-learn
```

Ensure MNIST data is available (will auto-download to `./data/MNIST/`).

### Execution

Open `precog_comp.ipynb` in Jupyter/VS Code and **run all cells sequentially** (top to bottom). The notebook is designed to be executed in order — later tasks depend on models and variables from earlier tasks.

**Approximate runtimes (with GPU):**
- Tasks 0–1 (dataset + training): ~2 minutes
- Task 2 (activation maximization): ~5 minutes (300 optimization steps per dream)
- Task 3 (GradCAM + attribution): ~3 minutes
- Task 4 (robust retraining + verification): ~5 minutes
- Task 5 (adversarial attacks): ~10–20 minutes (3000 steps × 20 restarts × 2 models)
- Task 6 (SAE training + interventions): ~3 minutes

**Total: ~30 minutes on GPU, longer on CPU.**

### Important Notes

- Task 3 and Task 4's verification cells reference `digit_images` and `make_bg_textured` which are defined in the Task 2 setup cell — ensure all cells run in order.
- Task 5 runtime depends heavily on how many ε-scan values are needed (if initial attack succeeds, the scan is skipped).
- The lazy model (`model`) and robust model (`robust_model`) are both kept in memory simultaneously for comparison.
- The SAE hook is attached to the lazy model's `fc` layer — cleaning up (`hook_handle.remove()`) is recommended after Task 6 if continuing to use the model.

---

## File Structure

```
precog/
├── precog_comp.ipynb         # Main notebook (this project — all 6 tasks)
├── README.md                 # This file
├── model_nanonet.pth         # Saved model weights (optional)
├── tasks_to_be_done.md       # Original task specification
├── data/
│   └── MNIST/
│       └── raw/              # MNIST dataset files (auto-downloaded)
├── images/                   # Output images (if saved)


The other `precog_fin_*.ipynb` files are earlier iterations/experiments. **`precog_comp.ipynb` is the final, comprehensive submission.**

---

## Design Decisions & Rationale

### Why NanoNet instead of ResNet-18?

- **Transparency:** 16-dim penultimate layer is small enough for complete SAE decomposition
- **Sufficiency:** Achieves 97% on regular MNIST, proving capacity is adequate
- **Speed:** Trains in seconds, enabling rapid experimentation
- **Interpretability:** Fewer layers/filters = easier to visualize all neurons

### Why textured backgrounds instead of flat colors?

Flat solid colors can be detected by a single pixel. The textured background distributes color information spatially, creating a more realistic and challenging bias scenario that mirrors real-world spurious correlations (e.g., snow in wolf images).

### Why hook BatchNorm layers for GradCAM instead of Conv layers?

NanoNet's architecture is `Conv → BN → ReLU → Pool`. Hooking BN captures post-normalization features before ReLU. Since we manually apply ReLU in the GradCAM code, this gives us the actual feature representations the network uses downstream, with proper normalization.

### Why density-based BG% instead of raw energy?

The background has ~6× more pixels than the digit foreground (~85% vs ~15%). Raw heatmap energy will always show ~85%+ on background even for a perfectly shape-focused model. Density-based measurement (mean intensity per region) normalizes for pixel count, making 50% the threshold for unbiased attention.

### Why ε=0.049 instead of 0.05?

The task requires ε **strictly less than** 0.05. Using 0.049 satisfies this with a small margin while maximizing the perturbation budget.

### Why residual SAE intervention instead of direct replacement?

The SAE has reconstruction error — replacing activations with `SAE.decode(modified)` introduces systematic artifacts. The residual method `orig + (decode(modified) - decode(original))` cancels out reconstruction error, applying only the targeted feature change. This is analogous to editing one word in a sentence without rewriting the whole paragraph.

### Why Adam-PGD with CE+CW blend instead of standard PGD?

Standard PGD with sign-based updates is noisy for targeted attacks with tight ε budgets. Adam provides adaptive per-parameter learning rates. The CE→CW loss blend provides strong gradient signal far from the boundary (CE) while refining precisely near it (CW margin). Multiple random restarts escape local minima.

---

## Evolution & Rejected Approaches

This project went through multiple iterations (`precog_fin.ipynb` → `precog_fin_33` → `precog_fin_55` → `precog_fin_f5` → `precog_fin_f6` → `precog_comp.ipynb`). Below are the key approaches that were tried and abandoned, and why.

### Dataset: Flat Foreground Coloring → Textured Backgrounds

**Earlier approach (`precog_fin.ipynb`):** The colored digit was created by multiplying the grayscale image directly by the color vector: `img * torch.tensor(color).view(3, 1, 1)`. This colored the **digit stroke** itself (foreground), leaving the background black.

**Why it was abandoned:** A flat-colored foreground on a black background is trivially separable — the model can detect digit shape just from the brightness pattern, and color from any single non-zero pixel. This makes the spurious correlation too easy to break and doesn't mirror real-world bias (e.g., textured snow behind wolves). The final textured-background approach distributes color spatially across the entire image, creating a more realistic and challenging bias.

### Model: ResNet-18 / LazyNet / ColorBiasedNet → NanoNet

**Earlier approaches:**
- **ResNet-18** (`precog_fin.ipynb`, commented out): Modified with `conv1=1×1` and `maxpool=Identity` to fit 28×28 inputs. Too large (11M params) — overparameterized for MNIST, slow to train, and the 512-dim penultimate layer is too large for meaningful SAE decomposition in Task 6.
- **LazyNet** (`precog_fin.ipynb`, commented out): Used **1×1 convolutions** (3→32→64→64) + MaxPool + global avg pool. Designed to be color-only, but the extreme architectural constraint (no spatial receptive field at all) made it unrealistic — real models don't have this limitation, they *choose* to ignore shape.
- **ColorBiasedNet** (`precog_fin_33.ipynb`): Similar 1×1 conv architecture (3→16→32) with MaxPool(4,4). Same problem as LazyNet — interesting as an extreme demo but proves nothing about real spurious correlations since the model *can't* learn shapes even if it wanted to.

**Why NanoNet won:** It uses standard **3×3 convolutions** (can learn both shape and color), achieves 97% on regular MNIST (proving sufficient capacity), yet still takes the color shortcut when trained on biased data. This makes the bias demonstration **realistic** — the model has the ability to learn shapes but *prefers* color. The 16-dim penultimate layer is also small enough for complete SAE analysis in Task 6.

### Task 4: WeightedRandomSampler / Rebel Mixup / Mild Jitter → Strong ColorJitter + Channel Permutation

**Failed Method 1 — WeightedRandomSampler:** The idea was to upweight the 5% counter-examples (images where digit color doesn't match the expected mapping) by 20×, forcing the model to see more conflicting color-digit pairs. **Why it failed:** The `ColoredMNIST` dataset is *stochastic* — it regenerates colors randomly on every `__getitem__` call. So the "rebel indices" identified at one point get different (mostly biased) colors when reloaded by the sampler. The 20× upweighting effectively did nothing.

**Failed Method 2 — Rebel Mixup:** Every training image was blended 50/50 with a counter-example. **Why it failed:** The blended images were muddy and unnatural — averaging a Red 0 with a Green 0 creates a brownish blob that hurts shape learning rather than helping it. The mixed signal confused the model rather than teaching it to ignore color.

**Failed Method 3 — Mild ColorJitter (hue=0.15):** Applied `ColorJitter(hue=0.15)` which rotates hue by ±54°. **Why it failed:** ±54° is only 30% of the color wheel. The model could still distinguish warm vs cool hues and maintain enough color discrimination to cheat. Accuracy plateaued at ~82%.

**The final fix** uses `hue=0.5` (±180° = full color wheel, any color can become any other color) plus random channel permutation (all 6 RGB orderings). Together these make color a purely random, uninformative signal — forcing the model to rely on shape.

### Task 5: Simple PGD → Adam-PGD with Multi-Restart

**Earlier approach (`precog_fin_55.ipynb`):** A basic PGD attack with SGD optimizer, fixed α=0.01, 100 steps, single run, and a naive minimum-ε finder that linearly sweeps in 0.01 increments.

**Why it was abandoned:** Simple sign-gradient PGD is noisy for targeted attacks with tight ε budgets (0.05). It frequently gets stuck in local minima and fails to find successful adversarial examples even when they exist. The final Adam-PGD uses adaptive learning rates, CE→CW loss blending (strong signal far from boundary, precision near it), cosine LR decay, and 20 random restarts to reliably find adversarial examples at minimal ε.

### Task 3: Total Energy BG% → Density-Based BG%

**Earlier approach (`precog_fin_55.ipynb`):** Measured background focus as `sum(heatmap × bg_mask) / sum(heatmap)` — total energy on background pixels divided by total energy everywhere.

**Why it was abandoned:** Background has ~6× more pixels than the digit foreground (~85% vs ~15%). Even a perfectly uniform heatmap would show ~85% energy on background. This makes the metric misleading — a model focusing equally everywhere would appear "86% background focused." The final density-based metric (`mean(heatmap on bg) / (mean(bg) + mean(fg))`) normalizes for pixel count, making 50% the neutral threshold.

### Evolution Timeline

```
precog_fin.ipynb      → Flat coloring, ResNet/LazyNet experiments, Tasks 1-3 only
precog_fin_22.ipynb   → Switched to textured backgrounds
precog_fin_33.ipynb   → Added ColorBiasedNet, full causal ablation proofs
precog_fin_55.ipynb   → Added Task 4 (ColorJitter+ChannPerm), Task 5 (simple PGD)
precog_fin_f5.ipynb   → Upgraded Task 5 to Adam-PGD, added Task 6 (SAE)
precog_fin_f6.ipynb   → Added Task 2 (activation maximization / dreams)
precog_comp.ipynb     → Final polished version with density-based metrics + all tasks
```

---

## Acknowledgments

This project was completed as part of the PreCog research group assessment. The task design ("The Lazy Artist") provides a pedagogically rich framework for understanding spurious correlations, model interpretability, adversarial robustness, and mechanistic interpretability through sparse autoencoders.
