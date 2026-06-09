# Feature Descriptors

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-Level Overview]]
- [[#A3. Strengths, Shortcomings & Limitations]]
  - [[#Likely Exam Questions]]
- [[#A4. Directions of Reasoning]]
- [[#A5. Standard Implementation]]
  - [[#a. Setup]]
  - [[#b. Steps (NCC — the standard similarity-based approach)]]
- [[#A6. Variations]]
  - [[#Raw Intensity Patch (Baseline)]]
  - [[#Local Histogram Descriptors]]
  - [[#HOG — Histogram of Oriented Gradients]]
  - [[#SIFT — Scale-Invariant Feature Transform]]
  - [[#Learned Descriptors]]

---

## A1. Purpose

A **feature descriptor** converts the local image patch around a detected keypoint into a compact $D$-dimensional vector $\mathbf{x} \in \mathbb{R}^D$ that can be compared across images. The goal is to enable **matching**: given keypoints in two images, find pairs whose descriptors are most similar. A good descriptor must be distinctive enough to identify correct matches and robust enough that the same physical point produces a similar vector under photometric and geometric variation. See [[feature-points-and-autocorrelation]] for detection, [[harris-corner-detector]] for a concrete detector, and [[feature-matching]] for how descriptors are compared at scale.

---

## A2. High-Level Overview

The local feature pipeline has three stages:

1. **Detection** — find repeatable interest points (keypoints) in each image.
2. **Description** — extract a $D$-dimensional descriptor $\mathbf{x}$ from the patch centred on each keypoint.
3. **Matching** — compare descriptor pairs; declare a match when $d(\mathbf{x}_{1,i},\, \mathbf{x}_{2,j}) < T$ for threshold $T$.

Key definitions:

| Term | Definition |
|---|---|
| $\mathbf{x}_{1,i} = [x_{1,i}^1, \ldots, x_{1,i}^D]$ | $D$-dim descriptor for keypoint $i$ in image $I_1$ |
| $\mathbf{x}_{2,j} = [x_{2,j}^1, \ldots, x_{2,j}^D]$ | $D$-dim descriptor for keypoint $j$ in image $I_2$ |
| $d(\mathbf{x}_1, \mathbf{x}_2)$ | Descriptor distance (e.g. L2 norm or 1−NCC) |
| $T$ | Matching threshold |
| $\boldsymbol{f},\, \boldsymbol{g}$ | Flattened patch intensity vectors from images $I(x,y)$, $J(x,y)$ |
| $\bar{f},\, \bar{g}$ | Mean intensities of patches $\boldsymbol{f}$ and $\boldsymbol{g}$ |
| $\Theta$ | Dominant gradient orientation at a keypoint (used by SIFT) |
| Affine intensity change | $g_i = a \cdot f_i + b$ (gain $a$, bias $b$); NCC is invariant to this |

**Ideal descriptor properties:** robust, distinctive, compact, efficient.

---

## A3. Strengths, Shortcomings & Limitations

**Strengths (of the descriptor approach generally)**
- Reduces matching to a simple vector-distance comparison.
- Localised to a patch — tolerant of background clutter beyond the patch.
- Histogram-based descriptors (HOG/SIFT) are robust to small spatial deformations because they pool gradient evidence over subregions.

**Shortcomings**
- Raw intensity patches: small geometric deformations cause large changes in score, since pixel alignment is assumed exactly.
- Local histogram descriptors: lack robustness to brightness change, noise, rotation, and scaling.
- NCC is undefined (division by zero) when a patch is uniform ($\sigma = 0$).
- SSD and SAD are sensitive to affine intensity changes (bias/gain).
- SIFT is computationally expensive and memory-heavy; originally patented (Lowe).
- Learned descriptors require training data and GPU; may not generalise to unseen domains.

**When to use which**

| Use case | Recommended |
|---|---|
| Fastest baseline / prototyping | SSD / SAD on raw patch |
| Illumination-robust matching | NCC |
| Wide-baseline, scale+rotation changes | SIFT |
| Edge / embedded / real-time hardware | ORB, SURF |
| Accuracy-first, GPU available | SuperPoint / learned |

---

### Likely Exam Questions

**Q:** What is the dimensionality of a raw intensity descriptor from an $11\times11$ patch? From a $15\times15$ patch?
**A:** $11\times11 = 121$-dim; $15\times15 = 225$-dim. Each pixel intensity is one element of the vector.

**Q:** Why is SSD not robust to a uniform brightness increase across a patch?
**A:** If $g_i = f_i + b$ for all $i$, then $\mathrm{SSD} = \sum_i b^2 = N b^2 \neq 0$, even though the patches are structurally identical. A global bias $b$ inflates the SSD score.

**Q:** What is the key weakness of raw intensity patch descriptors (both SSD and histogram-based) under small geometric deformations?
**A:** Small deformations break pixel-level alignment: the same physical point maps to a shifted/rotated location, so corresponding pixels no longer match up, causing a large score change even though the scene structure is similar.

**Q:** Describe the SIFT descriptor construction: how is the 128-dimensional vector formed?
**A:** (1) Extract a $16\times16$ patch centred on the keypoint, aligned to the dominant orientation $\Theta$ and weighted by a Gaussian ($\sigma = $ half window width). (2) Divide into a $4\times4$ grid of subregions. (3) Compute an 8-bin gradient orientation histogram in each subregion. (4) Concatenate: $4\times4\times8 = 128$ dimensions.

**Q:** What makes SIFT rotation-invariant, and what makes it scale-invariant?
**A:** Rotation invariance: the patch is rotated to align with the dominant gradient orientation $\Theta$ before computing the histogram, so the descriptor is in a canonical frame. Scale invariance: keypoints are detected with a DoG (Difference of Gaussians) multi-scale blob detector, so the same feature is found at any zoom level.

**Q:** NCC returns $-0.125$ for a pair of patches. What does this indicate, and when is NCC undefined?
**A:** NCC $\in [-1, 1]$; a value near 0 (here $-0.125$) means near-zero correlation — the patches are not similar. NCC is undefined when either patch is uniform (all pixels equal), making $\sqrt{\sum_i(f_i-\bar{f})^2} = 0$, i.e. division by zero.

---

## A4. Directions of Reasoning

### Forward — descriptor extraction to matching decision

**Given:** two images $I_1, I_2$, detected keypoints $F_1, F_2$, a descriptor type.
**Asked:** which keypoints correspond?
**Process:**
1. For each keypoint, extract patch → compute descriptor vector $\mathbf{x}$.
2. For each candidate pair $(\mathbf{x}_{1,i}, \mathbf{x}_{2,j})$, compute $d(\mathbf{x}_{1,i}, \mathbf{x}_{2,j})$.
3. Declare a match if $d < T$; pass matched pairs to [[feature-matching]] for further filtering.

### Reverse — score back to descriptor choice

**Given:** a matching score (e.g. SSD $= 36$) and knowledge that a constant bias $b$ separates two patches.
**Asked:** should this pair be considered a match? Which measure would handle the bias?
**Inference:** SSD absorbs $b^2$ per pixel, so even structurally identical patches yield SSD $= Nb^2 \neq 0$. Use NCC instead: it subtracts the mean, making it invariant to additive bias.

### Reverse — from SIFT dimension back to structure

**Given:** a 128-dimensional SIFT descriptor.
**Asked:** recover the grid structure.
**Inference:** $128 = 16 \times 8$, where $16 = 4 \times 4$ subregions and $8$ = orientation bins. The patch was $16\times16$ pixels, divided into $4\times4$ blocks of $4\times4$ pixels each.

---

## A5. Standard Implementation

### a. Setup

- **Input:** image $I(x,y)$; set of keypoints with pixel locations $(u,v)$.
- **Output:** descriptor vector $\mathbf{x} \in \mathbb{R}^D$ per keypoint.
- **Parameters:** patch size (e.g. $11\times11$, $15\times15$, $16\times16$); descriptor type; (for SIFT) number of subregions, orientation bins.
- **Notation:** $\boldsymbol{f} = (f_0, \ldots, f_{D-1})^\top$ is the intensity vector of the reference patch; $\boldsymbol{g}$ similarly for the candidate patch.

### b. Steps (NCC — the standard similarity-based approach)

1. **Extract patch** of size $W\times W$ centred on each keypoint; flatten to vector $\boldsymbol{f} \in \mathbb{R}^{W^2}$ (e.g. $15\times15 \to 225$-dim).
2. **Compute mean** $\bar{f} = \frac{1}{W^2}\sum_i f_i$.
3. **Centre the patch:** $A_i = f_i - \bar{f}$.
4. **Compute standard deviation factor:** $C = \sum_i A_i^2 = \sum_i (f_i - \bar{f})^2$.
5. **Store normalised NCC component** $\hat{f}_i = A_i / \sqrt{C}$ per descriptor (efficient precomputation — avoids recomputing at match time).
6. **At match time:** given two stored NCC components, $\mathrm{NCC}(\boldsymbol{f}, \boldsymbol{g}) = \sum_i \hat{f}_i \cdot \hat{g}_i$ — a simple dot product.
7. **Threshold:** declare a match if $\mathrm{NCC} > T$ (or equivalently if the distance $1 - \mathrm{NCC} < T'$).

> [!note] Degenerate patch check
> Before step 5, verify $C > 0$. If $C = 0$ (uniform patch, $\sigma = 0$), NCC is undefined. Flag the descriptor as degenerate and skip it.

For full formulas and similarity measures see [[patch-similarity-measures]].

---

## A6. Variations

### Raw Intensity Patch (Baseline)

- **Descriptor:** flatten $W\times W$ pixel values → $W^2$-dim vector (e.g. $11\times11 \to 121$-dim, $15\times15 \to 225$-dim).
- **Similarity:** compare with SSD, SAD, CC, or NCC (see [[patch-similarity-measures]] for formulas — not re-derived here).
- **Weakness:** assumes pixel-perfect alignment. A small geometric deformation (shift, rotation, scale change) breaks correspondence between individual pixels, causing a large change in SSD/SAD even when the patches look similar.
- **Example (from slides):**

$$f = \begin{pmatrix}10&10&10\\10&10&10\\10&10&10\end{pmatrix},\quad g = \begin{pmatrix}12&12&12\\12&12&12\\12&12&12\end{pmatrix} \implies \mathrm{SSD} = 9\times4 = 36$$

Two structurally different patch pairs can give identical SSD = 36.

> [!warning] SSD outlier sensitivity
> A single outlier pixel (e.g. $f_\text{centre} = 20$ vs $g_\text{centre} = 10$) gives $\mathrm{SSD} = 100$ but $\mathrm{SAD} = 10$. SSD squares the error, making it disproportionately sensitive to single bad pixels.

---

### Local Histogram Descriptors

- **Descriptor:** compute a histogram $H(q)$ of pixel intensities (or gradient magnitudes) over a local patch; use $H$ as the descriptor vector.
- **Advantage over raw patch:** reduces sensitivity to small spatial shifts — pooling over a region smooths out exact pixel positions.
- **Disadvantages:**
  - Still affected by small deformations.
  - Not robust to brightness changes (unless gradients are used), noise, rotation, or scaling.
  - No spatial structure within the histogram (loses where gradients are, not just their distribution).
- **Note:** local histograms are the conceptual precursor to HOG/SIFT.

---

### HOG — Histogram of Oriented Gradients

- **Concept:** compute image gradients (magnitude + orientation), then build local histograms of gradient orientations; pooled over spatial cells.
- **Role here:** HOG is the core computational unit inside SIFT. Each of SIFT's $4\times4$ subregions produces one HOG descriptor (an 8-bin orientation histogram weighted by gradient magnitude).
- **Standalone use:** HOG over a full image window is a widely used descriptor for object detection (e.g. pedestrian detection). SIFT applies the same idea locally around a keypoint.
- **Link:** [[feature-points-and-autocorrelation]] — gradient computation is shared with Harris corner detection.

---

### SIFT — Scale-Invariant Feature Transform

SIFT is the canonical local descriptor. It chains DoG-based scale detection with a HOG-based descriptor aligned to a dominant orientation.

**Scale detection (DoG)**
- Build a Gaussian image pyramid; compute Difference of Gaussians (DoG) at each scale.
- Detect keypoints as local extrema in DoG scale-space. Same physical point is detected regardless of image zoom → scale invariance.

**Descriptor extraction (9 steps)**

1. **Detect keypoints** using multi-scale DoG blob detector → each keypoint has a scale $s$.
2. **Compute gradient orientation histogram** over a neighbourhood of the keypoint at scale $s$.
3. **Identify dominant orientation** $\Theta$ = peak of the orientation histogram, $\Theta \in [0, 2\pi)$.
4. **Rotate** the coordinate frame to align with $\Theta$ → canonical orientation (source of rotation invariance).
5. **Extract $16\times16$ patch** centred on keypoint, sampled at keypoint scale $s$.
6. **Weight patch by Gaussian** with $\sigma = $ half the window width → reduces influence of pixels far from centre; adds robustness to exact keypoint localisation.
7. **Divide into $4\times4$ subregions** (16 cells of $4\times4$ pixels each).
8. **Compute 8-bin gradient orientation histogram** per subregion, accumulating gradient magnitudes into bins.
9. **Concatenate** all 16 histograms → $4\times4\times8 = \mathbf{128}$-dimensional descriptor vector.

> [!note] Slide notation
> Some slides write the Gaussian weighting step as: $\sigma = \frac{1}{2} \times \text{window width}$. The $16\times16$ window has $\sigma = 8$ pixels.

**Properties**

| Property | Detail |
|---|---|
| Dimensionality | 128 |
| Scale invariant | Yes — DoG detection at multiple scales |
| Rotation invariant | Yes — normalised to dominant orientation $\Theta$ |
| Illumination robust | Yes — uses gradient orientations, not raw intensities; handles up to day/night changes |
| Viewpoint tolerance | Up to ~60° out-of-plane rotation |
| Speed | Real-time capable |
| Patent status | Originally patented by David Lowe (patent now expired) |
| Citation count | 88,000+ (as of lecture slides) |

**Links:** [[harris-corner-detector]] (gradient computation), [[feature-matching]] (how SIFT descriptors are matched in practice), [[patch-similarity-measures]] (distance measures used at match time).

**Alternatives with similar motivation**

| Method | Key change vs SIFT |
|---|---|
| **SURF** (Speeded-Up Robust Features) | Approximates DoG with box filters (integral images) — faster; less accurate |
| **ORB** (Oriented FAST + Rotated BRIEF) | Binary descriptor; very fast; suitable for embedded/real-time with no patent issues |

---

### Learned Descriptors

- **Approach:** replace handcrafted gradient histograms with CNN-learned feature maps.
- **MagicPoint / SuperPoint (DeTone et al.):** end-to-end network trained to detect keypoints (MagicPoint) and compute descriptors (SuperPoint) jointly.
- **Advantages:** can learn task-specific invariances; potentially more distinctive than SIFT on complex textures.
- **Disadvantages:** require GPU at inference time; training requires large labelled datasets; may not generalise beyond training domain; classical methods (SIFT, ORB) remain competitive on edge/real-time hardware.
- **Status (as of lecture):** active research area; classical methods still the default for embedded deployment.

> [!warning] Beyond lecture scope
> The lecture names MagicPoint and SuperPoint as examples of learned descriptors but does not detail their architectures. The comparison below uses only information from the slides.

---

*Related topics:* [[feature-points-and-autocorrelation]] · [[harris-corner-detector]] · [[patch-similarity-measures]] · [[feature-matching]]
