# Stereo Block Matching

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Dense stereo matching — compute a **disparity map** $D(u,v)$ giving the horizontal pixel offset between corresponding points in a rectified stereo pair. From the disparity map, per-pixel depth $Z = fB/d$ is recovered without active illumination. Used whenever a calibrated, rectified stereo rig is available and a dense point cloud is needed.

---

## A2. High-level overview

**Prerequisites:** images have been undistorted ([[lens-distortion-and-removal]]) and rectified ([[stereo-rectification]]) so that epipolar lines are horizontal rows.

**Steps (no math):**
1. Formalise: the right image is (approximately) a horizontally shifted version of the left.
2. For every pixel $(u,v)$ in the left image, scan disparity candidates $d = 0 \ldots d_{\max}$ along the same row in the right image.
3. For each candidate, compute a block-level matching cost over a $w \times w$ window.
4. Choose the disparity $d^*$ that minimises (SSD/SAD) or maximises (NCC) the cost.
5. Store $d^*$ in the disparity map; handle border pixels by setting disparity to 0.
6. Convert disparity to depth; scale map for display.

**Key definitions:**

| Symbol | Meaning |
|---|---|
| $I_L(u,v),\, I_R(u,v)$ | Left and right rectified images |
| $d$ | Disparity — horizontal pixel shift; $d = u_L - u_R$ |
| $d_{\max}$ | Maximum disparity (set empirically) |
| $C(u,v,d)$ | Matching cost at pixel $(u,v)$ for disparity $d$ |
| $w$ | Block (window) width in pixels |
| $r = \lfloor w/2 \rfloor$ | Half-window radius (boundary margin) |
| $B$ | Stereo baseline (mm, distance between optical centres) |
| $f$ | Focal length (same units as $B$ after unit conversion) |
| $Z$ | Scene depth |
| $D_{\text{vis}}$ | Display-scaled disparity map (0–255) |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Simple to implement; easily parallelised on GPU/SIMD.
- Produces a dense disparity map — every pixel gets a depth estimate.
- No training data required; interpretable cost functions.

**Shortcomings / limitations**
- **Constant-disparity assumption within block** — fails near depth discontinuities.
- **Edge fattening** — foreground object boundaries bleed into background because the block straddles the edge.
- **Low-texture / homogeneous regions** (plain walls, sky) — many disparities give equal cost; result is arbitrary.
- **Occlusions** — pixels visible in only one image have no valid match; disparities are incorrect.
- **Local approach only** — ignores global image context; no smoothness guarantee between neighbouring pixels.
- **Window size trade-off** — small $w$: noisy disparities, fine detail preserved; large $w$: smoother map but edge-fattening increases.
- Computationally expensive without hardware acceleration ($O(W \cdot H \cdot d_{\max} \cdot w^2)$).

### Likely exam questions

**Q:** Write the stereo matching formalisation and cost-minimisation expression for disparity.
**A:**
$$I_L(u,v) \approx I_R(u - d(u,v),\; v)$$
$$d(u,v) = \arg\min_d\; C(u,v,d)$$
The right image is approximately the left image shifted left by $d$ pixels (canonical geometry, same row $v$).

---

**Q:** Describe the block-size window trade-off. What goes wrong at each extreme?
**A:** Small $w$ → each window contains little texture; cost function is noisy and disparities are unreliable (high variance). Large $w$ → the window spans multiple depths at boundaries; the block-matching assumption (constant disparity within window) is violated, causing **edge fattening** — the disparity of a foreground object is assigned to adjacent background pixels.

---

**Q:** What is BorderIgnore and why does it set disparity to 0?
**A:** BorderIgnore is the border-handling policy: if the $w \times w$ reference block or any candidate search block falls outside the image, no valid cost can be computed. The algorithm assigns $d = 0$ (no disparity / unknown depth) rather than fabricating a match. The margin that must be excluded is $r = \lfloor w/2 \rfloor$ pixels from each edge; for the right image the additional disparity shift must also fit: $u - d - r \geq 0$.

---

**Q:** How do you find `disparityMax` in practice?
**A:** Open both rectified images in ImageJ. Locate the closest feature to the camera (highest expected disparity — smallest $Z$). Since epipolar lines are aligned after rectification, $v$-coordinates should match. Measure $u_L$ and $u_R$ for that feature; $d_{\max} = u_L - u_R$.

---

**Q:** Given the 3×3 patches below, compute SSD($L$, $R_1$) and SSD($L$, $R_2$) and state which is the better match.

$$L = \begin{bmatrix}10&10&10\\10&20&10\\10&10&10\end{bmatrix},\quad R_1 = \begin{bmatrix}10&10&10\\10&18&10\\10&10&10\end{bmatrix},\quad R_2 = \begin{bmatrix}12&12&12\\12&12&12\\12&12&12\end{bmatrix}$$

**A:**
$$\text{SSD}(L,R_1) = (20-18)^2 = 4$$
$$\text{SSD}(L,R_2) = 8\times(10-12)^2 + (20-12)^2 = 32 + 64 = 96$$
$R_1$ is the better match (lower SSD). SSD is minimised for best match.

---

**Q:** Why does NCC use $C_{\text{NCC}} = 1 - \rho$ as its minimised cost rather than maximising $\rho$ directly?
**A:** The block-matching pseudocode searches for the **minimum** cost (uniform loop structure). Converting NCC to a cost $C_{\text{NCC}} = 1 - \rho \in [0, 2]$ allows the same `bestScore < score` logic to select the most correlated patch without changing the algorithm skeleton. See [[patch-similarity-measures]] for full cost-function forms.

---

## A4. Directions of reasoning

### Forward (inputs → disparity map)

**Given:** rectified pair $I_L$, $I_R$; block size $w$; $d_{\max}$.
**Compute:** for each $(u,v)$, evaluate $C(u,v,d)$ for $d = 0 \ldots d_{\max}$; select $d^* = \arg\min_d C$.
**Output:** disparity map $D$, then depth $Z = fB/d^*$, then visualisation $D_{\text{vis}}$.

### Reverse / inferential

**Given:** a known disparity value $d$ (or a depth $Z$) and camera parameters $f$, $B$.  
**Recover:** depth via $Z = fB/d$; or recover expected pixel location in right image: $u_R = u_L - d$ (same row $v$).  
**Diagnose failure:** if $D_{\text{vis}}$ is uniformly zero at a region → BorderIgnore fired (block outside image, or $u - d - r < 0$). If disparity is noisy in a flat-colour region → low texture; block matching unreliable.

**Given:** $Z$ and $f$, $B$.  
**Recover disparity:** $d = fB/Z$ (rearrangement of depth formula). Useful to estimate expected $d_{\max}$ from the minimum scene depth.

---

## A5. Standard implementation

### a. Setup

- **Input:** rectified left image $I_L$, rectified right image $I_R$ (both $H \times W$ pixels).
- **Parameters:** block width $w$ (odd integer); $d_{\max}$ (found via ImageJ on closest feature); cost function choice (SSD / SAD / NCC).
- **Output:** disparity map $D$ ($H \times W$, integer values $0 \ldots d_{\max}$); optionally scaled visualisation $D_{\text{vis}}$ and depth map $Z$.
- **Notation:** pixel coordinates $(u, v)$ (column, row); half-radius $r = \lfloor w/2 \rfloor$; window offsets $(i,j)$ ranging over the $w \times w$ neighbourhood.
- **Assumption:** canonical epipolar geometry — corresponding points lie on the same row $v$; disparity is always non-negative ($u_L \geq u_R$, objects are in front of cameras).

> [!note]
> The lecture slides use $(x,y)$ for pixel coordinates with $u,v$ for block offsets. The W11T practical uses $(u,v)$ for pixel coordinates and $(i,j)$ for window offsets. This file adopts the W11T convention (pixel = $(u,v)$, window offset = $(i,j)$) for consistency with the spec notation table.

### b. Steps

**Step 1 — Initialise disparity map.**
$$D \leftarrow \mathbf{0}_{H \times W}$$
All pixels start with zero disparity (no match / border default).

**Step 2 — Outer loop over all pixels.**
For each row $v = 1 \ldots H$, for each column $u = 1 \ldots W$:

**Step 3 — Initialise best score.**
$$\text{bestScore} \leftarrow C(u,v,0); \qquad \text{bestDisparity} \leftarrow 0$$
Seed with $d = 0$ (zero shift).

**Step 4 — Inner loop over disparity candidates.**
For $d = 1$ to $d_{\max}$:

**Step 4a — Boundary check (BorderIgnore).**
If the left block or the shifted right block falls outside the image, skip this $d$ (and set $D[v,u] = 0$ if no valid $d$ was found). Validity condition for right block:
$$u - d - r \geq 0 \quad \text{and} \quad u - d + r < W_{\text{image}}$$
Also require $r \leq u \leq W-r-1$ (left block fits) and $r \leq v \leq H-r-1$ (vertical margin).

**Step 4b — Compute block cost.**
Using SSD (minimise):
$$C_{\text{SSD}}(u,v,d) = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \bigl(I_L(u+i,\,v+j) - I_R(u+i-d,\,v+j)\bigr)^2$$

Using SAD (minimise):
$$C_{\text{SAD}}(u,v,d) = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \bigl|I_L(u+i,\,v+j) - I_R(u+i-d,\,v+j)\bigr|$$

Using NCC (convert to cost, minimise):
$$\rho(u,v,d) = \frac{\displaystyle\sum_{i,j}(L_{ij}-\bar{L})(R_{ij}-\bar{R}_d)}{\sqrt{\displaystyle\sum_{i,j}(L_{ij}-\bar{L})^2}\;\sqrt{\displaystyle\sum_{i,j}(R_{ij}-\bar{R}_d)^2}}, \qquad C_{\text{NCC}} = 1-\rho$$

See [[patch-similarity-measures]] for derivations and distributional assumptions (SSD ↔ Gaussian noise; SAD ↔ Laplace noise; NCC ↔ illumination-invariant).

**Step 4c — Update best.**
```
if score < bestScore:
    bestScore = score
    bestDisparity = d
```
(For NCC, cost $= 1 - \rho$ so same minimisation logic applies.)

**Step 5 — Store.**
$$D[v,u] \leftarrow \text{bestDisparity}$$

**Step 6 — Display scaling.**
$$D_{\text{vis}} = 255 \times \frac{D - D_{\min}}{D_{\max} - D_{\min}}$$
Raw integer disparity values are not displayable as a meaningful greyscale image without this step.

**Step 7 — Depth conversion.**
$$Z = \frac{f \cdot B}{d}$$
where $f$ and $B$ must be in consistent units (both mm, or $f$ converted to dimensionless pixels via $f = f_{\text{mm}}/s$). Depth is unstable / undefined when $d = 0$. See [[depth-from-disparity]] for full derivation and unit-conversion procedure.

> [!example]
> **Worked SSD example (3×3 patch, $w=3$, $r=1$):**
>
> $L$ has centre pixel 20, all others 10. $R_1$ has centre pixel 18, all others 10. $R_2$ is all 12.
>
> $\text{SSD}(L,R_1) = (20-18)^2 = 4$ (only centre differs)
>
> $\text{SSD}(L,R_2) = 8\times(10-12)^2 + (20-12)^2 = 32 + 64 = 96$
>
> $\Rightarrow$ $R_1$ is the best match. Disparity assigned = disparity of $R_1$ relative to $L$.

---

## A6. Variations

### Global approach — Total Variation with Potts model

**Change:** augment the per-pixel block cost with a **smoothness term** that penalises disparity changes between adjacent pixels.

$$E(u,v,d) = C_{\text{block}}(u,v,d) + \alpha \cdot \text{Smooth}(d,\, d_{\text{prev}})$$

$$\text{Smooth}(d, d_{\text{prev}}) = \begin{cases} 0 & d = d_{\text{prev}} \\ p & d \neq d_{\text{prev}} \end{cases} \quad \text{(Potts model)}$$

- $\alpha$ — smoothing weight (trade-off between data fidelity and regularity).
- $p$ — constant penalty for any disparity discontinuity.
- $d_{\text{prev}}$ — disparity of a neighbouring pixel (scanline predecessor or spatial neighbour).

**When used:** scenes with smooth, connected surfaces; reduces noise and fills low-texture regions better than purely local block matching.

**Limitation:** Potts model cannot represent gradual disparity ramps; transitions are always binary (same or different). Full global optimisation (e.g. graph cuts, belief propagation) is NP-hard in general; approximations are used in practice.

---

### ELAS — Efficient Large Scale Stereo

**Type:** semi-global / guided-stereo hybrid. Sparse support points (robustly matched features) constrain a triangulated planar surface model, which then guides the full dense disparity search.

**Key idea:** a sparse disparity map is computed first (via feature matching); a triangulated mesh interpolates a planar-surface prior over the image; the dense block-matching search for each pixel is restricted to a narrow band around the prior disparity.

**When used:** large-scale outdoor scenes (KITTI autonomous driving dataset); balance between computational cost and accuracy better than pure global methods.

---

### Deep stereo methods

**Change:** replace hand-crafted cost functions (SSD/SAD/NCC) and local/global optimisation with a learned end-to-end pipeline.

**Shared structure (classical and deep):** build a cost volume → select or regress disparity.

| Classical | Deep (e.g. CREStereo) |
|---|---|
| SSD / SAD / NCC | Learned feature similarity (AGCL) |
| Local or semi-global optimisation | Learned cost-volume regularisation (RUM) |
| Explicit Potts/TV smoothness | Implicit learned priors from data |
| No training data needed | Requires large labelled stereo datasets |
| Examples: block matching, SGM | Examples: PSMNet, RAFT-Stereo, CREStereo |

**CREStereo specifics:** Cascaded REcurrent Stereo matching network (CVPR 2022, Megvii). Coarse-to-fine recurrent refinement at scales $1/16 \to 1/8 \to 1/4$. AGCL (Adaptive Group Correlation Layer) combines 1D (horizontal) and 2D (horizontal + small vertical) local search with a deformable window — tolerates imperfect rectification.

See [[deep-learning-for-stereo-and-calibration]] for full deep-stereo coverage.

---

**Related topics:** [[epipolar-geometry]] · [[stereo-rectification]] · [[depth-from-disparity]] · [[stereo-depth-accuracy]] · [[patch-similarity-measures]] · [[deep-learning-for-stereo-and-calibration]]
