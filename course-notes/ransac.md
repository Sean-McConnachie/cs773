# RANSAC (RANdom SAmple Consensus)

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

RANSAC is a robust model-fitting algorithm used when a set of point correspondences (or data points) contains a **high fraction of outliers**, making ordinary least-squares fail catastrophically. In the image-stitching pipeline it replaces a single global least-squares homography fit after putative feature matching, instead iteratively proposing and testing candidate homographies to find the one supported by the largest set of geometrically consistent matches. It is applicable whenever you can (a) draw a minimal sample that determines a model and (b) evaluate every other point against that model.

---

## A2. High-level overview

**Steps (no math):**

1. Randomly select the **minimal seed group** of matches needed to determine the model.
2. **Fit** the model to the seed group exactly (DLT for homography).
3. **Count inliers**: evaluate every match in the full set; a match is an inlier if its geometric error is below the threshold.
4. **Repeat** steps 1–3 for a fixed number of iterations; **keep** the candidate with the most inliers.
5. **(Post-loop) Refit** the model via least-squares / DLT on **all** inliers of the best candidate.

**Key definitions:**

| Term | Definition |
|---|---|
| **Inlier** | A point whose distance to the hypothesised model is $\leq$ `inlierMappingThreshold` |
| **Outlier** | A point that exceeds the threshold |
| **Minimal seed group** | The smallest set of points that uniquely determines the model (4 point pairs for a homography) |
| **Minimal seed size for homography** | 4 correspondences — a homography has 8 DOF; each correspondence contributes 2 equations |
| `inlierMappingThreshold` | Distance threshold in pixels separating inliers from outliers |
| `numberOfRandomDraws` | Total number of RANSAC loop iterations |
| **DLT** | Direct Linear Transform — used to solve for $\mathbf{H}$ from the seed group (see [[homography-and-dlt]]) |
| **Collinearity check** | Reject a seed group if any three points are collinear (degenerate configuration) |

> [!note] Notation mapping
> The slides use $m, b$ for slope/intercept in the line-fitting illustration and $k, b$ in one variant. This file normalises to $m$ (slope) and $b$ (intercept) throughout. The matrix is $\mathbf{H}$ (bold upper-case) following spec conventions; slides write it as $H$.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Tolerates a large fraction of outliers (up to ~50%) that would destroy a global least-squares fit.
- Conceptually simple; easy to implement with any parameterisable model.
- Post-loop DLT refinement on all inliers gives a high-quality final estimate.

**Shortcomings / Limitations**
- Requires outlier fraction $< 50\%$; the guarantee breaks down at or above 50%.
- Number of iterations must be set in advance (or adaptively); too few iterations risk missing the true model.
- The threshold `inlierMappingThreshold` (1–2 px) must be chosen appropriately — too tight rejects valid inliers; too loose admits outliers.
- Seed points must be **distinct** and **no triple collinear**; degenerate draws must be discarded and resampled.
- Non-deterministic: different runs can (rarely) give different results.

---

### Likely exam questions

**Q: What is RANSAC's core guarantee, and what condition must hold for it to apply?**
**A:** Given enough random draws, RANSAC finds the correct model if the fraction of outliers is **less than 50%**. The guarantee relies on the probability that at least one draw consists entirely of inliers being high after many iterations.

---

**Q: List the four steps of the RANSAC loop (in order) for fitting a homography.**
**A:**
1. Randomly select 4 distinct, non-collinear point-pair matches (minimal seed group).
2. Compute $\mathbf{H}$ from the 4 pairs using DLT.
3. Count inliers over **all** matches: a match is an inlier if the mapping distance $< $ `inlierMappingThreshold`.
4. Repeat for `numberOfRandomDraws` iterations; keep the $\mathbf{H}$ with the most inliers.
   *(Post-loop: refit DLT on all inliers of the best $\mathbf{H}$.)*

---

**Q: In RANSAC for line fitting, the 10 points below are given. A sample selects $(77, 3282)$ and $(91, 3915)$. What line does this determine, and how many inliers result with threshold $d \leq 3$ px?**

> [!example] Numeric example (W7T, slides p. 22)
> Points: $(57,2025),(51,2233),(60,2442),(63,2653),(67,2864),(72,3073),(77,3282),(82,3494),(90,3705),(91,3915)$
>
> **Step 1 — Fit line through the two sampled points:**
> $$3282 = 77m + b, \quad 3915 = 91m + b$$
> Subtract: $633 = 14m \Rightarrow m = 45$. Then $b = 3282 - 77 \times 45 = -183$.
> Line: $y = 45x - 183$.
>
> **Step 2 — Count inliers** using $d = |mx - y + b| / \sqrt{m^2 + 1} = |45x - y - 183| / \sqrt{2026}$:
>
> | Point | $|45x-y-183|$ | $d$ | Inlier? |
> |---|---|---|---|
> $(57,2025)$ | $|2565-2025-183|=357$ | $357/\sqrt{2026}\approx 7.9$ | No |
> $(51,2233)$ | $|2295-2233-183|=121$ | $\approx 2.7$ | **Yes** |
> $(60,2442)$ | $|2700-2442-183|=75$ | $\approx 1.7$ | **Yes** |
> $(63,2653)$ | $|2835-2653-183|=1$ | $\approx 0.02$ | **Yes** |
> $(67,2864)$ | $|3015-2864-183|=32$ | $\approx 0.7$ | **Yes** |
> $(72,3073)$ | $|3240-3073-183|=16$ | $\approx 0.36$ | **Yes** |
> $(77,3282)$ | $|3465-3282-183|=0$ | $0$ | **Yes** |
> $(82,3494)$ | $|3690-3494-183|=13$ | $\approx 0.29$ | **Yes** |
> $(90,3705)$ | $|4050-3705-183|=162$ | $\approx 3.6$ | No |
> $(91,3915)$ | $|4095-3915-183|=3$ | $\approx 0.07$ | **Yes** |
>
> **Result: 9 inliers** ($(57,2025)$ and $(90,3705)$ are outliers).

---

**Q: Why is the minimal seed group size 4 for homography estimation?**
**A:** A homography $\mathbf{H}$ is a $3\times3$ matrix with 8 degrees of freedom (9 entries minus 1 for scale). Each point correspondence supplies 2 independent equations. Therefore $\lceil 8/2 \rceil = 4$ correspondences are needed to solve the system exactly.

---

**Q: What is the collinearity check in RANSAC, and why is it needed?**
**A:** Before using a seed group, compute the triangle area for every triple of selected points:
$$\text{area} = \tfrac{1}{2}\bigl|p_1[x](p_2[y]-p_3[y]) + p_2[x](p_3[y]-p_1[y]) + p_3[x](p_1[y]-p_2[y])\bigr|$$
If any area $< 10^{-5}$, the triple is collinear and the DLT system is degenerate (no unique homography can be recovered); the seed group must be rejected and resampled.

---

**Q: Why does least squares fail where RANSAC succeeds?**
**A:** Least squares minimises the sum of squared residuals over **all** points, so even a few large-error outliers pull the solution away from the true model. RANSAC fits only to a minimal seed (pure inliers, if the draw is lucky) and scores by inlier count, making it immune to the outlier errors as long as outliers $< 50\%$.

---

## A4. Directions of reasoning

### Forward (standard): data → model → inliers

**Given:** A set of putative point-pair matches $\{(\mathbf{p}_i, \mathbf{p}'_i)\}$, threshold $\varepsilon$, iteration count $N$.
**Asked:** Best-fit model (e.g., homography $\mathbf{H}$) and the set of inliers.
**Key inference:** Run the loop; on each iteration the model is fit to 4 randomly drawn pairs; all other pairs are scored; after $N$ draws, the model with the largest inlier count is selected; a final DLT on all inliers of that model refines it.

### Reverse / inferential: model & points → inlier count / best hypothesis

**Given:** A hypothesised model (line $y = mx + b$, or a homography $\mathbf{H}$) and a set of points; possibly two competing hypotheses.
**Asked:** How many inliers does each hypothesis have? Which hypothesis is better?
**Key inference:**
- For a line: $d = |mx - y + b| / \sqrt{m^2 + 1}$; count points with $d \leq \varepsilon$.
- For a homography: project $\tilde{\mathbf{p}}_i$ through $\mathbf{H}$, dehomogenise, compute Euclidean distance to $\mathbf{p}'_i$; count points below threshold.
- The hypothesis with more inliers is preferred — no need to fit; just evaluate.

> [!example] A4 reverse example
> Two line hypotheses are proposed. Hypothesis A: $y = 45x - 183$. Hypothesis B: $y = 2x + 1000$. Using the 10-point dataset above with $\varepsilon = 3$ px, Hypothesis A yields 9 inliers and Hypothesis B yields 1 inlier. Therefore Hypothesis A is selected.

---

## A5. Standard implementation

### A5a. Setup

| Item | Value / description |
|---|---|
| **Input** | Set of $n$ putative matches $\{(\mathbf{p}_i, \mathbf{p}'_i)\}$ (pixel coordinates) |
| **Output** | Best model $\mathbf{H}^*$, inlier index set $\mathcal{I}^*$ |
| **Model** | Homography $\mathbf{H}$ ($3\times3$, 8 DOF) — for panorama stitching |
| **Minimal seed** | $s = 4$ point pairs |
| `inlierMappingThreshold` $\varepsilon$ | 1–2 pixels |
| `numberOfRandomDraws` $N$ | $> 1000$ |
| **Seed validity** | All 4 points distinct; no triple collinear (triangle area $< 10^{-5}$) |
| **Inlier test** | Euclidean distance in image plane $< \varepsilon$ after applying $\mathbf{H}$ |
| **Model fitting** | DLT via SVD (see [[homography-and-dlt]]) |

### A5b. Steps

1. **Initialise.** Set $\text{bestInlierCount} = 0$; $\mathbf{H}^* = \mathbf{I}$.

2. **Loop** for $k = 1, \ldots, N$:

   **2a. Draw seed.** Randomly select 4 distinct match indices $\{i_1, i_2, i_3, i_4\} \subset \{1,\ldots,n\}$.

   **2b. Validity check.** For every triple from the 4 selected source points, compute:
   $$\text{area} = \tfrac{1}{2}\bigl|p_{i_1}[x](p_{i_2}[y]-p_{i_3}[y]) + p_{i_2}[x](p_{i_3}[y]-p_{i_1}[y]) + p_{i_3}[x](p_{i_1}[y]-p_{i_2}[y])\bigr|$$
   If any area $< 10^{-5}$, **discard** and redraw.

   **2c. Fit model (DLT).** Stack the 4 seed pairs into the $8 \times 9$ matrix $\mathbf{A}$:
   $$\mathbf{A}\mathbf{h} = \mathbf{0}, \quad \mathbf{A} \in \mathbb{R}^{8\times9}$$
   Solve via SVD: $\mathbf{A} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$; $\mathbf{h} = $ last row of $\mathbf{V}^\top$ (smallest singular value). Reshape to $3\times3$ to get candidate $\mathbf{H}_k$.

   **2d. Count inliers.** For **every** match $(\mathbf{p}_i, \mathbf{p}'_i)$, $i = 1,\ldots,n$:
   $$\tilde{\mathbf{q}}_i = \mathbf{H}_k \tilde{\mathbf{p}}_i, \quad \mathbf{q}_i = \tilde{\mathbf{q}}_i / \tilde{q}_{i,3}$$
   $$e_i = \|\mathbf{q}_i - \mathbf{p}'_i\|_2$$
   Mark $i$ as inlier if $e_i < \varepsilon$. Count $c_k = |\mathcal{I}_k|$.

   **2e. Update best.** If $c_k > \text{bestInlierCount}$: $\text{bestInlierCount} \leftarrow c_k$; $\mathcal{I}^* \leftarrow \mathcal{I}_k$; $\mathbf{H}^* \leftarrow \mathbf{H}_k$.

3. **Post-loop refinement.** Re-run DLT using **all** inlier pairs $\{(\mathbf{p}_i, \mathbf{p}'_i) : i \in \mathcal{I}^*\}$ (now an overdetermined system solved by least-squares DLT via SVD). This gives the final $\mathbf{H}^*$.

4. **Output** $\mathbf{H}^*$ and $\mathcal{I}^*$. Optionally visualise only inlier matches to verify correctness.

> [!note] Line-fitting variant of step 2c–2d
> For a 2-D line $y = mx + b$ (minimal seed = 2 points): solve the $2\times2$ linear system for $m$ and $b$. Inlier distance:
> $$d = \frac{|mx_i - y_i + b|}{\sqrt{m^2 + 1}}$$
> A point is an inlier if $d \leq \varepsilon$.

---

## A6. Variations

### Line fitting (2-D)

- **Seed size:** 2 points (a line has 2 DOF: $m$, $b$).
- **Model fit:** Solve $2\times2$ linear system $\begin{pmatrix}x_1&1\\x_2&1\end{pmatrix}\begin{pmatrix}m\\b\end{pmatrix}=\begin{pmatrix}y_1\\y_2\end{pmatrix}$ exactly.
- **Inlier test:** Point-to-line distance $d = |mx - y + b|/\sqrt{m^2+1} < \varepsilon$.
- **Post-loop:** Refit line via ordinary least-squares on all inliers.
- Used in the W7T tutorial illustration; see numeric example in A3.

### Affine transformation

- **Seed size:** 3 non-collinear point pairs (affine has 6 DOF; each pair → 2 equations).
- **Model fit:** Solve exact $6\times6$ system; or use pseudoinverse if overdetermined.
- **Inlier test:** Euclidean distance after applying affine map.
- See [[affine-transformation-fitting]].

### Homography (standard, image stitching)

- **Seed size:** 4 non-collinear point pairs (8 DOF).
- **Model fit:** DLT + SVD.
- **Inlier test:** Reprojection distance $< \varepsilon$ (1–2 px).
- Connects to [[homography-and-dlt]], [[feature-matching]], [[image-warping-and-stitching]].

### Adaptive RANSAC

- **Change:** Instead of a fixed $N$, update the required number of iterations dynamically based on the current best inlier ratio $\hat{w} = c^*/n$:
  $$N = \frac{\log(1-p)}{\log(1-\hat{w}^s)}$$
  where $p$ is the desired probability of success (e.g., 0.99) and $s$ is the seed size.
- **When used:** When the outlier rate is unknown or highly variable; avoids over- or under-sampling.

> [!warning] Adaptive RANSAC beyond course slides
> The formula above is the standard reference derivation. It is not explicitly derived in the W5L or W7T slides; treat it as context, not examinable course content unless confirmed by the lecturer.
