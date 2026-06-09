# Feature Matching

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

Feature matching finds reliable correspondences between keypoints detected in two images by comparing their local descriptors, producing a set of **putative matches** that downstream algorithms (RANSAC, homography estimation, stitching) can use. It is the second stage in the robust feature-based alignment pipeline, after keypoint detection and description and before geometric model fitting. Without it, alignment requires known ground-truth correspondences, which are unavailable in practice.

---

## A2. High-level overview

### Robust feature-based alignment pipeline (umbrella)

1. **Extract** — detect Harris corners in both images; compute a descriptor (e.g. NCC patch, SIFT) at each.
2. **Putative match** — for each left descriptor, find the best and second-best matching right descriptor; apply Lowe's ratio test to filter out ambiguous matches.
3. **Hypothesise** — (RANSAC) randomly sample a minimal set of putative matches; fit transformation $T$ (homography, rigid, affine). See [[ransac]].
4. **Verify** — count inliers consistent with $T$; repeat until best $T$ found.
5. **Warp** — apply best $T$ to align/stitch the images. See [[image-warping-and-stitching]].

Feature matching covers steps 1–2 of this pipeline.

### Key definitions

| Term | Definition |
|---|---|
| **Putative match** | A candidate correspondence accepted by appearance alone (descriptor similarity), not yet verified geometrically |
| **Ambiguous match** | A query feature whose best and second-best candidates are similarly close in descriptor space (e.g. repeating fence pickets) |
| **Brute-force matching** | Exhaustive comparison: every left descriptor vs. every right descriptor |
| $d_1$ | Descriptor distance to the nearest-neighbour (best match) |
| $d_2$ | Descriptor distance to the second nearest-neighbour |
| $\text{NCC}_1$, $\text{NCC}_2$ | NCC similarity score of best and second-best match (higher = more similar) |
| $r$ | Ratio used in Lowe's test; for distances: $d_1/d_2$; for NCC scores: $\text{NCC}_2/\text{NCC}_1$ |
| $\tau$ | Ratio test threshold, typically $\in [0.8,\,0.9]$ |
| **Harris corner** | Interest point from which a descriptor patch is extracted. See [[harris-corner-detector]] |
| **Descriptor** | Fixed-length vector representation of the local patch around a keypoint. See [[feature-descriptors]] |
| **NCC** | Normalised Cross-Correlation similarity measure (range $[-1,1]$, higher = better). See [[patch-similarity-measures]] |
| **SSD** | Sum of Squared Differences (lower = better distance measure) |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Simple to implement; brute-force is exact (no approximate nearest-neighbour error).
- Lowe's ratio test is highly effective at removing ambiguous matches with a single scalar threshold.
- NCC precomputation amortises per-descriptor cost so only the dot product (numerator) is computed at match time.
- Works with any descriptor type (SSD, NCC, SIFT, ORB, SURF, SuperPoint).

**Shortcomings / limitations**
- **Brute-force is $O(N^2)$** in the number of keypoints — slow for large feature sets.
- Lowe's ratio test still passes some false matches; geometric verification (RANSAC) is required downstream.
- Repeated or self-similar structure (brick walls, fence posts, tiled floors) causes many ambiguous matches that the ratio test must discard, reducing the number of usable correspondences.
- Ratio test rejects valid matches in scenes with genuinely similar textures.
- NCC requires zero-mean patches; sensitive to photometric changes beyond mean shift (e.g. contrast change). SSD is not scale-invariant. SIFT/ORB needed for robustness to scale/rotation. See [[feature-descriptors]].
- A single keypoint may match multiple candidates one-to-many; the algorithm keeps only one-to-one correspondences via the ratio test.

---

### Likely exam questions

**Q:** What is the purpose of Lowe's ratio test, and what is the acceptance criterion?

**A:** It filters out ambiguous matches where the best candidate is not significantly more similar than the second best. For distance metrics: accept if $d_1/d_2 < \tau$ (typically $\tau = 0.8$–$0.9$). For NCC (similarity, higher = better): accept if $\text{NCC}_2/\text{NCC}_1 < 0.9$ and the ratio is non-negative.

---

**Q:** Why does repeated structure (e.g. a row of fence posts) cause problems in feature matching?

**A:** A keypoint on one fence post has multiple visually indistinguishable candidates in the other image. The best and second-best NCC scores are nearly equal, so $r = \text{NCC}_2/\text{NCC}_1 \approx 1$ — the match is flagged as ambiguous and rejected.

---

**Q:** Given $d_1 = 10$, $d_2 = 25$ (SSD distances), and $\tau = 0.8$, is the match accepted or rejected?

**A:** $d_1/d_2 = 10/25 = 0.4 < 0.8$ → **accepted** (distinctive match).

---

**Q:** Given $d_1 = 12$, $d_2 = 14$ (SSD distances), and $\tau = 0.8$, is the match accepted or rejected?

**A:** $d_1/d_2 = 12/14 \approx 0.857 > 0.8$ → **rejected** (ambiguous match).

---

**Q:** In the brute-force NCC algorithm, why is precomputation important and what is stored per descriptor?

**A:** Computing NCC from scratch for every left–right pair is expensive. By precomputing the mean-subtracted values $A_i = f_i - \bar{f}$ and the scalar $C = \sum_i A_i^2$ once per descriptor, matching reduces to computing only the dot-product numerator $\sum_i A_i B_i$ at query time; the denominator $\sqrt{C \cdot D}$ uses stored scalars.

---

**Q (numeric — SSD ratio test):** Patches $Q$, $P_1$, $P_2$ (3×3) are given (see A5b, step 5). Compute SSD scores and apply Lowe's ratio test with $\tau = 0.9$.

**A:** $\text{SSD}(Q,P_1) = 4+4+4+0+4+1+4+1+1 = 23$; $\text{SSD}(Q,P_2) = 4+1+1+1+4+0+1+1+4 = 17$. Best match: $P_2$ ($d_1=17$), second-best: $P_1$ ($d_2=23$). Ratio $= 17/23 \approx 0.739 < 0.9$ → **accepted**.

> [!note]
> The slides pose these patch matrices as an exercise (slide 19 of W4L_pt1) without publishing the answer. The SSD values above are computed from the given matrices; verify by expanding $(q_i - p_i)^2$ element-wise.

---

## A4. Directions of reasoning

### Forward / standard (descriptor scores → accept/reject decision)

**Given:** best descriptor distance (or NCC score) $d_1$ (or $\text{NCC}_1$) and second-best $d_2$ (or $\text{NCC}_2$), threshold $\tau$.

**Asked:** is this match a putative correspondence?

**Inference:**
- Distance metric: compute $r = d_1/d_2$; if $r < \tau$ → accept, else reject.
- NCC metric: compute $r = \text{NCC}_2/\text{NCC}_1$; if $r < 0.9$ and $r \geq 0$ → accept, else reject.

### Reverse / inferential (given scores, recover the decision or bound on $\tau$)

**Given:** best and second-best scores and the outcome (accepted or rejected); recover the implied threshold range or verify consistency.

**Example (W5T, slide 32):**
- Case 1: $r = 0.987 \geq 0.9$ → rejected (ambiguous). For this to be accepted would require $\tau > 0.987$.
- Case 2: $r = 0.2 < 0.9$ → accepted. Any $\tau \in (0.2, 1.0]$ would accept this match.
- Negative ratio (NCC scores have opposite signs) → reject regardless of $\tau$.

**Another reverse direction:** given that a match was rejected and $\tau = 0.8$, what can we infer? The ratio $d_1/d_2 \geq 0.8$, i.e. the two candidates were within a factor of 0.8 in distance — essentially indistinguishable.

---

## A5. Standard implementation

### a. Setup

**Algorithm:** brute-force feature matching with NCC and Lowe's ratio test.

**Inputs:**
- Feature set $F_1 = \{(\mathbf{x}_i, \mathbf{f}_i)\}$ from image $I(x,y)$ (left): keypoint locations and descriptors.
- Feature set $F_2 = \{(\mathbf{x}'_j, \mathbf{g}_j)\}$ from image $J(x,y)$ (right): keypoint locations and descriptors.
- Each descriptor is a $15 \times 15 = 225$-dimensional vector extracted at a [[harris-corner-detector]] keypoint.

**Parameters:**
- Ratio threshold $\tau \in [0.8, 0.9]$ (course uses 0.9 for NCC formulation).
- Patch size $15 \times 15$ pixels.

**Output:** List of putative matches $\{(\mathbf{x}_i, \mathbf{x}'_{j^*})\}$ passing the ratio test.

**Notation (normalised to course scheme):**
- $\mathbf{f} = (f_1, \ldots, f_{225})$, $\bar{f} = \frac{1}{225}\sum_i f_i$
- $A_i = f_i - \bar{f}$, $C = \sum_i A_i^2$
- $\mathbf{g} = (g_1, \ldots, g_{225})$, $\bar{g} = \frac{1}{225}\sum_i g_i$
- $B_i = g_i - \bar{g}$, $D = \sum_i B_i^2$

> [!note]
> The slides use $f$, $g$ (scalars for patch pixel values) and $A_i$, $B_i$, $C_i$, $D_i$ (per-element and per-descriptor scalars). Here $C$ and $D$ are the per-descriptor sum-of-squares scalars (not per-element).

---

### b. Steps

**Precomputation (once per descriptor, done after Harris detection):**

1. For each descriptor $\mathbf{f}_i$ in $F_1$ and $\mathbf{g}_j$ in $F_2$:
   $$\bar{f}_i = \frac{1}{225}\sum_{k=1}^{225}f_{ik}, \quad A_{ik} = f_{ik} - \bar{f}_i, \quad C_i = \sum_{k=1}^{225} A_{ik}^2$$
   Store $(A_{ik}, C_i)$ for each left descriptor; similarly store $(B_{jk}, D_j)$ for each right descriptor.

**Brute-force matching (outer loop over left descriptors):**

2. For each left descriptor $\mathbf{f}_i$, initialise:
   $$\text{bestMatch} \leftarrow -\infty, \quad \text{secondBestMatch} \leftarrow -\infty, \quad j^* \leftarrow \text{None}$$

3. Inner loop — for each right descriptor $\mathbf{g}_j$, compute NCC:
   $$\text{NCC}(\mathbf{f}_i, \mathbf{g}_j) = \frac{\sum_{k=1}^{225} A_{ik} \cdot B_{jk}}{\sqrt{C_i \cdot D_j}}$$

4. Update running best and second-best:
   $$\text{if NCC} > \text{bestMatch}: \quad \text{secondBestMatch} \leftarrow \text{bestMatch},\; \text{bestMatch} \leftarrow \text{NCC},\; j^* \leftarrow j$$
   $$\text{else if NCC} > \text{secondBestMatch}: \quad \text{secondBestMatch} \leftarrow \text{NCC}$$

5. After inner loop — apply Lowe's ratio test:
   $$r = \frac{\text{secondBestMatch}}{\text{bestMatch}}$$
   Accept (report putative match $\mathbf{x}_i \leftrightarrow \mathbf{x}'_{j^*}$) iff $r < 0.9$ **and** $r \geq 0$.

> [!note]
> For NCC (similarity, higher = better), the ratio is $\text{NCC}_2/\text{NCC}_1$ — the inverse of the distance-based form $d_1/d_2$. Both express the same idea: the best match must be sufficiently more distinctive than the second-best.

---

**Worked SSD ratio-test exercise (W4L_pt1, slide 19):**

Query and candidate patches:

$$Q = \begin{bmatrix} 10 & 200 & 10 \\ 10 & 200 & 10 \\ 200 & 200 & 200 \end{bmatrix}, \quad P_1 = \begin{bmatrix} 12 & 198 & 12 \\ 10 & 202 & 11 \\ 198 & 201 & 199 \end{bmatrix}, \quad P_2 = \begin{bmatrix} 8 & 201 & 9 \\ 11 & 198 & 10 \\ 201 & 199 & 202 \end{bmatrix}$$

$$\text{SSD}(Q,P_1) = (10{-}12)^2+(200{-}198)^2+(10{-}12)^2+(10{-}10)^2+(200{-}202)^2+(10{-}11)^2+(200{-}198)^2+(200{-}201)^2+(200{-}199)^2 = 4+4+4+0+4+1+4+1+1 = 23$$

$$\text{SSD}(Q,P_2) = (10{-}8)^2+(200{-}201)^2+(10{-}9)^2+(10{-}11)^2+(200{-}198)^2+(10{-}10)^2+(200{-}201)^2+(200{-}199)^2+(200{-}202)^2 = 4+1+1+1+4+0+1+1+4 = 17$$

Best match: $P_2$ ($d_1 = 17$), second-best: $P_1$ ($d_2 = 23$).

$$r = \frac{d_1}{d_2} = \frac{17}{23} \approx 0.739 < 0.9 \implies \textbf{accepted}$$

---

**W5T ratio test cases (slide 32):**

> [!example]
> **Case 1:** $\text{NCC}_1 = 0.8$, $\text{NCC}_2 = 0.79$. $r = 0.79/0.8 = 0.9875 \geq 0.9$ → **rejected** (ambiguous).
>
> **Case 2:** $\text{NCC}_1 = 0.8$, $\text{NCC}_2 = 0.16$. $r = 0.16/0.8 = 0.2 < 0.9$ → **accepted** (distinctive).
>
> **Case 3 (negative ratio):** One or both NCC scores are negative → reject regardless.

---

## A6. Variations

### Distance-based ratio test (SSD / SAD / Euclidean)

**Change:** Use descriptor distances $d_1$ (nearest-neighbour), $d_2$ (second-nearest). Lower distance = better match.

$$\text{Accept iff } \frac{d_1}{d_2} < \tau, \quad \tau \in [0.8, 0.9]$$

Used with SSD, SAD, or Euclidean distance on SIFT 128-D vectors (Lowe's original formulation). See [[patch-similarity-measures]] for SSD/SAD/NCC definitions.

### Similarity-based ratio test (NCC)

**Change:** Use NCC scores $\text{NCC}_1 \geq \text{NCC}_2$ (higher = better). Ratio is inverted:

$$\text{Accept iff } \frac{\text{NCC}_2}{\text{NCC}_1} < 0.9 \text{ and ratio} \geq 0$$

The slides use this form explicitly for the brute-force NCC algorithm (W4L_pt1, W5T).

### Binary descriptor matching (Hamming distance)

**Change:** Descriptors (ORB, BRIEF) are binary; distance is Hamming distance (bit-XOR count). Ratio test applies identically with Hamming distances instead of Euclidean. Brute-force matching uses XOR+popcount. See [[feature-descriptors]].

### Approximate nearest-neighbour (ANN) matching

**Change:** Replace exhaustive inner loop with a $k$-d tree or FLANN index. Returns approximate best and second-best matches faster than $O(N)$ per query. Ratio test applied identically. Used in practice for large feature sets; not covered in detail in the course slides.

### Cross-check (mutual consistency) filter

**Change:** After brute-force matching in both directions (left→right and right→left), retain only matches that are mutual nearest-neighbours (left's best match in right also has left as its best match in left). Complementary to ratio test; reduces false matches further. Not explicitly covered in course slides.

### RANSAC + geometric verification (downstream)

After putative matches are generated by the above pipeline, [[ransac]] is applied to fit a geometric model ($T$ = homography, affine, rigid) and identify inliers, completing the full alignment pipeline. See also [[homography-and-dlt]], [[image-warping-and-stitching]].
