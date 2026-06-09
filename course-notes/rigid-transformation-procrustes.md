# Rigid Transformation — Procrustes / Orthogonal Procrustes Problem

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

Given $N$ corresponding 2D point pairs $\{\mathbf{x}_i\} \leftrightarrow \{\mathbf{x}'_i\}$, find the rigid (Euclidean) transformation $\mathbf{x}' = \mathbf{R}\mathbf{x} + \mathbf{t}$ — rotation plus translation, 3 dof — that minimises mean squared residual distance. Used wherever distances and angles must be preserved: aligning feature point clouds between two images, registering pre-operative to intra-operative data in image-guided surgery, and as the inner model in a RANSAC loop for robust image alignment.

---

## A2. High-level overview

**Steps (no math):**
1. Subtract the mean from each point set to remove translation (centre the clouds).
2. Build the 2×2 cross-covariance matrix $\mathbf{H}$ from the centred point clouds, decompose it with SVD, and extract the rotation.
3. Recover the translation from the two centroid vectors and the recovered rotation.

**Key definitions:**

| Term | Definition |
|---|---|
| Rigid / Euclidean transform | $\mathbf{x}' = \mathbf{R}\mathbf{x} + \mathbf{t}$; 3 dof (2D); preserves all distances and angles |
| $\mathbf{R}$ | $2\times2$ orthogonal rotation matrix, $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$, $\det\mathbf{R}=+1$ |
| $\mathbf{t}$ | $2\times1$ translation vector |
| $\bar{\mathbf{x}},\,\bar{\mathbf{x}}'$ | Centroid (mean) of source and target point sets |
| $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$ | Centred (mean-subtracted) source point |
| $\tilde{\mathbf{x}}'_i = \mathbf{x}'_i - \bar{\mathbf{x}}'$ | Centred target point |
| $\widetilde{\mathbf{X}},\,\widetilde{\mathbf{X}}'$ | $N\times2$ matrices with rows $\tilde{\mathbf{x}}_i^\top$, $\tilde{\mathbf{x}}'^\top_i$ |
| $\mathbf{H}$ | $2\times2$ **cross-covariance** matrix $\widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}'$ — **not** a homography or Harris structure tensor |
| $\mathbf{U},\mathbf{L},\mathbf{V}$ | SVD factors of $\mathbf{H}$: $\mathbf{H}=\mathbf{U}\mathbf{L}\mathbf{V}^\top$ |
| $\mathbf{D}$ | Diagonal reflection-correction matrix $\mathrm{diag}(1,\det(\mathbf{U}\mathbf{V}^\top))$ |
| Procrustes error | $E_\text{Procr.} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{R}\mathbf{x}_i+\mathbf{t}-\mathbf{x}'_i\|_2^2$ |

> [!note] Overloaded symbol $\mathbf{H}$
> In this topic $\mathbf{H} \equiv \widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}'$ is the **cross-covariance** (prediction) matrix — a $2\times2$ matrix. The same letter is used elsewhere in CS773 for the **homography** ($3\times3$, 8 dof) and for the **Harris structure tensor** ($2\times2$ built from image gradients). Context always disambiguates.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Closed-form, non-iterative — runs in $O(N)$ after forming $\mathbf{H}$ (SVD is fixed-size $2\times2$).
- Optimal least-squares solution (global minimum guaranteed for the orthogonal Procrustes problem).
- Numerically stable via SVD; reflection ambiguity is handled exactly by the $\mathbf{D}$ correction.

**Shortcomings / limitations**
- Assumes **exact correspondences** — one outlier can corrupt the result severely (combine with [[ransac]] for robustness).
- Restricted to **rigid** motion only — if the true transformation includes scale, shear, or perspective, use [[affine-transformation-fitting]] or [[homography-and-dlt]].
- Requires **at least 2 non-collinear point pairs** in 2D (2 pairs give 4 equations, matching 3 unknowns, with slight over-determination); 1 pair is degenerate (translation only, no unique rotation).
- Translation cannot be solved simultaneously with rotation — the centering step is mandatory first.

### Likely exam questions

**Q:** Write out the three steps of the Procrustes algorithm in order.
**A:** (1) Subtract the centroid from each point set: $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$, $\tilde{\mathbf{x}}'_i = \mathbf{x}'_i - \bar{\mathbf{x}}'$. (2) Form $\mathbf{H}=\widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}'$, compute SVD $\mathbf{H}=\mathbf{U}\mathbf{L}\mathbf{V}^\top$, set $\mathbf{R}=\mathbf{U}\mathbf{D}\mathbf{V}^\top$ with $\mathbf{D}=\mathrm{diag}(1,\det(\mathbf{U}\mathbf{V}^\top))$. (3) $\mathbf{t}=\bar{\mathbf{x}}'-\mathbf{R}\bar{\mathbf{x}}$.

---

**Q:** Why is the $\mathbf{D}$ matrix needed in $\mathbf{R}=\mathbf{U}\mathbf{D}\mathbf{V}^\top$? What happens without it?
**A:** SVD can produce $\mathbf{U}\mathbf{V}^\top$ with $\det=-1$, which is a reflection, not a proper rotation. $\mathbf{D}=\mathrm{diag}(1,\det(\mathbf{U}\mathbf{V}^\top))$ flips the sign of the last singular vector of $\mathbf{U}$ when needed, forcing $\det\mathbf{R}=+1$. Without it the algorithm might return a reflected solution that minimises the Frobenius norm but is not physically realisable as a rotation.

---

**Q:** Why must you centre the point sets **before** computing $\mathbf{H}$?
**A:** The rotation optimisation decouples from translation only after centring — without it, $\mathbf{H}$ conflates rotation and translation, and the SVD will return an incorrect $\mathbf{R}$. Centring removes translation from the data so the remaining problem is purely rotational.

---

**Q:** Using the course worked example: $A=\{(2,2),(2,4),(5,6)\}$, $B=\{(-3,-3),(-3,-5),(-6,-7)\}$. What are $\mathbf{R}$ and $\mathbf{t}$?
**A:** $\bar{A}=(3,4)$, $\bar{B}=(-4,-5)$. After centring and computing $\mathbf{H}$, SVD gives $\mathbf{R}=\begin{pmatrix}-1&0\\0&-1\end{pmatrix}$ (180° rotation). Then $\mathbf{t}=(-4,-5)-(-1,0;0,-1)(3,4)^\top=(-1,-1)^\top$.

---

**Q:** What is the minimum number of point correspondences needed to fit a rigid transformation, and why?
**A:** 2 non-collinear pairs. A rigid 2D transform has 3 unknowns ($\theta$, $t_x$, $t_y$); 2 point pairs give 4 scalar equations, which over-determines the system. 1 pair is insufficient (rotation is unconstrained).

---

**Q:** Given $\mathbf{R}=\begin{pmatrix}-1&0\\0&-1\end{pmatrix}$ and $\mathbf{t}=\begin{pmatrix}-1\\-1\end{pmatrix}$, what physical motion does this represent?
**A:** A 180° rotation about the origin (since $\mathbf{R}=\mathrm{diag}(-1,-1)$ reverses both axes), followed by a translation of $(-1,-1)$. All inter-point distances are preserved; this is a proper rigid motion.

---

## A4. Directions of reasoning

### Forward / standard (given point sets → find $\mathbf{R},\mathbf{t}$)

**Given:** $N$ corresponding pairs $\{\mathbf{x}_i\},\{\mathbf{x}'_i\}$.
**Find:** Rotation $\mathbf{R}$ and translation $\mathbf{t}$ that best map source to target.
**Key manipulation:** Centering decouples translation; cross-covariance + SVD extracts rotation optimally; translation then follows from the centroid equation.

### Reverse / inferential (given $\mathbf{R},\mathbf{t}$ → verify / recover motion)

**Given:** Known or recovered $\mathbf{R}$ and $\mathbf{t}$.
**Find:** The physical rigid motion, or verify by forward-mapping source points.

**Inference steps:**
1. **Interpret $\mathbf{R}$**: read rotation angle from $\mathbf{R}=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$. If $\mathbf{R}=\mathrm{diag}(-1,-1)$, then $\cos\theta=-1\Rightarrow\theta=\pi$ (180°).
2. **Verify a point**: compute $\mathbf{R}\mathbf{x}_i+\mathbf{t}$ and check it equals $\mathbf{x}'_i$.
3. **Course example check**: $\mathbf{R}(2,2)^\top+(-1,-1)^\top = (-2,-2)^\top+(-1,-1)^\top = (-3,-3)^\top = \mathbf{x}'_1$. Correct.
4. **Recover original source**: given $\mathbf{x}' = \mathbf{R}\mathbf{x}+\mathbf{t}$, invert as $\mathbf{x}=\mathbf{R}^\top(\mathbf{x}'-\mathbf{t})$ (since $\mathbf{R}^{-1}=\mathbf{R}^\top$ for orthogonal matrices).

---

## A5. Standard implementation

### A5a. Setup

| Item | Detail |
|---|---|
| Input | $N$ corresponding 2D point pairs: $\mathbf{x}_i=(x_i,y_i)^\top$, $\mathbf{x}'_i=(x'_i,y'_i)^\top$, $i=1,\ldots,N$ |
| Representation | $\mathbf{X}$ and $\mathbf{X}'$ are $N\times2$ matrices (rows = point coordinates) |
| Output | Rotation $\mathbf{R}$ ($2\times2$), translation $\mathbf{t}$ ($2\times1$) |
| Assumptions | Correspondences are known and correct; transformation is rigid (distance-preserving) |
| Objective | $\min_{\mathbf{R},\mathbf{t}}\;\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{R}\mathbf{x}_i+\mathbf{t}-\mathbf{x}'_i\|_2^2$ subject to $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$, $\det\mathbf{R}=+1$ |

### A5b. Steps

**Step 1 — Centre both point sets.**

$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{x}_i, \qquad \bar{\mathbf{x}}' = \frac{1}{N}\sum_{i=1}^{N}\mathbf{x}'_i$$

$$\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}, \qquad \tilde{\mathbf{x}}'_i = \mathbf{x}'_i - \bar{\mathbf{x}}'$$

Form the $N\times2$ centred matrices $\widetilde{\mathbf{X}}$ (rows = $\tilde{\mathbf{x}}_i^\top$) and $\widetilde{\mathbf{X}}'$ (rows = $\tilde{\mathbf{x}}'^\top_i$).

> [!warning] Centre BEFORE forming $\mathbf{H}$
> If you skip centring, the cross-covariance conflates rotation and translation and the SVD gives a wrong rotation.

**Step 2 — Form cross-covariance and solve orthogonal Procrustes.**

Form the $2\times2$ cross-covariance matrix:
$$\mathbf{H} = \widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}'$$

Compute SVD:
$$\mathbf{H} = \mathbf{U}\mathbf{L}\mathbf{V}^\top, \qquad \mathbf{U}^\top\mathbf{U}=\mathbf{V}^\top\mathbf{V}=\mathbf{I}, \qquad \mathbf{L}=\mathrm{diag}(\sigma_1,\sigma_2),\;\sigma_1\geq\sigma_2\geq0$$

Apply reflection correction and extract rotation:
$$\mathbf{D} = \mathrm{diag}\!\left(1,\;\det(\mathbf{U}\mathbf{V}^\top)\right), \qquad \mathbf{R} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$$

- If $\det(\mathbf{U}\mathbf{V}^\top)=+1$: $\mathbf{D}=\mathbf{I}$, so $\mathbf{R}=\mathbf{U}\mathbf{V}^\top$ directly. Corresponds to a proper rotation.
- If $\det(\mathbf{U}\mathbf{V}^\top)=-1$: $\mathbf{D}=\mathrm{diag}(1,-1)$, which flips the sign of the second column of $\mathbf{U}$, making $\det\mathbf{R}=+1$.

**Step 3 — Recover translation.**

$$\mathbf{t} = \bar{\mathbf{x}}' - \mathbf{R}\bar{\mathbf{x}}$$

> [!note] Order matters
> Translation is computed **after** rotation. The formula expresses that the centroid of the source, after rotation, must land on the centroid of the target.

---

## Worked example (W5T / W6T Q14)

**Given:**
$$A = \{(2,2),\,(2,4),\,(5,6)\}, \qquad B = \{(-3,-3),\,(-3,-5),\,(-6,-7)\}$$
($B$ was constructed by rotating $A$ by $\theta=\pi$ then translating by $\mathbf{t}=(-1,-1)^\top$ — the algorithm should recover exactly these.)

**Step 1 — Centroids and centering:**
$$\bar{A} = (3,4), \quad \bar{B} = (-4,-5)$$

$$\widetilde{A} = \begin{bmatrix}-1&-2\\-1&0\\2&2\end{bmatrix}, \qquad \widetilde{B} = \begin{bmatrix}1&2\\1&0\\-2&-2\end{bmatrix}$$

**Step 2 — Cross-covariance:**

$$\mathbf{H} = \widetilde{A}^\top\widetilde{B} = \begin{bmatrix}-1&-1&2\\-2&0&2\end{bmatrix}\begin{bmatrix}1&2\\1&0\\-2&-2\end{bmatrix} = \begin{bmatrix}-6&-6\\-6&-8\end{bmatrix}$$

> [!note] Note from W5T vs W6T
> W5T reports $\mathbf{H}=\begin{pmatrix}-6&-6\\-8&-8\end{pmatrix}$ while W6T reports $\begin{pmatrix}-6&-6\\-6&-8\end{pmatrix}$. The W6T computation matches the matrix multiplication shown explicitly in both sources; W5T's entry appears to be a transcription slip. Both sources agree on the final $\mathbf{R}$ and $\mathbf{t}$.

SVD (via `np.linalg.svd`):
$$\mathbf{U}=\begin{pmatrix}-0.6464&-0.7630\\-0.7630&0.6464\end{pmatrix}, \quad \mathbf{V}=\begin{pmatrix}0.6464&0.7630\\0.7630&0.6464\end{pmatrix}$$

$\det(\mathbf{U}\mathbf{V}^\top)=1 \Rightarrow \mathbf{D}=\mathbf{I}$

$$\mathbf{R} = \mathbf{U}\mathbf{V}^\top = \begin{pmatrix}-1&0\\0&-1\end{pmatrix}$$

This is a 180° rotation ($\theta=\pi$), as expected.

**Step 3 — Translation:**

$$\mathbf{t} = \bar{B} - \mathbf{R}\bar{A} = \begin{pmatrix}-4\\-5\end{pmatrix} - \begin{pmatrix}-1&0\\0&-1\end{pmatrix}\begin{pmatrix}3\\4\end{pmatrix} = \begin{pmatrix}-4\\-5\end{pmatrix} - \begin{pmatrix}-3\\-4\end{pmatrix} = \begin{pmatrix}-1\\-1\end{pmatrix}$$

**Verification (reverse direction):** $\mathbf{R}(2,2)^\top + (-1,-1)^\top = (-2,-2)^\top + (-1,-1)^\top = (-3,-3)^\top = \mathbf{x}'_1$. Correct.

> [!example] Exam tip: recover the motion from $\mathbf{R}$
> $\mathbf{R}=\mathrm{diag}(-1,-1) = \begin{pmatrix}\cos\pi&-\sin\pi\\\sin\pi&\cos\pi\end{pmatrix}$ confirms $\theta=180°$. To find the original source from a mapped point: $\mathbf{x} = \mathbf{R}^\top(\mathbf{x}'-\mathbf{t})$.

---

## A6. Variations

### Orthogonal Procrustes (matrix form)

The rotation step is equivalent to the classical **Orthogonal Procrustes Problem**:

$$\min_{\mathbf{R}}\|\mathbf{P}\mathbf{R}-\mathbf{Q}\|_F \quad\text{subject to}\quad \mathbf{R}^\top\mathbf{R}=\mathbf{I}$$

where $\mathbf{P}=\widetilde{\mathbf{X}}$, $\mathbf{Q}=\widetilde{\mathbf{X}}'$ in our notation. This is equivalent to maximising $\mathrm{tr}(\mathbf{R}^\top\mathbf{P}^\top\mathbf{Q})$, solved by computing SVD of $\mathbf{P}^\top\mathbf{Q}$ and setting $\mathbf{R}=\mathbf{U}\mathbf{V}^\top$. Reference: Golub & van Loan.

### Similarity transform (scale + rotation + translation)

Adds a uniform scale $s$ as a 4th parameter. After centring, $s$ is found as $s = \mathrm{tr}(\mathbf{L}\mathbf{D}^\top)/\sum_i\|\tilde{\mathbf{x}}_i\|^2$; rotation and translation follow identically. Not covered in depth in CS773 — use [[affine-transformation-fitting]] if scale is needed.

### Robust Procrustes (inside RANSAC)

Use the 3-step SVD method as the **model estimator** inside a [[ransac]] loop: sample 2 correspondences, compute $(\mathbf{R},\mathbf{t})$, count inliers (points where $\|\mathbf{R}\mathbf{x}_i+\mathbf{t}-\mathbf{x}'_i\|_2 < \epsilon$), iterate. This is the standard pipeline for robust image alignment using rigid transforms from [[feature-matching]] output.

### 3D rigid registration (ICP)

Extends to 3D: $\mathbf{x},\mathbf{x}'\in\mathbb{R}^3$; $\mathbf{H}$ becomes $3\times3$; $\mathbf{D}=\mathrm{diag}(1,1,\det(\mathbf{U}\mathbf{V}^\top))$. When correspondences are unknown, the Iterative Closest Point (ICP) algorithm alternates between finding nearest-neighbour correspondences and solving Procrustes. Not assessed in CS773.

---

**Related topics:** [[singular-value-decomposition]] · [[homogeneous-coordinates-and-transformations]] · [[affine-transformation-fitting]] · [[feature-matching]]
