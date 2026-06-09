# Homography and DLT

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level Overview]]
- [[#A3. Strengths, Shortcomings & Limitations]]
  - [[#Likely Exam Questions]]
- [[#A4. Directions of Reasoning]]
- [[#A5. Standard Implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

The **homography** is the appropriate 2D transformation model when two images are related by either a view of a **planar surface** (e.g. rectifying a billboard) or two cameras sharing the **same optical centre** (e.g. a pure-rotation panorama). The **Direct Linear Transformation (DLT)** algorithm computes the 3×3 homography matrix $\mathbf{H}$ from $N \geq 4$ point correspondences by converting the mapping equation into a homogeneous linear system solved by SVD.

> [!note] Notation clash
> In this course $\mathbf{H}$ is overloaded: (1) the $2\times2$ cross-covariance matrix in the Procrustes/rigid-body problem (see [[rigid-transformation-procrustes]]), and (2) the $3\times3$ homography here. The context (dimensionality) disambiguates.

---

## A2. High-level Overview

**Key definitions**

| Term | Definition |
|------|-----------|
| Homography $\mathbf{H}$ | $3\times3$ plane projective transformation matrix; 9 entries but only **8 dof** (scale-ambiguous) |
| $w$ | Homogeneous scale factor (unknown per point) |
| $\tilde{\mathbf{x}} = (x,y,1)^\top$ | Homogeneous image coordinate in projective space $\mathcal{P}^2$ |
| Cross-product constraint | $\tilde{\mathbf{x}}' \times \mathbf{H}\tilde{\mathbf{x}} = \mathbf{0}$; eliminates $w$; yields 3 eqns/match (2 independent) |
| Design matrix $\mathbf{A}$ | $2N \times 9$ matrix stacking two rows per correspondence |
| $\mathbf{h}$ | $9\times1$ vectorisation of $\mathbf{H}$ (row-major: $[h_{11},h_{12},h_{13},h_{21},\ldots,h_{33}]^\top$) |
| DLT | Direct Linear Transformation — the standard algorithm to solve $\mathbf{A}\mathbf{h}=\mathbf{0}$ |

**Two physical scenarios where homography applies**
1. **Planar surface**: any two views of a flat object (a book, a road marking, a billboard).
2. **Same optical centre (pure rotation / panorama)**: two cameras at the same centre looking in different directions — all scene rays map consistently through a single 3×3 matrix.

**Overview steps** (no math yet)
1. Collect $N \geq 4$ point correspondences $(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}'_i)$, no three collinear.
2. For each pair, write out two linear equations in the nine entries of $\mathbf{H}$.
3. Stack all equations into $\mathbf{A}\mathbf{h} = \mathbf{0}$ (homogeneous system).
4. Solve by constrained homogeneous least squares using SVD.
5. Reshape the $9\times1$ solution into the $3\times3$ matrix $\mathbf{H}$.

---

## A3. Strengths, Shortcomings & Limitations

**Strengths**
- Exact model for the two physical scenarios above; no approximation.
- Closed-form SVD solution — fast, no iterative optimisation needed.
- Straight lines remain straight (projective-line preservation).
- 4-point minimal case gives an exact (not over-determined) solution.

**Shortcomings / Limitations**
- Only valid for **planar scenes** or **pure-rotation cameras**; breaks for general 3D scenes or cameras with different centres (need full fundamental/essential matrix).
- 8 dof vs 6 (affine): more parameters to estimate → less stable with few matches.
- Sensitive to degenerate configurations: collinear point triples make $\mathbf{A}$ rank-deficient.
- Scale ambiguity: $\mathbf{H}$ and $c\mathbf{H}$ represent the same transformation for any $c \neq 0$.
- No built-in robustness to outliers — pair with [[ransac]] in practice.
- Algebraic error minimised by DLT ≠ geometric (reprojection) error; normalisation of coordinates (not detailed here) improves conditioning.

---

### Likely Exam Questions

**Q:** What is the minimum number of point correspondences needed to compute a homography, and why?

**A:** 4. A homography has 8 dof (9 entries, scale-free). Each point pair contributes 2 independent equations, so $2 \times 4 = 8$ equations just determines the 8 unknowns. No three of the four points may be collinear, or $\mathbf{A}$ becomes rank-deficient.

---

**Q:** Starting from $w\tilde{\mathbf{x}}' = \mathbf{H}\tilde{\mathbf{x}}$, derive the two linear equations per correspondence used in DLT.

**A:** Take the cross-product of both sides with $\tilde{\mathbf{x}}'$:
$$\tilde{\mathbf{x}}' \times w\tilde{\mathbf{x}}' = \tilde{\mathbf{x}}' \times \mathbf{H}\tilde{\mathbf{x}} = \mathbf{0}$$
since $\mathbf{a}\times\mathbf{a}=\mathbf{0}$. Expanding with $\mathbf{h}_1^\top,\mathbf{h}_2^\top,\mathbf{h}_3^\top$ as rows of $\mathbf{H}$ gives three equations; the third is a linear combination of the first two, so only two are used:
$$\begin{bmatrix}\mathbf{0}^\top & \tilde{\mathbf{x}}_i^\top & -y'_i\,\tilde{\mathbf{x}}_i^\top\\\tilde{\mathbf{x}}_i^\top & \mathbf{0}^\top & -x'_i\,\tilde{\mathbf{x}}_i^\top\end{bmatrix}\mathbf{h} = \mathbf{0}$$

---

**Q:** Why is the DLT solved by finding the last row of $\mathbf{V}^\top$ from the SVD of $\mathbf{A}$?

**A:** We minimise $\|\mathbf{A}\mathbf{h}\|^2$ subject to $\|\mathbf{h}\|=1$. This is the eigenvector of $\mathbf{A}^\top\mathbf{A}$ corresponding to the smallest eigenvalue. By the SVD $\mathbf{A}=\mathbf{U}\mathbf{D}\mathbf{V}^\top$, the columns of $\mathbf{V}$ are those eigenvectors in decreasing singular-value order, so the last column of $\mathbf{V}$ (= last row of $\mathbf{V}^\top$) gives $\mathbf{h}$.

---

**Q:** Name the two physical scenarios in which a homography exactly relates two images.

**A:** (1) Two views of a **planar surface**. (2) Two cameras with the **same optical centre** (pure rotation — the standard panorama-stitching setup).

---

**Q:** In the W7T worked example, the DLT solution for the 4-point homography is given. Show how to verify that the point $(2,2,1)^\top$ maps to $(-2,-2,1)^\top$.

**A:** Compute $\mathbf{H}\tilde{\mathbf{x}} = \mathbf{H}[2,2,1]^\top \approx [-0.312, -0.312, 0.156]^\top$. Divide by the third component $0.156$: $[-2,-2,1]^\top$. This matches $\tilde{\mathbf{x}}'$. (The scale factor $w=0.156$ is absorbed by the homogeneous division.)

---

**Q:** Why does the cross-product constraint give 3 equations per match but only 2 are used?

**A:** The third equation ($x'_i \mathbf{h}_2^\top\tilde{\mathbf{x}}_i - y'_i \mathbf{h}_1^\top\tilde{\mathbf{x}}_i = 0$) is a linear combination of the first two, so it provides no additional information — including it would only add a redundant row to $\mathbf{A}$ and is conventionally dropped.

---

## A4. Directions of Reasoning

### Forward / Standard (correspondences → $\mathbf{H}$)

**Given:** $N \geq 4$ point pairs $\{(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}'_i)\}$, no three collinear.  
**Asked:** The $3\times3$ homography $\mathbf{H}$ such that $w\tilde{\mathbf{x}}' \approx \mathbf{H}\tilde{\mathbf{x}}$.  
**Key steps:** Build $\mathbf{A}$ ($2N\times9$) → SVD → last row of $\mathbf{V}^\top$ → reshape.

### Reverse / Inferential ($\mathbf{H}$ → map points or diagnose scenario)

**Given:** A known $\mathbf{H}$ and a source point $\tilde{\mathbf{x}}$.  
**Asked:** Target point $\tilde{\mathbf{x}}'$.  
**Key inference:** Compute $\mathbf{s} = \mathbf{H}\tilde{\mathbf{x}}$; divide by the third component: $\tilde{\mathbf{x}}' = \mathbf{s}/s_3$.

**Given:** A known $\mathbf{H}$ and a physical setup description.  
**Asked:** Identify whether the scenario is a planar surface view or a pure-rotation camera pair.  
**Key inference:** Both physically yield a $3\times3$ projective relationship; if the scene is 3D and cameras differ in position, $\mathbf{H}$ breaks down and an epipolar model ([[epipolar-geometry]]) is needed instead.

---

## A5. Standard Implementation

### A5a. Setup

| Item | Detail |
|------|--------|
| Input | $N \geq 4$ point correspondences $\{(\mathbf{x}_i, \mathbf{x}'_i)\}$, $\mathbf{x}_i=(x_i,y_i)$, $\mathbf{x}'_i=(x'_i,y'_i)$ |
| Homogeneous form | $\tilde{\mathbf{x}}_i = (x_i, y_i, 1)^\top \in \mathcal{P}^2$ |
| Unknown | $\mathbf{h} \in \mathbb{R}^9$ (vectorised $\mathbf{H}$, row-major), unit-norm constraint |
| Output | $3\times3$ matrix $\mathbf{H}$ (defined up to scale) |
| Degeneracy | No three source or target points collinear |
| Constraint | $\|\mathbf{h}\|=1$ (removes scale ambiguity) |

### A5b. Steps

**Step 1 — Write the mapping equation.**

$$w\begin{bmatrix}x'_i\\y'_i\\1\end{bmatrix} = \begin{bmatrix}h_{11}&h_{12}&h_{13}\\h_{21}&h_{22}&h_{23}\\h_{31}&h_{32}&h_{33}\end{bmatrix}\begin{bmatrix}x_i\\y_i\\1\end{bmatrix}$$

where $w$ is an unknown scalar that differs per point.

**Step 2 — Apply the cross-product constraint to eliminate $w$.**

$$\tilde{\mathbf{x}}'_i \times \mathbf{H}\tilde{\mathbf{x}}_i = \mathbf{0}$$

Expanding (with $\mathbf{h}_k^\top$ = $k$-th row of $\mathbf{H}$):

$$\begin{bmatrix}x'_i\\y'_i\\1\end{bmatrix}\times\begin{bmatrix}\mathbf{h}_1^\top\tilde{\mathbf{x}}_i\\\mathbf{h}_2^\top\tilde{\mathbf{x}}_i\\\mathbf{h}_3^\top\tilde{\mathbf{x}}_i\end{bmatrix} = \begin{bmatrix}y'_i\,\mathbf{h}_3^\top\tilde{\mathbf{x}}_i - \mathbf{h}_2^\top\tilde{\mathbf{x}}_i\\\mathbf{h}_1^\top\tilde{\mathbf{x}}_i - x'_i\,\mathbf{h}_3^\top\tilde{\mathbf{x}}_i\\x'_i\,\mathbf{h}_2^\top\tilde{\mathbf{x}}_i - y'_i\,\mathbf{h}_1^\top\tilde{\mathbf{x}}_i\end{bmatrix} = \mathbf{0}$$

**Step 3 — Extract two independent linear equations per point pair.**

Rearranging equations 1 and 2 into a $2\times9$ row block (using $\tilde{\mathbf{x}}_i^\top = [x_i\ y_i\ 1]$):

$$\underbrace{\begin{bmatrix}\mathbf{0}^\top & \tilde{\mathbf{x}}_i^\top & -y'_i\,\tilde{\mathbf{x}}_i^\top\\\tilde{\mathbf{x}}_i^\top & \mathbf{0}^\top & -x'_i\,\tilde{\mathbf{x}}_i^\top\end{bmatrix}}_{2\times9}\mathbf{h} = \mathbf{0}$$

Written out entry-by-entry for point $i$:

$$\begin{bmatrix}0 & 0 & 0 & x_i & y_i & 1 & -y'_i x_i & -y'_i y_i & -y'_i\\x_i & y_i & 1 & 0 & 0 & 0 & -x'_i x_i & -x'_i y_i & -x'_i\end{bmatrix}\mathbf{h} = \mathbf{0}$$

**Step 4 — Stack all $N$ point pairs into the design matrix $\mathbf{A}$.**

$$\mathbf{A}_{2N\times9} = \begin{bmatrix}0 & 0 & 0 & x_1 & y_1 & 1 & -y'_1 x_1 & -y'_1 y_1 & -y'_1\\x_1 & y_1 & 1 & 0 & 0 & 0 & -x'_1 x_1 & -x'_1 y_1 & -x'_1\\\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\0 & 0 & 0 & x_N & y_N & 1 & -y'_N x_N & -y'_N y_N & -y'_N\\x_N & y_N & 1 & 0 & 0 & 0 & -x'_N x_N & -x'_N y_N & -x'_N\end{bmatrix}$$

giving the homogeneous system $\mathbf{A}\mathbf{h} = \mathbf{0}$.

**Step 5 — Solve homogeneous least squares.**

$$\min_{\mathbf{h}}\|\mathbf{A}\mathbf{h}\|^2 \quad \text{s.t.} \quad \|\mathbf{h}\|=1$$

Compute SVD: $\mathbf{A} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$.

Solution $\mathbf{h}$ = **last row of $\mathbf{V}^\top$** (= last column of $\mathbf{V}$), i.e. the right singular vector corresponding to the **smallest singular value**.

Python: `U, D, Vt = np.linalg.svd(A)` → `h = Vt[-1]`

**Step 6 — Reshape to $3\times3$.**

$$\mathbf{H} = \mathbf{h}\text{.reshape}(3,3)$$

**Step 7 — Verify (optional but useful).**

For each test point, compute $\mathbf{s} = \mathbf{H}\tilde{\mathbf{x}}_i$ and check $\mathbf{s}/s_3 \approx \tilde{\mathbf{x}}'_i$.

> [!example] Worked numeric example (W7T)
> **Given 4 correspondences:**
> $$\{(2,2),(2,4),(6,4),(6,2)\} \to \{(-2,-2),(-1,-4),(-6,-1),(-6,-5)\}$$
>
> **$\mathbf{A}$ is $8\times9$** (2 rows per point, 4 points).
>
> **DLT solution** (`H = Vt[-1].reshape(3,3)`):
> $$\mathbf{H} \approx \begin{bmatrix}0.160 & 0.094 & -0.820\\0.146 & -0.094 & -0.418\\-0.053 & -0.016 & 0.293\end{bmatrix}$$
>
> **Verification** for point $(2,2,1)^\top$:
> $$\mathbf{H}\begin{bmatrix}2\\2\\1\end{bmatrix} \approx \begin{bmatrix}-0.312\\-0.312\\0.156\end{bmatrix} \xrightarrow{\div 0.156} \begin{bmatrix}-2\\-2\\1\end{bmatrix} \checkmark$$

---

## A6. Variations

### Homogeneous vs inhomogeneous least squares

| | DLT (homography) | Affine LS |
|---|---|---|
| System | $\mathbf{A}\mathbf{h}=\mathbf{0}$ (homogeneous) | $\mathbf{A}\mathbf{q}=\mathbf{b}$ (inhomogeneous) |
| Objective | $\min\|\mathbf{A}\mathbf{h}\|^2$ s.t. $\|\mathbf{h}\|=1$ | $\min\|\mathbf{A}\mathbf{q}-\mathbf{b}\|^2$ |
| Solution | SVD → last row of $\mathbf{V}^\top$ | Normal equations $(\mathbf{A}^\top\mathbf{A})\mathbf{q}=\mathbf{A}^\top\mathbf{b}$; pseudoinverse |
| Why different | Scale ambiguity: trivial solution $\mathbf{h}=\mathbf{0}$ must be excluded | No scale ambiguity; RHS $\mathbf{b}$ is known |
| Min matches | 4 (no 3 collinear) | 3 |

See [[affine-transformation-fitting]] for the inhomogeneous affine case.

### DLT inside RANSAC (robust homography)

Standard DLT is sensitive to outlier correspondences. In practice (e.g. panorama stitching):

1. Randomly sample **4 non-collinear** matches.
2. Estimate $\mathbf{H}$ via DLT on the 4 points.
3. Count inliers across **all** matches using a reprojection-error threshold.
4. Repeat for a fixed number of iterations; keep the $\mathbf{H}$ with the most inliers.
5. Re-run DLT on **all inliers** for a refined estimate.

See [[ransac]] for the full RANSAC framework.

### Normalised DLT

Before building $\mathbf{A}$, translate and scale each point set so that the centroid is at the origin and the mean distance to the origin is $\sqrt{2}$. Improves numerical conditioning of the SVD. Not detailed in course slides; mentioned in Hartley & Zisserman.

> [!warning] Normalised DLT not assessed in course slides — flag if seen in exam question.

### Application: stereo rectification

A rectifying homography $\mathbf{H}$ is applied to each image in a stereo pair to make epipolar lines horizontal, simplifying disparity search. See [[stereo-rectification]].

### Application: image warping and stitching

$\mathbf{H}$ is used to warp one image into the coordinate frame of another. Backward (inverse) warping with bilinear interpolation is preferred to avoid holes. See [[image-warping-and-stitching]].

---

**Related topics:** [[singular-value-decomposition]] · [[homogeneous-coordinates-and-transformations]] · [[affine-transformation-fitting]] · [[ransac]] · [[stereo-rectification]] · [[image-warping-and-stitching]]
