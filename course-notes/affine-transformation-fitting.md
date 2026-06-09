# Affine Transformation Fitting

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

Finds the 6-parameter affine mapping $\mathbf{x}' = \mathbf{M}\mathbf{x} + \mathbf{t}$ that best aligns two sets of 2D point correspondences in a least-squares sense. Use it when the scene is roughly planar, the camera is roughly orthographic, and you need to preserve parallelism but not necessarily distances or angles — e.g. initialising a more complex model, or aligning images with mild viewpoint change. Requires at least 3 non-degenerate point matches.

---

## A2. High-level overview

**Key definitions:**

| Term | Definition |
|---|---|
| $\mathbf{M}$ | $2\times2$ linear part (encodes rotation, scale, shear); entries $m_1,m_2,m_3,m_4$ |
| $\mathbf{t}$ | Translation vector $(t_1,t_2)^\top$ |
| $\mathbf{q}$ | Stacked parameter vector $[m_1,m_2,m_3,m_4,t_1,t_2]^\top$ (6 unknowns) |
| $\mathbf{A}$ | Design matrix ($2N\times6$); one $2\times6$ block per correspondence |
| $\mathbf{b}$ | Observation vector ($2N\times1$); stacked target coordinates |
| 6 dof | Affine has 6 degrees of freedom; preserves parallelism, but not lengths or angles |
| Parallelism | Parallel lines map to parallel lines (invariant under affine) |
| Normal equations | $(\mathbf{A}^\top\mathbf{A})\mathbf{q} = \mathbf{A}^\top\mathbf{b}$; the closed-form LS optimality condition |
| Moore–Penrose pseudoinverse | $\mathbf{q} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$; unique solution when $\mathbf{A}$ is full column rank |

**Steps (no math):**
1. Collect $N\geq3$ point correspondences $(\mathbf{x}_i, \mathbf{x}'_i)$.
2. For each correspondence, build a $2\times6$ block row and a $2\times1$ target block.
3. Stack all blocks into $\mathbf{A}$ ($2N\times6$) and $\mathbf{b}$ ($2N\times1$).
4. Solve the normal equations for $\mathbf{q}$ via the pseudoinverse.
5. Reshape $\mathbf{q}$ into $\mathbf{M}$ and $\mathbf{t}$; assemble the $3\times3$ homogeneous matrix $\mathbf{T}$.

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Linear system — no iteration, no initialisation needed.
- Closed-form solution ($N>3$ gives overdetermined system solved exactly via LS).
- Minimum 3 matches suffices (exact system, no redundancy needed).
- More general than rigid/similarity (handles scale and shear), less complex than homography.

**Shortcomings / limitations:**
- Preserves parallelism only; cannot model perspective foreshortening.
- Sensitive to outlier correspondences — should be wrapped in [[ransac]].
- Requires $\mathbf{A}^\top\mathbf{A}$ to be invertible (fails if all points are collinear or correspondences are degenerate).
- Assumes a planar or approximately orthographic scene; fails for strong perspective.

---

### Likely exam questions

**Q:** How many point correspondences are needed to fit an affine transformation, and why?
**A:** Minimum 3. Each correspondence gives 2 equations; 6 unknowns need at least 3 pairs ($2\times3=6$). With exactly 3 the system is exactly determined; with more it is overdetermined and solved by least squares.

---

**Q:** Write the $2\times6$ block row for a single correspondence $(x_i,y_i)\to(x_i',y_i')$ and state what $\mathbf{q}$ is.
**A:**
$$\begin{bmatrix}x_i&y_i&0&0&1&0\\0&0&x_i&y_i&0&1\end{bmatrix}\mathbf{q}=\begin{pmatrix}x_i'\\y_i'\end{pmatrix},\qquad\mathbf{q}=[m_1,m_2,m_3,m_4,t_1,t_2]^\top$$

---

**Q:** Derive the normal equations from the least-squares objective $\|\mathbf{A}\mathbf{q}-\mathbf{b}\|^2$.
**A:** Expand $(\mathbf{A}\mathbf{q}-\mathbf{b})^\top(\mathbf{A}\mathbf{q}-\mathbf{b}) = \mathbf{q}^\top\mathbf{A}^\top\mathbf{A}\mathbf{q} - 2\mathbf{q}^\top\mathbf{A}^\top\mathbf{b} + \mathbf{b}^\top\mathbf{b}$. Differentiate w.r.t. $\mathbf{q}$ and set to zero: $2\mathbf{A}^\top\mathbf{A}\mathbf{q}-2\mathbf{A}^\top\mathbf{b}=\mathbf{0}$, giving $(\mathbf{A}^\top\mathbf{A})\mathbf{q}=\mathbf{A}^\top\mathbf{b}$.

---

**Q:** Using the worked example with 4 points $X=\{(2,2),(2,4),(6,4),(6,2)\}$ and $X'=\{(-3,-1),(-1,-3),(-5,-3),(-7,-1)\}$, what is the resulting transformation matrix $\mathbf{T}$?
**A:**
$$\mathbf{T}=\begin{bmatrix}-1&1&-3\\0&-1&1\\0&0&1\end{bmatrix}$$
i.e. $m_1=-1,m_2=1,m_3=0,m_4=-1,t_1=-3,t_2=1$.

---

**Q:** What geometric property does an affine transformation preserve that a rigid transformation also preserves, and what does it lose compared to rigid?
**A:** Affine preserves parallelism (parallel lines stay parallel). It loses distance preservation and angle preservation: it can scale and shear, so lengths and angles can change. Rigid preserves both distances and angles.

---

**Q:** What is the shape of $\mathbf{A}$ and $\mathbf{b}$ for $N=4$ point correspondences? What is the shape of $\mathbf{q}$?
**A:** $\mathbf{A}$ is $8\times6$, $\mathbf{b}$ is $8\times1$, $\mathbf{q}$ is $6\times1$.

---

## A4. Directions of reasoning

### Forward / standard (fitting: correspondences → transform)

**Given:** $N\geq3$ point pairs $\{(\mathbf{x}_i,\mathbf{x}'_i)\}$.
**Asked:** $\mathbf{M}$ and $\mathbf{t}$ (equivalently $\mathbf{q}$).
**Key inference:** Build $\mathbf{A}$ and $\mathbf{b}$; solve $\mathbf{q}=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$; read off $\mathbf{M}=\bigl[\begin{smallmatrix}m_1&m_2\\m_3&m_4\end{smallmatrix}\bigr]$, $\mathbf{t}=(t_1,t_2)^\top$.

### Reverse / inferential (applying the transform: given M,t → map a point)

**Given:** Fitted $\mathbf{M}$ and $\mathbf{t}$ (or the $3\times3$ homogeneous $\mathbf{T}$), and a source point $\mathbf{x}=(x,y)^\top$.
**Asked:** Where does $\mathbf{x}$ map?
**Key inference:** Compute directly $\mathbf{x}'=\mathbf{M}\mathbf{x}+\mathbf{t}$, or in homogeneous form $\tilde{\mathbf{x}}'=\mathbf{T}\tilde{\mathbf{x}}$ with $\tilde{\mathbf{x}}=(x,y,1)^\top$.

> [!example] Reverse example (from W7T worked example)
> $\mathbf{T}=\bigl[\begin{smallmatrix}-1&1&-3\\0&-1&1\\0&0&1\end{smallmatrix}\bigr]$. Apply to $(2,2)^\top$:
> $$\mathbf{x}'=\begin{bmatrix}-1&1\\0&-1\end{bmatrix}\begin{pmatrix}2\\2\end{pmatrix}+\begin{pmatrix}-3\\1\end{pmatrix}=\begin{pmatrix}0\\-2\end{pmatrix}+\begin{pmatrix}-3\\1\end{pmatrix}=\begin{pmatrix}-3\\-1\end{pmatrix}$$
> Matches $X'_1=(-3,-1)$ — correct.

**Interpreting the matrix:** The columns of $\mathbf{M}$ show where the unit basis vectors map; $\mathbf{t}$ is where the origin maps. $\det(\mathbf{M})<0$ indicates a reflection component; $|\det(\mathbf{M})|$ gives the area scaling factor.

---

## A5. Standard implementation

### A5a. Setup

- **Input:** $N$ point correspondences $\{(\mathbf{x}_i,\mathbf{x}'_i)\}_{i=1}^N$ with $\mathbf{x}_i=(x_i,y_i)^\top$, $\mathbf{x}'_i=(x_i',y_i')^\top$, $N\geq3$.
- **Output:** Parameter vector $\mathbf{q}\in\mathbb{R}^6$; equivalently affine matrix $\mathbf{M}\in\mathbb{R}^{2\times2}$ and translation $\mathbf{t}\in\mathbb{R}^2$, assembled into $\mathbf{T}\in\mathbb{R}^{3\times3}$.
- **Assumption:** Points are not all collinear; $\mathbf{A}^\top\mathbf{A}$ is invertible (rank 6).
- **Notation:** $\mathbf{q}=[m_1,m_2,m_3,m_4,t_1,t_2]^\top$ where $\mathbf{M}=\bigl[\begin{smallmatrix}m_1&m_2\\m_3&m_4\end{smallmatrix}\bigr]$.

### A5b. Steps

**Step 1 — Per-correspondence block.**
For each $i=1,\ldots,N$, form the $2\times6$ block and $2\times1$ target:
$$\mathbf{A}_i = \begin{bmatrix}x_i&y_i&0&0&1&0\\0&0&x_i&y_i&0&1\end{bmatrix}, \qquad \mathbf{b}_i=\begin{pmatrix}x_i'\\y_i'\end{pmatrix}$$

**Step 2 — Stack.**
Concatenate vertically to form the full system:
$$\mathbf{A}=\begin{bmatrix}\mathbf{A}_1\\\mathbf{A}_2\\\vdots\\\mathbf{A}_N\end{bmatrix}\in\mathbb{R}^{2N\times6},\qquad\mathbf{b}=\begin{bmatrix}\mathbf{b}_1\\\mathbf{b}_2\\\vdots\\\mathbf{b}_N\end{bmatrix}\in\mathbb{R}^{2N}$$

So $\mathbf{A}\mathbf{q}=\mathbf{b}$ is the overdetermined linear system ($N>3$).

**Step 3 — Formulate least-squares objective.**
$$\min_{\mathbf{q}}\;\|\mathbf{A}\mathbf{q}-\mathbf{b}\|^2$$

**Step 4 — Expand the objective.**
Using $\|\mathbf{u}\|^2=\mathbf{u}^\top\mathbf{u}$ and $(uv)^\top=v^\top u^\top$:
$$\|\mathbf{A}\mathbf{q}-\mathbf{b}\|^2 = \mathbf{q}^\top\mathbf{A}^\top\mathbf{A}\,\mathbf{q} - 2\mathbf{q}^\top\mathbf{A}^\top\mathbf{b} + \mathbf{b}^\top\mathbf{b}$$

**Step 5 — Differentiate and set to zero.**
$$\frac{\partial}{\partial\mathbf{q}}\|\mathbf{A}\mathbf{q}-\mathbf{b}\|^2 = 2\mathbf{A}^\top\mathbf{A}\,\mathbf{q} - 2\mathbf{A}^\top\mathbf{b} = \mathbf{0}$$

**Normal equations:**
$$(\mathbf{A}^\top\mathbf{A})\,\mathbf{q} = \mathbf{A}^\top\mathbf{b}$$

**Step 6 — Solve via Moore–Penrose pseudoinverse.**
$$\mathbf{q} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$$

Python — two equivalent methods:
```python
# Method 1: pinv
q = np.linalg.pinv(A) @ b

# Method 2: normal equations
q = np.linalg.inv(A.T @ A) @ (A.T @ b)
```

**Step 7 — Assemble the transformation matrix.**
$$\mathbf{T}=\begin{bmatrix}m_1&m_2&t_1\\m_3&m_4&t_2\\0&0&1\end{bmatrix}$$

> [!example] Worked example (W7T, 4 points)
> **Source:** $X=\{(2,2),(2,4),(6,4),(6,2)\}$; **Target:** $X'=\{(-3,-1),(-1,-3),(-5,-3),(-7,-1)\}$
>
> Design matrix $\mathbf{A}$ ($8\times6$) and $\mathbf{b}$ ($8\times1$):
> $$\mathbf{b}=\begin{bmatrix}-3\\-1\\-1\\-3\\-5\\-3\\-7\\-1\end{bmatrix},\quad
> \mathbf{A}=\begin{bmatrix}2&2&0&0&1&0\\0&0&2&2&0&1\\2&4&0&0&1&0\\0&0&2&4&0&1\\6&4&0&0&1&0\\0&0&6&4&0&1\\6&2&0&0&1&0\\0&0&6&2&0&1\end{bmatrix}$$
>
> Solution: $\mathbf{q}=[-1,1,0,-1,-3,1]^\top$ (note $m_3\approx4.45\times10^{-16}\approx0$ in numpy output)
>
> $$\mathbf{T}=\begin{bmatrix}-1&1&-3\\0&-1&1\\0&0&1\end{bmatrix}$$

---

## A6. Variations

### Rigid transformation (Procrustes) — fewer dof — [[rigid-transformation-procrustes]]

**Change:** Restrict $\mathbf{M}$ to be a rotation matrix $\mathbf{R}$ (orthogonal, $\det=+1$); 3 dof instead of 6.
**Consequence:** Cannot model scale or shear. The LS problem is no longer linear in the unknowns (orthogonality constraint makes it nonlinear). Solved by the Procrustes procedure: centre both sets, recover $\mathbf{R}$ via SVD of the cross-covariance matrix $\widetilde{X}^\top\widetilde{X}'$, then solve for $\mathbf{t}$.
**When to use:** When the transformation is known to be a rigid body motion (distances preserved); minimum 2 correspondences.

### Homography / DLT — more dof — [[homography-and-dlt]]

**Change:** Replace the $2\times2$ affine matrix with a full $3\times3$ projective matrix $\mathbf{H}$ (8 dof, 9 parameters up to scale); points in [[homogeneous-coordinates-and-transformations]] are used.
**Consequence:** The system is **homogeneous** ($\mathbf{A}\mathbf{h}=\mathbf{0}$, not $\mathbf{A}\mathbf{q}=\mathbf{b}$) because $\mathbf{H}$ is only defined up to scale. Solved via SVD of $\mathbf{A}$ ($2N\times9$); solution is the last row of $\mathbf{V}^\top$ (right singular vector for smallest singular value). See [[singular-value-decomposition]]. Minimum 4 non-collinear matches.
**When to use:** Planar scene viewed under full perspective, or two cameras sharing the same optical centre (panorama stitching). Preserves straight lines but not parallelism, angles, or distances.

> [!note] Comparison table
> | Property | Rigid (Procrustes) | **Affine** | Homography (DLT) |
> |---|---|---|---|
> | dof | 3 | **6** | 8 |
> | Distances preserved | Yes | No | No |
> | Angles preserved | Yes | No | No |
> | Parallelism preserved | Yes | **Yes** | No |
> | Straight lines preserved | Yes | Yes | Yes |
> | Min. correspondences | 2 | **3** | 4 |
> | LS system type | nonlinear (SVD trick) | **inhomogeneous** $\mathbf{Aq}=\mathbf{b}$ | homogeneous $\mathbf{Ah}=\mathbf{0}$ |
> | Solver | SVD of cross-covariance | **normal equations / pinv** | SVD of $\mathbf{A}$ |

### Affine fitting within RANSAC — [[ransac]]

**Change:** Wrap the above fitting procedure inside the RANSAC loop. At each iteration randomly sample 3 correspondences (minimal set), fit $\mathbf{T}$, count inliers; after the loop re-fit on all inliers.
**When to use:** When putative correspondences contain outliers (typical in feature-matching pipelines).

> [!note] Notation mapping from slides
> W7T uses $m_{11},m_{12},m_{21},m_{22}$ for the four entries of $\mathbf{M}$, whereas W4L_pt2 uses $m_1,m_2,m_3,m_4$. Both refer to the same quantities in row-major order: $m_1=m_{11}$, $m_2=m_{12}$, $m_3=m_{21}$, $m_4=m_{22}$.
