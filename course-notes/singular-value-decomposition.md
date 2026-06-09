# Singular Value Decomposition (SVD)

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Definitions]]
  - [[#Steps (no math)]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps — 8-step compute procedure]]
- [[#A6. Variations]]
  - [[#Homogeneous least squares (solving $\mathbf{A}\mathbf{h}=\mathbf{0}$)]]
  - [[#Orthogonal Procrustes rotation recovery]]

---

## A1. Purpose

SVD decomposes any $m \times n$ real matrix $\mathbf{A}$ into three structured factors $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$ whose columns/rows are orthonormal and whose diagonal captures the "energy" of $\mathbf{A}$ in decreasing order. In this course it is used for two critical tasks: **(1)** solving the homogeneous constrained least-squares problem $\min\|\mathbf{A}\mathbf{h}\|$ s.t. $\|\mathbf{h}\|=1$ (needed for DLT homography fitting and camera calibration), and **(2)** recovering the optimal rotation in the Orthogonal Procrustes problem (rigid transformation fitting). Both arise wherever geometry must be estimated from noisy point correspondences.

---

## A2. High-level overview

### Definitions

| Term | Definition |
|---|---|
| **Singular values** $\sigma_i$ | Non-negative square roots of eigenvalues of $\mathbf{A}\mathbf{A}^\top$; ordered $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $\mathbf{U}$ ($m \times m$) | Orthogonal matrix; columns are the orthonormal eigenvectors of $\mathbf{A}\mathbf{A}^\top$ (left singular vectors) |
| $\mathbf{L}$ ($m \times n$) | "Diagonal" matrix with $\sigma_i$ on the main diagonal, zeros elsewhere (same shape as $\mathbf{A}$) |
| $\mathbf{V}$ ($n \times n$) | Orthogonal matrix; columns are the orthonormal eigenvectors of $\mathbf{A}^\top\mathbf{A}$ (right singular vectors) |
| **Orthogonal vectors** | Two vectors $\mathbf{u}$, $\mathbf{v}$ with $\mathbf{u} \cdot \mathbf{v} = 0$ (perpendicular) |
| **Orthonormal vectors** | Orthogonal AND each has unit magnitude: $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ |
| **Orthogonal matrix** | Square matrix with $\mathbf{U}^\top\mathbf{U} = \mathbf{U}\mathbf{U}^\top = \mathbf{I}$; inverse equals transpose |

> [!note]
> The slides use $\mathbf{L}$ for the singular-value diagonal matrix throughout. The standard textbook notation is $\mathbf{\Sigma}$ — both denote the same $m \times n$ matrix with $\sigma_i$ on the diagonal.

### Steps (no math)

1. Compute $\mathbf{A}\mathbf{A}^\top$ ($m \times m$).
2. Find eigenvalues of $\mathbf{A}\mathbf{A}^\top$; sort descending.
3. Compute singular values $\sigma_i = \sqrt{\lambda_i}$; form $\mathbf{L}$.
4. Find eigenvectors of $\mathbf{A}\mathbf{A}^\top$; normalize each.
5. Assemble as columns of $\mathbf{U}$ (matching eigenvalue order).
6. Compute $\mathbf{A}^\top\mathbf{A}$ ($n \times n$).
7. Find eigenvectors of $\mathbf{A}^\top\mathbf{A}$; normalize; assemble as columns of $\mathbf{V}$.
8. Verify: $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Works on any $m \times n$ matrix (square or not).
- Singular values are always real and non-negative regardless of matrix entries.
- Positive eigenvalues of $\mathbf{A}\mathbf{A}^\top$ and $\mathbf{A}^\top\mathbf{A}$ are identical — allows computing $\mathbf{V}$ efficiently.
- Directly yields the solution to constrained homogeneous least squares (last row of $\mathbf{V}^\top$) without matrix inversion.
- Guaranteed to give a proper rotation in Procrustes (with the $\mathbf{D}$ correction) even when the naive cross-covariance gives a reflection.

**Shortcomings / limitations**
- Computationally expensive for large matrices (not done by hand beyond small examples).
- Sign of singular vectors is arbitrary; eigenvector directions can flip between implementations — the $\mathbf{D}$ correction in Procrustes handles the rotation case but not all uses.
- Hand computation is error-prone for anything larger than $3 \times 2$; in practice always use `np.linalg.svd`.
- $\mathbf{L}$ is $m \times n$, not square — easy to get the dimension wrong when assembling the product.

---

### Likely exam questions

**Q:** What are the dimensions of $\mathbf{U}$, $\mathbf{L}$, and $\mathbf{V}$ for an $m \times n$ matrix $\mathbf{A}$, and which matrix product is used to obtain each?
**A:** $\mathbf{U}$ is $m \times m$ (eigenvectors of $\mathbf{A}\mathbf{A}^\top$); $\mathbf{L}$ is $m \times n$ (singular values on diagonal); $\mathbf{V}$ is $n \times n$ (eigenvectors of $\mathbf{A}^\top\mathbf{A}$).

---

**Q:** For $\mathbf{A} = \begin{pmatrix}1&-1&3\\3&1&1\end{pmatrix}$, compute $\mathbf{A}\mathbf{A}^\top$ and hence the singular values.
**A:**
$$\mathbf{A}\mathbf{A}^\top = \begin{pmatrix}11&5\\5&11\end{pmatrix}$$
Eigenvalues: $\lambda_1 = 16$, $\lambda_2 = 6$. Singular values: $\sigma_1 = \sqrt{16} = 4$, $\sigma_2 = \sqrt{6}$.

---

**Q:** How do you solve $\min_{\mathbf{h}} \|\mathbf{A}\mathbf{h}\|$ subject to $\|\mathbf{h}\|=1$ using SVD?
**A:** Compute SVD of $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$. The solution is the last row of $\mathbf{V}^\top$ (equivalently, the last column of $\mathbf{V}$) — the right singular vector corresponding to the smallest singular value.

---

**Q:** In the Procrustes problem, why is $\mathbf{D} = \operatorname{diag}(1, \det(\mathbf{U}\mathbf{V}^\top))$ inserted into $\mathbf{R} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$?
**A:** SVD can produce a reflection ($\det(\mathbf{U}\mathbf{V}^\top) = -1$) when the optimal solution is a rotation. The $\mathbf{D}$ matrix corrects for this, ensuring $\det(\mathbf{R}) = +1$ (proper rotation, not a reflection).

---

**Q:** For the full SVD of $\mathbf{A} = \begin{pmatrix}0&1\\1&1\\1&0\end{pmatrix}$, write down $\mathbf{U}$, $\mathbf{L}$, and $\mathbf{V}$.
**A:** (See the worked example in A5b below.) $\mathbf{L} = \begin{pmatrix}\sqrt{3}&0\\0&1\\0&0\end{pmatrix}$;
$$\mathbf{U} = \begin{pmatrix}1/\sqrt{6}&1/\sqrt{2}&1/\sqrt{3}\\2/\sqrt{6}&0&-1/\sqrt{3}\\1/\sqrt{6}&-1/\sqrt{2}&1/\sqrt{3}\end{pmatrix}, \quad \mathbf{V} = \begin{pmatrix}1/\sqrt{2}&1/\sqrt{2}\\1/\sqrt{2}&-1/\sqrt{2}\end{pmatrix}$$

---

**Q:** Are the singular values of $\mathbf{A}$ the same as the eigenvalues of $\mathbf{A}$? Explain.
**A:** No. Singular values are $\sigma_i = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^\top)}$ — square roots of eigenvalues of $\mathbf{A}\mathbf{A}^\top$ (always real, always $\geq 0$). Eigenvalues of $\mathbf{A}$ can be negative, complex, and are only defined for square matrices.

---

## A4. Directions of reasoning

### Forward / standard: $\mathbf{A} \to \mathbf{U}, \mathbf{L}, \mathbf{V}$

**Given:** matrix $\mathbf{A}$ ($m \times n$).
**Asked:** factors $\mathbf{U}$, $\mathbf{L}$, $\mathbf{V}^\top$ such that $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$.
**Key steps:** form $\mathbf{A}\mathbf{A}^\top$, solve for eigenvalues/eigenvectors, normalize, assemble columns; repeat for $\mathbf{A}^\top\mathbf{A}$; diagonal of $\mathbf{L}$ is $\{\sqrt{\lambda_i}\}$ in descending order.

### Reverse / inferential — smallest singular vector as solution

**Given:** overdetermined homogeneous system $\mathbf{A}\mathbf{h} = \mathbf{0}$ with constraint $\|\mathbf{h}\|=1$.
**Asked:** which vector $\mathbf{h}$ minimises $\|\mathbf{A}\mathbf{h}\|$?
**Key inference:** $\|\mathbf{A}\mathbf{h}\|^2 = \mathbf{h}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{h}$, so minimising this over the unit sphere selects the eigenvector of $\mathbf{A}^\top\mathbf{A}$ with the smallest eigenvalue — equivalently the last row of $\mathbf{V}^\top$ in the SVD of $\mathbf{A}$.

### Reverse / inferential — rotation from SVD factors

**Given:** SVD factors $\mathbf{U}$, $\mathbf{L}$, $\mathbf{V}^\top$ of the cross-covariance $\mathbf{H} = \widetilde{\mathbf{X}}^\top \widetilde{\mathbf{X}}'$.
**Asked:** optimal rotation $\mathbf{R}$.
**Key inference:** $\mathbf{R} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ where $\mathbf{D} = \operatorname{diag}(1, \det(\mathbf{U}\mathbf{V}^\top))$. When $\det = 1$, $\mathbf{D} = \mathbf{I}$. When $\det = -1$, the last column of $\mathbf{U}$ is effectively negated, converting a reflection into a rotation.

---

## A5. Standard implementation

### A5a. Setup

| Item | Detail |
|---|---|
| **Input** | Real matrix $\mathbf{A}$, size $m \times n$ |
| **Output** | $\mathbf{U}$ ($m\times m$, orthogonal), $\mathbf{L}$ ($m\times n$, diagonal with $\sigma_i \geq 0$), $\mathbf{V}$ ($n\times n$, orthogonal) such that $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$ |
| **Constraints** | $\mathbf{U}^\top\mathbf{U} = \mathbf{I}_m$; $\mathbf{V}^\top\mathbf{V} = \mathbf{I}_n$; $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| **Notation** | $\lambda_i$ = eigenvalue; $\hat{\mathbf{x}}$ = normalized eigenvector; $\|\mathbf{x}\| = \sqrt{x_1^2 + \cdots + x_n^2}$ |

### A5b. Steps — 8-step compute procedure

1. **Compute $\mathbf{A}\mathbf{A}^\top$** — produces an $m \times m$ symmetric positive semi-definite matrix.

2. **Find eigenvalues of $\mathbf{A}\mathbf{A}^\top$** — solve $\det(\mathbf{A}\mathbf{A}^\top - \lambda\mathbf{I}) = 0$; sort so $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$.

3. **Compute singular values** — $\sigma_i = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^\top)}$ for each $i$; assemble $\mathbf{L}$ ($m \times n$) with $\sigma_i$ on the diagonal.

4. **Find eigenvectors of $\mathbf{A}\mathbf{A}^\top$** — for each $\lambda_i$ solve $(\mathbf{A}\mathbf{A}^\top - \lambda_i\mathbf{I})\mathbf{x} = \mathbf{0}$; get raw eigenvector $\mathbf{x}_i$.

5. **Normalize eigenvectors** — $\hat{\mathbf{x}}_i = \mathbf{x}_i / \|\mathbf{x}_i\|$; these are the columns of $\mathbf{U}$ in eigenvalue order.

6. **Assemble $\mathbf{U}$** — $\mathbf{U} = [\hat{\mathbf{x}}_1 \mid \hat{\mathbf{x}}_2 \mid \cdots \mid \hat{\mathbf{x}}_m]$ (each column is a left singular vector).

7. **Compute $\mathbf{A}^\top\mathbf{A}$** ($n \times n$), find its eigenvectors (the positive eigenvalues are the same as those of $\mathbf{A}\mathbf{A}^\top$), normalize them, and assemble $\mathbf{V}$ in the same eigenvalue order.

8. **Verify** — confirm $\mathbf{U}\mathbf{L}\mathbf{V}^\top = \mathbf{A}$ (optional sanity check; use `np.linalg.svd` in practice).

---

> [!example]
> ### Full worked example: 3×2 matrix (from W5T)
>
> $$\mathbf{A} = \begin{pmatrix}0&1\\1&1\\1&0\end{pmatrix}$$
>
> **Step 1 — Compute $\mathbf{A}\mathbf{A}^\top$ (3×3):**
> $$\mathbf{A}\mathbf{A}^\top = \begin{pmatrix}1&1&0\\1&2&1\\0&1&1\end{pmatrix}$$
>
> **Step 2 — Eigenvalues of $\mathbf{A}\mathbf{A}^\top$:**
> $$\det(\mathbf{A}\mathbf{A}^\top - \lambda\mathbf{I}) = 0 \implies (3-\lambda)(1-\lambda)\lambda = 0$$
> $$\lambda_1 = 3, \quad \lambda_2 = 1, \quad \lambda_3 = 0$$
>
> **Step 3 — Singular values; form $\mathbf{L}$ (3×2):**
> $$\sigma_1 = \sqrt{3}, \quad \sigma_2 = \sqrt{1} = 1, \quad \sigma_3 = 0$$
> $$\mathbf{L} = \begin{pmatrix}\sqrt{3}&0\\0&1\\0&0\end{pmatrix}$$
>
> **Step 4 — Raw eigenvectors of $\mathbf{A}\mathbf{A}^\top$:**
> $$\mathbf{x}_1 = \begin{pmatrix}1\\2\\1\end{pmatrix}, \quad \mathbf{x}_2 = \begin{pmatrix}1\\0\\-1\end{pmatrix}, \quad \mathbf{x}_3 = \begin{pmatrix}1\\-1\\1\end{pmatrix}$$
>
> **Step 5 — Normalize:**
> $$\hat{\mathbf{x}}_1 = \begin{pmatrix}1/\sqrt{6}\\2/\sqrt{6}\\1/\sqrt{6}\end{pmatrix}, \quad \hat{\mathbf{x}}_2 = \begin{pmatrix}1/\sqrt{2}\\0\\-1/\sqrt{2}\end{pmatrix}, \quad \hat{\mathbf{x}}_3 = \begin{pmatrix}1/\sqrt{3}\\-1/\sqrt{3}\\1/\sqrt{3}\end{pmatrix}$$
>
> **Step 6 — Assemble $\mathbf{U}$ (3×3):**
> $$\mathbf{U} = \begin{pmatrix}1/\sqrt{6}&1/\sqrt{2}&1/\sqrt{3}\\2/\sqrt{6}&0&-1/\sqrt{3}\\1/\sqrt{6}&-1/\sqrt{2}&1/\sqrt{3}\end{pmatrix}$$
>
> **Step 7 — Compute $\mathbf{A}^\top\mathbf{A}$ (2×2); find eigenvectors; assemble $\mathbf{V}$ (2×2):**
> $$\mathbf{A}^\top\mathbf{A} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$$
> Eigenvalues: $\lambda_1=3$, $\lambda_2=1$. Raw eigenvectors: $\begin{pmatrix}1\\1\end{pmatrix}$ and $\begin{pmatrix}1\\-1\end{pmatrix}$.
> $$\mathbf{V} = \begin{pmatrix}1/\sqrt{2}&1/\sqrt{2}\\1/\sqrt{2}&-1/\sqrt{2}\end{pmatrix}$$
>
> **Step 8 — Final result:**
> $$\mathbf{A} = \begin{pmatrix}0&1\\1&1\\1&0\end{pmatrix} = \underbrace{\begin{pmatrix}1/\sqrt{6}&1/\sqrt{2}&1/\sqrt{3}\\2/\sqrt{6}&0&-1/\sqrt{3}\\1/\sqrt{6}&-1/\sqrt{2}&1/\sqrt{3}\end{pmatrix}}_{\mathbf{U}}\underbrace{\begin{pmatrix}\sqrt{3}&0\\0&1\\0&0\end{pmatrix}}_{\mathbf{L}}\underbrace{\begin{pmatrix}1/\sqrt{2}&1/\sqrt{2}\\1/\sqrt{2}&-1/\sqrt{2}\end{pmatrix}^\top}_{\mathbf{V}^\top}$$

---

## A6. Variations

### Homogeneous least squares (solving $\mathbf{A}\mathbf{h}=\mathbf{0}$)

**Context:** Used in DLT (see [[homography-and-dlt]] and [[camera-calibration-dlt]]) where the system $\mathbf{A}\mathbf{h} = \mathbf{0}$ has more equations than unknowns and noise means no exact zero exists.

**Change to standard SVD:** no modification to the SVD computation itself; the only difference is how the output is used.

**Steps:**
1. Assemble $\mathbf{A}$ from correspondences (e.g. $8 \times 9$ for 4-point DLT).
2. Compute $\mathbf{A} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$.
3. Take the **last row of $\mathbf{V}^\top$** (= last column of $\mathbf{V}$) as $\hat{\mathbf{h}}$.
4. Reshape if needed (e.g. $1 \times 9 \to 3 \times 3$ for a homography).

> [!note]
> Why last row of $\mathbf{V}^\top$? Because $\|\mathbf{A}\mathbf{h}\|^2 = \mathbf{h}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{h}$ is minimised by the eigenvector of $\mathbf{A}^\top\mathbf{A}$ with the **smallest** eigenvalue, which appears as the last row of $\mathbf{V}^\top$ since singular values are sorted in descending order. Equivalently: the right singular vector corresponding to $\sigma_\text{min}$.

**Python (NumPy):**
```python
u, s, vt = np.linalg.svd(A)
h = vt[-1]          # last row of V^T
H = h.reshape(3, 3) # for homography
```

---

### Orthogonal Procrustes rotation recovery

**Context:** Used in [[rigid-transformation-procrustes]] to recover the rotation $\mathbf{R}$ between two centred point sets. Also the mathematical core of [[camera-calibration-dlt]] pose recovery.

**Problem:** given centred point matrices $\widetilde{\mathbf{X}}$ ($N \times 2$) and $\widetilde{\mathbf{X}}'$ ($N \times 2$), find orthogonal $\mathbf{R}$ minimising
$$\min_{\mathbf{R}} \|\widetilde{\mathbf{X}}\mathbf{R} - \widetilde{\mathbf{X}}'\|_F \quad \text{s.t.} \quad \mathbf{R}^\top\mathbf{R} = \mathbf{I}$$
(equivalently, maximise $\operatorname{tr}(\mathbf{R}^\top \widetilde{\mathbf{X}}^\top \widetilde{\mathbf{X}}')$).

**Steps:**
1. Form cross-covariance: $\mathbf{H} = \widetilde{\mathbf{X}}^\top \widetilde{\mathbf{X}}'$ ($2 \times 2$).
2. SVD: $\mathbf{H} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$ with $\sigma_1 \geq \sigma_2 \geq 0$.
3. Compute rotation with reflection correction:
   $$\mathbf{R} = \mathbf{U}\mathbf{D}\mathbf{V}^\top, \quad \mathbf{D} = \operatorname{diag}\!\left(1,\, \det(\mathbf{U}\mathbf{V}^\top)\right)$$
4. Compute translation: $\mathbf{t} = \bar{\mathbf{x}}' - \mathbf{R}\bar{\mathbf{x}}$.

> [!note]
> The slide and tutorial use the notation $\mathbf{H} = \mathbf{U}\mathbf{L}\mathbf{V}^\top$ for this SVD (same $\mathbf{L}$ notation as above). $\det(\mathbf{U}\mathbf{V}^\top) = +1$ → pure rotation (clockwise); $-1$ → would-be reflection, corrected to rotation (anti-clockwise interpretation in 2D).

> [!example]
> ### Worked example: recover $\mathbf{R}$ and $\mathbf{t}$ (from W5T/W6T)
>
> **Given:**
> $A = \{(2,2),(2,4),(5,6)\}$, $B = \{(-3,-3),(-3,-5),(-6,-7)\}$
> (B generated by rotating A by $\pi$ then translating by $(-1,-1)^\top$)
>
> **Step 1 — Means and centring:**
> $$\bar{\mathbf{x}}_A = (3,4), \quad \bar{\mathbf{x}}_B = (-4,-5)$$
> $$\widetilde{A} = \{(-1,-2),(-1,0),(2,2)\}, \quad \widetilde{B} = \{(1,2),(1,0),(-2,-2)\}$$
>
> **Step 2 — Cross-covariance:**
> $$\mathbf{H} = \widetilde{A}^\top\widetilde{B} = \begin{pmatrix}-6&-6\\-8&-8\end{pmatrix}$$
>
> **SVD (via `np.linalg.svd`):**
> $$\mathbf{U} = \begin{pmatrix}-0.6464&-0.7630\\-0.7630&0.6464\end{pmatrix}, \quad \mathbf{V} = \begin{pmatrix}0.6464&0.7630\\0.7630&0.6464\end{pmatrix}$$
>
> **Rotation ($\det(\mathbf{U}\mathbf{V}^\top) = 1 \Rightarrow \mathbf{D} = \mathbf{I}$):**
> $$\mathbf{R} = \mathbf{U}\mathbf{V}^\top = \begin{pmatrix}-1&0\\0&-1\end{pmatrix} \quad (180° \text{ rotation, as expected})$$
>
> **Step 3 — Translation:**
> $$\mathbf{t} = \begin{pmatrix}-4\\-5\end{pmatrix} - \begin{pmatrix}-1&0\\0&-1\end{pmatrix}\begin{pmatrix}3\\4\end{pmatrix} = \begin{pmatrix}-4\\-5\end{pmatrix} - \begin{pmatrix}-3\\-4\end{pmatrix} = \begin{pmatrix}-1\\-1\end{pmatrix} \checkmark$$

---

*Related topics:* [[eigenvalues-eigenvectors]] · [[homography-and-dlt]] · [[rigid-transformation-procrustes]] · [[camera-calibration-dlt]]
