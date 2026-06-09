# Homogeneous Coordinates and Transformations

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps — 2D affine transforms]]
  - [[#b2. Steps — 3D affine transforms]]
  - [[#b3. Steps — WRF ↔ CRF (extrinsic transform)]]
- [[#A6. Variations — 2D transformation model hierarchy]]

---

## A1. Purpose

Homogeneous coordinates embed $n$-dimensional Euclidean space into $(n{+}1)$-dimensional projective space, so that **translations, affine transforms, and perspective projection all become matrix multiplications** (no special-casing required). They are the universal currency for geometric operations in computer vision: encoding 2D/3D rigid-body motion, the extrinsic $(\mathbf{R}\mid\mathbf{t})$ matrix, and the full projection chain $\mathbf{P} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]$.

---

## A2. High-level overview

1. Represent a Cartesian point in homogeneous form by appending a scale $w$ (usually $w=1$).
2. Apply the desired transform as a single matrix multiply.
3. Recover the Cartesian result by dividing by the last component and dropping it.

### Key definitions

| Term | Definition |
|---|---|
| **Homogeneous coordinate** | Extra scale coordinate $w$ appended; point defined up to a non-zero scalar multiple. |
| **1D homogeneous** | Cartesian $x$ ↔ any $\begin{bmatrix}x_h \\ w\end{bmatrix}$ with $x = x_h/w$. |
| **2D homogeneous** | Cartesian $(x,y)$ ↔ $\tilde{\mathbf{x}} = (x,y,1)^\top$ (normalised); general form $(x_h, y_h, w)^\top$ with recovery $(x_h/w,\, y_h/w)$. |
| **3D homogeneous** | Cartesian $(x,y,z)$ ↔ $\tilde{\mathbf{X}} = (x,y,z,1)^\top$; general recovery divides by 4th component. |
| **Projective space $\mathcal{P}^2$** | $\mathbb{R}^2$ extended so parallel lines meet at points at infinity (where $w=0$). |
| **Affine transformation** | Linear map + translation; preserves parallelism. Expressed as a single homogeneous matrix. |
| **WRF** | World Reference Frame — coordinates $(X,Y,Z)$ in mm/cm. |
| **CRF** | Camera Reference Frame — coordinates $(x,y,z)$ in mm, centred at optical centre. |
| **Extrinsic matrix $(\mathbf{R}\mid\mathbf{t})$** | $4{\times}4$ homogeneous matrix mapping WRF → CRF; contains $3{\times}3$ rotation $\mathbf{R}$ and $3{\times}1$ translation $\mathbf{t}$. |
| **Rotation matrix $\mathbf{R}$** | Orthonormal ($\mathbf{R}^\top\mathbf{R}=\mathbf{I}$, $\det\mathbf{R}=+1$); its inverse equals its transpose. |
| **Right-hand rule** | $\mathbf{r}_1 \times \mathbf{r}_2 = \mathbf{r}_3$ for the columns of $\mathbf{R}$. |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Unifies translation, rotation, scale, shear, and perspective projection into a single matrix multiply.
- Composition of transforms = matrix multiplication (apply from right to left).
- Clean formula for inverse of extrinsic: exploits $\mathbf{R}^{-1} = \mathbf{R}^\top$.
- Projective transforms (homography) naturally expressed — parallel lines and points at infinity are first-class objects.

**Shortcomings / limitations**
- Division by $w$ is undefined when $w=0$ (points at infinity; valid in projective sense, not Cartesian).
- Rotation matrices have 9 entries but only 3 dof (SO(3)) — redundant parameterisation prone to numerical drift; must re-orthonormalise after many accumulated multiplications.
- Composing many transforms accumulates floating-point error.
- Gimbal lock occurs with Euler-angle sequences (not a homogeneous-coordinates issue per se, but relevant to $\mathbf{R}_x\mathbf{R}_y\mathbf{R}_z$ chains).

---

### Likely exam questions

**Q:** Write the $3{\times}3$ homogeneous matrix for a 2D **rotation** of $90°$ counter-clockwise, and apply it to the point $(7,8)^\top$.

**A:**
$$\mathbf{R}_{90} = \begin{pmatrix}0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1\end{pmatrix}, \qquad \mathbf{R}_{90}\begin{pmatrix}7\\8\\1\end{pmatrix} = \begin{pmatrix}-8\\7\\1\end{pmatrix}$$
Result in Cartesian: $(-8, 7)^\top$.

---

**Q:** Convert the homogeneous point $(12, 24, 9, 3)^\top$ to Cartesian coordinates.

**A:** Divide by $w=3$: $(12/3,\; 24/3,\; 9/3) = (4, 8, 3)^\top$.

---

**Q:** What are the 4 properties that the columns of a rotation matrix must satisfy?

**A:** (i) $\mathbf{r}_1\cdot\mathbf{r}_2 = 0$; (ii) $\mathbf{r}_1\cdot\mathbf{r}_3 = 0$; (iii) $\mathbf{r}_2\cdot\mathbf{r}_3 = 0$ (pairwise orthogonality); (iv) $|\mathbf{r}_1|=|\mathbf{r}_2|=|\mathbf{r}_3|=1$ (unit length). Equivalently $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$. The additional constraint $\mathbf{r}_1\times\mathbf{r}_2=\mathbf{r}_3$ (right-hand rule) ensures $\det\mathbf{R}=+1$ (proper rotation, not reflection).

---

**Q:** Write the formula for $(\mathbf{R}\mid\mathbf{t})^{-1}$ and state what it does geometrically.

**A:**
$$(\mathbf{R}\mid\mathbf{t})^{-1} = \begin{pmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t} \\ \mathbf{0}^\top & 1\end{pmatrix}$$
It maps CRF back to WRF. The upper-left $\mathbf{R}^\top$ undoes the rotation; the upper-right $-\mathbf{R}^\top\mathbf{t}$ undoes the translation expressed in the rotated frame.

---

**Q:** Why do translations require homogeneous coordinates but simple rotations and scaling do not?

**A:** Rotation and scaling are **linear maps** expressible as ordinary (Cartesian) matrix multiplication. Translation is an **affine** operation — adding a constant — which cannot be expressed as a matrix multiply on Cartesian vectors. Homogeneous coordinates embed the point in one higher dimension, turning the additive translation into the final column of the transform matrix, so the whole affine map collapses into a single multiply.

---

**Q:** Name the 4 levels of the 2D transformation hierarchy in order of increasing dof, and state what geometric property is preserved at each level.

**A:**

| Model | dof | Preserved property |
|---|---|---|
| Rigid (Euclidean) | 3 | Distances and angles |
| Similarity | 4 | Angles (not distances) |
| Affine | 6 | Parallelism |
| Projective (Homography) | 8 | Collinearity (straight lines stay straight) |

---

## A4. Directions of reasoning

### Forward / standard (Cartesian → homogeneous → transformed → Cartesian)

**Given:** A Cartesian point $\mathbf{p} = (x,y)^\top$ and a sequence of 2D transforms (e.g. translate then rotate).
**Asked:** Transformed Cartesian point.
**Key step:** Append 1 → multiply left-to-right by each transform matrix (right-to-left in composition) → divide result by $w$ and drop it.

### Inverse / CRF → WRF

**Given:** Extrinsic matrix $(\mathbf{R}\mid\mathbf{t})$ (WRF→CRF) and a point in CRF.
**Asked:** Corresponding point in WRF.
**Key inference:** Use $(\mathbf{R}\mid\mathbf{t})^{-1} = \bigl(\begin{smallmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t}\end{smallmatrix}\bigr)$. No explicit matrix inversion needed — exploit $\mathbf{R}^{-1}=\mathbf{R}^\top$ from orthonormality.

### Reverse / recover transform from correspondences

**Given:** Point pairs $\{\mathbf{x}_i, \mathbf{x}'_i\}$.
**Asked:** Which transform model fits?
**Key inference:** Count dof vs. number of pairs; choose the least-complex model sufficient (see hierarchy in A6). See [[rigid-transformation-procrustes]] and [[affine-transformation-fitting]] for the fitting algorithms.

---

## A5. Standard implementation

### a. Setup

| Item | Detail |
|---|---|
| Input | A point $\mathbf{p} \in \mathbb{R}^n$ (Cartesian); one or more transform parameters |
| Output | Transformed point in Cartesian |
| Assumption | Homogeneous $w\neq 0$ after transform (no points at infinity) |
| Convention | Column vectors; matrices applied on the left; $C = \cos\theta$, $S = \sin\theta$ |

---

### b. Steps — 2D affine transforms

**Step 1. Convert to homogeneous:** $(x,y)^\top \to \tilde{\mathbf{p}} = (x,y,1)^\top$.

**Step 2. Choose / compose the $3{\times}3$ matrix:**

**Translation** by $(t_x, t_y)$:
$$\mathbf{T} = \begin{pmatrix}1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1\end{pmatrix}$$

**Scaling** by $(\alpha, \beta)$:
$$\mathbf{S} = \begin{pmatrix}\alpha & 0 & 0 \\ 0 & \beta & 0 \\ 0 & 0 & 1\end{pmatrix}$$

**Rotation** anti-clockwise by $\theta$:
$$\mathbf{R} = \begin{pmatrix}C & -S & 0 \\ S & C & 0 \\ 0 & 0 & 1\end{pmatrix}, \quad C=\cos\theta,\; S=\sin\theta$$

**Shearing** with parameters $(H_x, H_y)$:
$$\mathbf{H} = \begin{pmatrix}1 & H_x & 0 \\ H_y & 1 & 0 \\ 0 & 0 & 1\end{pmatrix}$$

**Step 3. Multiply:** $\tilde{\mathbf{p}}' = \mathbf{M}\,\tilde{\mathbf{p}}$.

**Step 4. Recover Cartesian:** if $w'=1$, read off $(x', y')$; otherwise divide by $w'$.

> [!example] Worked 2D examples (from slides)
> - **Translation** $(t_x,t_y)=(2,3)$ applied to $(3,4,1)^\top$: result $(5,7,1)^\top \to (5,7)$
> - **Scaling** $(\alpha,\beta)=(2,3)$ applied to $(2,2,1)^\top$: result $(4,6,1)^\top \to (4,6)$
> - **Rotation** $\theta=90°$ ($C=0, S=1$) applied to $(7,8,1)^\top$: result $(-8,7,1)^\top \to (-8,7)$
> - **Shearing** $(H_x=3, H_y=-2)$ applied to $(2,1,1)^\top$: result $(5,-3,1)^\top \to (5,-3)$

> [!note] Key trig values
> | $\theta$ | $-90°$ | $0°$ | $90°$ | $180°$ |
> |---|---|---|---|---|
> | $\cos\theta$ | $0$ | $1$ | $0$ | $-1$ |
> | $\sin\theta$ | $-1$ | $0$ | $1$ | $0$ |

---

### b2. Steps — 3D affine transforms

All matrices are $4{\times}4$; point is $\tilde{\mathbf{P}} = (X,Y,Z,1)^\top$.

**Translation** by $(t_x,t_y,t_z)$:
$$\mathbf{T} = \begin{pmatrix}1&0&0&t_x\\0&1&0&t_y\\0&0&1&t_z\\0&0&0&1\end{pmatrix}$$

**Scaling** by $(S_x,S_y,S_z)$:
$$\mathbf{S} = \begin{pmatrix}S_x&0&0&0\\0&S_y&0&0\\0&0&S_z&0\\0&0&0&1\end{pmatrix}$$

**Rotation about $x$-axis** by $\theta$:
$$\mathbf{R}_x = \begin{pmatrix}1&0&0&0\\0&C&-S&0\\0&S&C&0\\0&0&0&1\end{pmatrix}$$

**Rotation about $y$-axis** by $\theta$:
$$\mathbf{R}_y = \begin{pmatrix}C&0&S&0\\0&1&0&0\\-S&0&C&0\\0&0&0&1\end{pmatrix}$$

**Rotation about $z$-axis** by $\theta$:
$$\mathbf{R}_z = \begin{pmatrix}C&-S&0&0\\S&C&0&0\\0&0&1&0\\0&0&0&1\end{pmatrix}$$

> [!note] Sign convention for $\mathbf{R}_y$
> Note the sign swap compared to $\mathbf{R}_x$ and $\mathbf{R}_z$: the $-S$ entry is at position $(3,1)$ (lower-left of the $3{\times}3$ sub-block), not $(1,3)$. This follows from the right-hand rule with $y$-axis pointing "backwards".

**General 3D rotation** (Euler decomposition):
$$\mathbf{R}_{\theta_x\theta_y\theta_z} = \mathbf{R}_x\,\mathbf{R}_y\,\mathbf{R}_z = \begin{pmatrix}r_{11}&r_{12}&r_{13}\\r_{21}&r_{22}&r_{23}\\r_{31}&r_{32}&r_{33}\end{pmatrix}$$

**3D Shearing:**
$$\mathbf{H} = \begin{pmatrix}1&H_{yx}&H_{zx}&0\\H_{xy}&1&H_{zy}&0\\H_{xz}&H_{yz}&1&0\\0&0&0&1\end{pmatrix}$$

**Rotation matrix properties** (orthonormality + right-hand rule):
$$\mathbf{r}_i \cdot \mathbf{r}_j = \begin{cases}1 & i=j \\ 0 & i\neq j\end{cases}, \qquad \mathbf{r}_1 \times \mathbf{r}_2 = \mathbf{r}_3 \implies \det\mathbf{R} = +1$$
$$\therefore \quad \mathbf{R}^{-1} = \mathbf{R}^\top$$

---

### b3. Steps — WRF ↔ CRF (extrinsic transform)

**Forward (WRF → CRF):** Apply extrinsic $4{\times}4$ matrix:
$$\begin{pmatrix}x\\y\\z\\1\end{pmatrix} = \underbrace{\begin{pmatrix}\mathbf{R}_{3\times3} & \mathbf{t}_{3\times1} \\ \mathbf{0}^\top & 1\end{pmatrix}}_{(\mathbf{R}\mid\mathbf{t})}\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$

Expanded:
$$(\mathbf{R}\mid\mathbf{t}) = \begin{pmatrix}r_{11}&r_{12}&r_{13}&t_x\\r_{21}&r_{22}&r_{23}&t_y\\r_{31}&r_{32}&r_{33}&t_z\\0&0&0&1\end{pmatrix}$$

**Inverse (CRF → WRF):** Factor $(\mathbf{R}\mid\mathbf{t}) = \mathbf{T}_{\text{hom}}\,\mathbf{R}_{\text{hom}}$, invert each factor, multiply in reverse order. Exploiting $\mathbf{R}^{-1}=\mathbf{R}^\top$:

$$(\mathbf{R}\mid\mathbf{t})^{-1} = \begin{pmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t} \\ \mathbf{0}^\top & 1\end{pmatrix}$$

> [!note]
> This feeds directly into the full camera projection chain $\mathbf{P}=\mathbf{K}[\mathbf{R}\mid\mathbf{t}]$ covered in [[pinhole-camera-model]].

---

## A6. Variations — 2D transformation model hierarchy

The 2D model hierarchy is a key exam topic. Each level adds degrees of freedom and drops a preserved property. Fitting algorithms differ per level.

### Variant 1 — Rigid (Euclidean), 3 dof

**Change to setup:** $\mathbf{M}=\mathbf{R}$ (proper rotation only); no shear or scale.
$$\mathbf{T}_{\text{rigid}} = \begin{pmatrix}\mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1\end{pmatrix}, \quad \mathbf{R} \in SO(2), \quad \text{dof} = 3 \; (1\text{ rotation} + 2\text{ translation})$$
**Preserved:** Distances and angles (isometry).
**Fitting:** SVD-based Procrustes; requires $\geq 2$ point correspondences.
See [[rigid-transformation-procrustes]].

### Variant 2 — Similarity, 4 dof

**Change:** Adds uniform scale $s>0$.
$$\mathbf{T}_{\text{sim}} = \begin{pmatrix}s\mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1\end{pmatrix}, \quad \text{dof} = 4 \; (1\text{ rotation} + 1\text{ scale} + 2\text{ translation})$$
**Preserved:** Angles (shape, aspect ratio) — distances change uniformly.

### Variant 3 — Affine, 6 dof

**Change:** $\mathbf{M}$ is a general $2{\times}2$ invertible matrix (encodes rotation + anisotropic scale + shear).
$$\mathbf{T}_{\text{affine}} = \begin{pmatrix}\mathbf{M}_{2\times2} & \mathbf{t} \\ \mathbf{0}^\top & 1\end{pmatrix}, \quad \text{dof} = 6$$
**Preserved:** Parallelism (parallel lines remain parallel); ratios of lengths along lines; midpoints.
**Fitting:** Linear least squares; $\geq 3$ correspondences required.
See [[affine-transformation-fitting]].

> [!note]
> Affine is a good approximation for viewpoint changes of roughly planar objects under approximately orthographic cameras, and can initialise fitting for more complex models.

### Variant 4 — Projective (Homography), 8 dof

**Change:** Full $3{\times}3$ matrix $\mathbf{H}$ (9 entries, but scale is arbitrary → 8 dof).
$$w'\begin{pmatrix}x'\\y'\\1\end{pmatrix} = \mathbf{H}_{3\times3}\begin{pmatrix}x\\y\\1\end{pmatrix}, \quad \text{dof} = 8$$
**Preserved:** Collinearity only — straight lines stay straight; parallelism, angles, and distances NOT maintained.
**Fitting:** Direct Linear Transformation (DLT); $\geq 4$ correspondences required.
**Physical scenarios:** (1) Two views of a planar surface; (2) two cameras sharing the same optical centre (pure rotation).
See [[homography-and-dlt]].

### Summary table

| Model | dof | Min. correspondences | Preserved | Fitting method |
|---|---|---|---|---|
| Rigid | 3 | 2 | Distances + angles | SVD Procrustes |
| Similarity | 4 | 2 | Angles | Extended Procrustes |
| Affine | 6 | 3 | Parallelism | Least squares (normal equations) |
| Projective | 8 | 4 | Collinearity | DLT (homogeneous least squares) |

---

## Related topics

[[rigid-transformation-procrustes]] · [[affine-transformation-fitting]] · [[homography-and-dlt]] · [[pinhole-camera-model]] · [[vector-matrix-algebra]]
