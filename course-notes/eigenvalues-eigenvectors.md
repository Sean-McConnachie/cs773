# Eigenvalues & Eigenvectors

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps — brute-force procedure]]
  - [[#A5c. 2×2 shortcut]]
- [[#A6. Variations]]

---

## A1. Purpose

Given a square matrix $\mathbf{A}$, eigenvalue/eigenvector analysis finds the special scalar-vector pairs $(\lambda, \mathbf{x})$ where $\mathbf{A}$ acts as a pure scaling (no rotation) along $\mathbf{x}$. In CS773 this is used to (1) characterise the local gradient structure of an image via the Harris structure tensor $\mathbf{H}$, (2) define the axis directions and lengths of the SSD error ellipse, and (3) feed into SVD-based pipelines (rigid transformation, calibration). Eigenvalue analysis is the bridge from a raw 2×2 matrix to a geometric interpretation.

---

## A2. High-level overview

**Key definitions:**

| Term | Definition |
|---|---|
| **Eigenvalue** $\lambda$ | Scalar satisfying $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$ for some non-zero $\mathbf{x}$. |
| **Eigenvector** $\mathbf{x}$ | Non-zero vector satisfying $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$. The zero vector is **excluded by definition**. |
| **Characteristic equation** | $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$; roots are the eigenvalues. |
| **Characteristic polynomial** | Polynomial in $\lambda$ obtained by expanding the determinant; degree = matrix size. |
| **Smallest-integer convention** | When eigenvectors are infinitely many (scalar multiples), report the one with smallest integer components. |
| **Trace** | $\operatorname{tr}(\mathbf{A}) = \sum_i a_{ii}$; equals the sum of eigenvalues. |
| **Determinant** | $\det(\mathbf{A})$; equals the product of eigenvalues. |

**Steps (no math):**

1. Write the characteristic equation $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$.
2. Expand the determinant to get the characteristic polynomial.
3. Solve the polynomial for $\lambda_1, \lambda_2, \ldots$
4. For each $\lambda_i$, substitute back into $(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{x} = \mathbf{0}$ and solve for $\mathbf{x}$ (exclude $\mathbf{x} = \mathbf{0}$).
5. Apply smallest-integer convention to choose a canonical eigenvector.

---

## A3. Strengths, shortcomings & limitations

**Strengths:**

- Provides a complete geometric characterisation of how a matrix transforms space (stretching and direction).
- For symmetric matrices (e.g. the Harris structure tensor $\mathbf{H}$), eigenvectors are always orthogonal.
- The det/trace identities ($\det = \lambda_1\lambda_2$, $\operatorname{tr} = \lambda_1 + \lambda_2$) allow the Harris cornerness score to be computed without explicit eigenvalue calculation.

**Shortcomings / limitations:**

- Only defined for **square** matrices.
- The zero vector is never a valid eigenvector (must be explicitly excluded).
- Eigenvectors are not unique — any scalar multiple is equally valid, so a convention is required for reproducible answers.
- For matrices larger than 2×2, analytical solutions to the characteristic polynomial may not exist (degree ≥ 5 has no closed form).
- Complex eigenvalues can arise (e.g. for rotation matrices); not relevant for the symmetric $\mathbf{H}$ used in Harris.

---

### Likely exam questions

**Q:** State the defining equation for an eigenvector and explain why the zero vector is excluded.

**A:** $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$. The zero vector satisfies this equation for *any* $\lambda$, so including it would make eigenvalues undefined; eigenvectors must be non-zero by definition.

---

**Q:** Find the eigenvalues of $\mathbf{A} = \begin{pmatrix}2 & 2 \\ 5 & -1\end{pmatrix}$.

**A:** $\det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)(-1-\lambda) - 10 = \lambda^2 - \lambda - 12 = (\lambda-4)(\lambda+3) = 0$, so $\lambda_1 = 4$, $\lambda_2 = -3$.

---

**Q:** Find the eigenvectors of $\mathbf{A} = \begin{pmatrix}2 & 2 \\ 5 & -1\end{pmatrix}$ for $\lambda_1 = 4$ and $\lambda_2 = -3$, using the smallest-integer convention.

**A:** For $\lambda_1 = 4$: $(\mathbf{A} - 4\mathbf{I})\mathbf{x} = \mathbf{0}$ gives $-2x_1 + 2x_2 = 0 \Rightarrow x_1 = x_2$, so $\mathbf{x}_1 = \begin{pmatrix}1\\1\end{pmatrix}$. For $\lambda_2 = -3$: $5x_1 + 2x_2 = 0 \Rightarrow x_1 = -\tfrac{2}{5}x_2$; choosing $x_2 = 5$ gives $\mathbf{x}_2 = \begin{pmatrix}2\\-5\end{pmatrix}$.

---

**Q (W6T Q12):** For $\mathbf{A} = \begin{pmatrix}-12 & -11 \\ 22 & 21\end{pmatrix}$, find the eigenvalues and the angles each eigenvector makes with the x-axis.

**A:** Characteristic polynomial: $\lambda^2 - 9\lambda - 10 = 0 \Rightarrow \lambda_1 = 10$, $\lambda_2 = -1$.
- $\lambda_1 = 10$: $\mathbf{v}_1 = \begin{pmatrix}1\\-2\end{pmatrix}$. $\theta_1 = \tan^{-1}(-2/1) \approx -63.4°$; add $180°$ → $\theta_1 \approx 117°$.
- $\lambda_2 = -1$: $\mathbf{v}_2 = \begin{pmatrix}1\\-1\end{pmatrix}$. $\theta_2 = \tan^{-1}(-1/1) = -45°$; add $180°$ → $\theta_2 = 135°$.

---

**Q:** What do the eigenvalues and eigenvectors of the matrix form of an ellipse tell you?

**A:** The eigenvectors give the directions of the semi-axes. The semi-axis lengths are $1/\sqrt{\lambda}$ for each eigenvalue. A larger $\lambda$ corresponds to a *shorter* axis.

---

**Q:** How does the Harris cornerness score avoid computing eigenvalues explicitly?

**A:** Using the identities $\det(\mathbf{H}) = \lambda_1\lambda_2$ and $\operatorname{tr}(\mathbf{H}) = \lambda_1 + \lambda_2$, the score $C = \lambda_1\lambda_2 - \alpha(\lambda_1+\lambda_2)^2$ is rewritten as $C = \det(\mathbf{H}) - \alpha\,[\operatorname{tr}(\mathbf{H})]^2$, which uses only the matrix entries directly.

---

## A4. Directions of reasoning

### Forward: matrix → eigenvalues → eigenvectors

- **Given:** a square matrix $\mathbf{A}$.
- **Asked:** the $(\lambda, \mathbf{x})$ pairs.
- **Key steps:** solve $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ for $\lambda$; back-substitute each $\lambda$ into $(\mathbf{A} - \lambda\mathbf{I})\mathbf{x} = \mathbf{0}$.

### Reverse: verify a claimed eigenvector/eigenvalue pair

- **Given:** a candidate pair $(\lambda, \mathbf{x})$ and matrix $\mathbf{A}$.
- **Asked:** is this a valid pair?
- **Key steps:** compute $\mathbf{A}\mathbf{x}$; compute $\lambda\mathbf{x}$; check equality. Also check $\mathbf{x} \neq \mathbf{0}$.

> [!example]
> Verify that $\mathbf{v} = (1,1)^T$ is an eigenvector of $\mathbf{A} = \begin{pmatrix}2&2\\-4&8\end{pmatrix}$ with $\lambda = 4$:
> $\mathbf{A}\mathbf{v} = \begin{pmatrix}4\\4\end{pmatrix} = 4\begin{pmatrix}1\\1\end{pmatrix}$ ✓.
> Check that $\mathbf{w} = (2,1)^T$ is NOT: $\mathbf{A}\mathbf{w} = \begin{pmatrix}6\\0\end{pmatrix} \neq 4\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}8\\4\end{pmatrix}$ ✗.

### Inferential: eigenvalues → region classification (Harris context)

- **Given:** eigenvalues $\lambda_1$, $\lambda_2$ of $\mathbf{H}$ at a pixel.
- **Asked:** flat, edge, or corner?
- **Key inference:**
  - Both small → flat (no gradient in any direction).
  - One large, one $\approx 0$ → edge (gradient in one direction only).
  - Both large → corner (gradient in all directions).

---

## A5. Standard implementation

### A5a. Setup

- **Input:** a square $n \times n$ matrix $\mathbf{A}$ (in CS773 almost always $2 \times 2$).
- **Output:** eigenvalues $\lambda_1, \lambda_2, \ldots$; eigenvectors $\mathbf{x}_1, \mathbf{x}_2, \ldots$ (smallest-integer convention unless normalisation is requested).
- **Assumptions:** $\mathbf{A}$ is square; zero eigenvectors are excluded.
- **Notation:** $2 \times 2$ case $\mathbf{A} = \begin{pmatrix}a & b \\ c & d\end{pmatrix}$; $\mathbf{I}$ = identity.

---

### A5b. Steps — brute-force procedure

1. **Form $\mathbf{A} - \lambda\mathbf{I}$:**
$$\mathbf{A} - \lambda\mathbf{I} = \begin{pmatrix}a - \lambda & b \\ c & d - \lambda\end{pmatrix}$$

2. **Write the characteristic equation:**
$$\det(\mathbf{A} - \lambda\mathbf{I}) = (a-\lambda)(d-\lambda) - bc = 0$$

3. **Expand to the characteristic polynomial and solve for $\lambda$:**
$$\lambda^2 - (a+d)\lambda + (ad - bc) = 0$$
Use the quadratic formula or factorise.

4. **For each eigenvalue $\lambda_i$, solve $(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{x} = \mathbf{0}$:**
Write out the two simultaneous equations; use either row to express one component in terms of the other.

5. **Apply smallest-integer convention:**
Choose the simplest non-zero integer vector satisfying the relation found in step 4.

> [!warning]
> Never report $\mathbf{x} = \mathbf{0}$ as an eigenvector. If both equations give $0 = 0$, it means any non-zero vector works — pick $(1, 0)^T$ or the simplest non-trivial choice.

> [!example]
> **Full worked example:** $\mathbf{A} = \begin{pmatrix}2 & 2 \\ 5 & -1\end{pmatrix}$
>
> **Step 1–3:**
> $$\det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)(-1-\lambda) - 10 = \lambda^2 - \lambda - 12 = 0$$
> $$(\lambda - 4)(\lambda + 3) = 0 \implies \lambda_1 = 4,\quad \lambda_2 = -3$$
>
> **Step 4–5, $\lambda_1 = 4$:**
> $$(\mathbf{A} - 4\mathbf{I})\mathbf{x} = \begin{pmatrix}-2 & 2 \\ 5 & -5\end{pmatrix}\mathbf{x} = \mathbf{0} \implies -2x_1 + 2x_2 = 0 \implies x_1 = x_2$$
> $$\mathbf{x}_1 = \begin{pmatrix}1\\1\end{pmatrix}$$
>
> **Step 4–5, $\lambda_2 = -3$:**
> $$(\mathbf{A} + 3\mathbf{I})\mathbf{x} = \begin{pmatrix}5 & 2 \\ 5 & 2\end{pmatrix}\mathbf{x} = \mathbf{0} \implies 5x_1 + 2x_2 = 0 \implies x_1 = -\tfrac{2}{5}x_2$$
> Set $x_2 = 5$: $\mathbf{x}_2 = \begin{pmatrix}2\\-5\end{pmatrix}$

---

### A5c. 2×2 shortcut

For $\mathbf{A} = \begin{pmatrix}a & b \\ c & d\end{pmatrix}$, compute directly:

$$\boxed{\lambda_{1,2} = \frac{\operatorname{tr}(\mathbf{A}) \pm \sqrt{\operatorname{tr}(\mathbf{A})^2 - 4\det(\mathbf{A})}}{2}}$$

where $\operatorname{tr}(\mathbf{A}) = a + d$ and $\det(\mathbf{A}) = ad - bc$.

**Direct eigenvector read (no row reduction):**

$$\text{If } b \neq 0: \quad \mathbf{x}_{1,2} = \begin{pmatrix}b \\ \lambda_{1,2} - a\end{pmatrix}$$

$$\text{If } c \neq 0: \quad \mathbf{x}_{1,2} = \begin{pmatrix}\lambda_{1,2} - d \\ c\end{pmatrix}$$

> [!note]
> The shortcut formulas come from the same characteristic polynomial — they just pre-solve it. The direct eigenvector formula is derived by observing that row 1 of $(\mathbf{A} - \lambda\mathbf{I})\mathbf{x} = \mathbf{0}$ gives $bx_2 = (\lambda - a)x_1$.

**Eigenvector angle with x-axis:**

$$\theta = \tan^{-1}\!\left(\frac{v_y}{v_x}\right)$$

If $\theta < 0°$, **add 180°** to obtain the canonical angle in $[0°, 180°]$.

> [!example]
> **W6T Q12:** $\mathbf{A} = \begin{pmatrix}-12 & -11 \\ 22 & 21\end{pmatrix}$, $\lambda_1 = 10$, $\lambda_2 = -1$.
>
> $\mathbf{v}_1 = \begin{pmatrix}1\\-2\end{pmatrix}$: $\theta_1 = \tan^{-1}(-2) \approx -63.4° \;\xrightarrow{+180°}\; 117°$
>
> $\mathbf{v}_2 = \begin{pmatrix}1\\-1\end{pmatrix}$: $\theta_2 = \tan^{-1}(-1) = -45° \;\xrightarrow{+180°}\; 135°$

---

## A6. Variations

### Ellipse interpretation (structure tensor context)

The SSD error function, after Taylor approximation, takes the quadratic form:

$$E(u,v) \approx \begin{pmatrix}u & v\end{pmatrix}\mathbf{H}\begin{pmatrix}u \\ v\end{pmatrix}$$

The level sets of $E$ (i.e. $E = \text{const}$) are **ellipses**. The eigenstructure of $\mathbf{H}$ gives:

| Eigenstructure | Geometric meaning |
|---|---|
| Eigenvectors $\mathbf{x}_1, \mathbf{x}_2$ | Directions of the ellipse semi-axes (principal axes of the error surface). |
| $1/\sqrt{\lambda_i}$ | Length of the semi-axis along $\mathbf{x}_i$. |
| Larger $\lambda$ | Steeper $E$ → shorter ellipse axis (direction of maximum intensity change). |
| Smaller $\lambda$ | Flatter $E$ → longer ellipse axis (direction of minimum intensity change). |

**Worked example** (from the tutorial): the axis-aligned ellipse $\tfrac{x^2}{121} + \tfrac{y^2}{36} = 1$ written in matrix form gives $\mathbf{A} = \begin{pmatrix}1/121 & 0 \\ 0 & 1/36\end{pmatrix}$. Its eigenvalues are $\lambda_1 = 1/121$, $\lambda_2 = 1/36$; eigenvectors are $(1,0)^T$, $(0,1)^T$. Semi-axes: $1/\sqrt{1/121} = 11$ and $1/\sqrt{1/36} = 6$, recovering the original intercepts $(\pm 11, 0)$ and $(0, \pm 6)$.

This links directly to the Harris structure tensor: the **eigenvectors of $\mathbf{H}$** indicate the dominant gradient directions at a pixel, and $1/\sqrt{\lambda}$ sets the extent of the SSD ellipse along each axis. See [[harris-corner-detector]] for the full pipeline and region classification by eigenvalue magnitudes.

> [!note]
> **Ellipse orientation rule (axis-aligned case):** $\tfrac{x^2}{a^2} + \tfrac{y^2}{b^2} = 1$. Larger denominator under $x^2$ → horizontal semi-axis is $a$; larger denominator under $y^2$ → vertical semi-axis is $b$.

---

### Region classification via eigenvalue magnitudes

| Eigenvalues of $\mathbf{H}$ | Region type | Harris $C$ |
|---|---|---|
| Both $\lambda_1, \lambda_2 \approx 0$ | **Flat** | $\|C\|$ small |
| One large, one $\approx 0$ | **Edge** | $C < 0$ |
| Both large (and similar) | **Corner** | $C > 0$ and large |

The cornerness score avoids explicit eigenvalue computation using:

$$C = \det(\mathbf{H}) - \alpha\,[\operatorname{tr}(\mathbf{H})]^2 = \lambda_1\lambda_2 - \alpha(\lambda_1+\lambda_2)^2, \quad \alpha \approx 0.04\text{–}0.06$$

See [[harris-corner-detector]] for the full 8-step Harris pipeline and NMS.

---

### Relation to SVD

The columns of $\mathbf{U}$ in SVD ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$) are eigenvectors of $\mathbf{A}\mathbf{A}^T$; columns of $\mathbf{V}$ are eigenvectors of $\mathbf{A}^T\mathbf{A}$. Singular values $\sigma_i = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^T)}$. Eigenvalue computation is therefore the computational core of SVD. See [[singular-value-decomposition]].

For foundational matrix algebra (determinants, trace, matrix multiplication) see [[vector-matrix-algebra]].
