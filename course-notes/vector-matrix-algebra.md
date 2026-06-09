# Vector & Matrix Algebra

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Key definitions & notation]]
  - [[#A5b. Operation formulas]]
- [[#A6. Variations & related topics]]

---

## A1. Purpose

Vector and matrix algebra is the mathematical foundation for nearly every operation in CS773: describing points and directions in 3D space, computing geometric relationships (angles, projections, distances), transforming coordinates between frames, and setting up the linear systems that cameras, calibration, and feature matching all depend on. Reach for these tools whenever a problem involves spatial geometry, coordinate change, or the structure of a linear map.

---

## A2. High-level overview

1. Represent points and directions as **vectors**; represent linear maps as **matrices**.
2. Use **dot product** to measure angles and project one vector onto another.
3. Use **cross product** to find a vector perpendicular to two others (e.g. plane normals).
4. Use **matrix multiplication** to compose transformations and map vectors into new spaces.
5. Use **determinant / inverse** to check invertibility and reverse a linear map.
6. Use the **plane equation** and **ray equation** to express geometric objects analytically.

**Key definitions:**

| Term | Definition |
|---|---|
| **Scalar** | A single real number, written in italics: $f$, $\lambda$, $d$ |
| **Vector** | Ordered list of scalars representing displacement; written bold lowercase: $\mathbf{u}$, $\mathbf{v}$, $\mathbf{p}$ |
| **Unit vector** | A vector of magnitude 1; denoted $\hat{\mathbf{u}}$ |
| **Matrix** | Rectangular array of scalars representing a linear map; bold uppercase: $\mathbf{M}$, $\mathbf{R}$ |
| **Magnitude** | Euclidean length of a vector: $|\mathbf{u}| = \sqrt{\sum_i u_i^2}$ |
| **Normalisation** | Dividing a vector by its magnitude to make it a unit vector |
| **Transpose** | Swaps rows and columns: $(M^T)_{ij} = M_{ji}$ |
| **Dot product** | Scalar result of $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = |\mathbf{u}||\mathbf{v}|\cos\theta$ |
| **Orthogonal projection** | Component of $\mathbf{u}$ in the direction of $\mathbf{v}$, denoted $\mathbf{u}_v$ |
| **Cross product** | 3D vector perpendicular to both inputs: $\mathbf{u} \times \mathbf{v}$ |
| **Determinant** | Scalar measure of a matrix's scaling factor; $\det(\mathbf{M}) = 0$ iff not invertible |
| **Inverse** | $\mathbf{M}^{-1}$ such that $\mathbf{M}\mathbf{M}^{-1} = \mathbf{M}^{-1}\mathbf{M} = \mathbf{I}$ |
| **Plane (normal form)** | $\mathbf{n} \cdot \mathbf{p} = d$ where $\mathbf{n}$ is a unit normal and $d$ is the distance from origin |
| **Ray** | Parametric line $p(t) = \mathbf{o} + \mathbf{d}\,t$; $\mathbf{o}$ = origin, $\mathbf{d}$ = direction, $t \in \mathbb{R}$ |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Exact, closed-form formulas — no iteration needed for the operations listed here.
- Operations compose cleanly via matrix multiplication, enabling efficient multi-step transforms.
- Dot and cross products encode geometry (angles, perpendicularity, area) directly.

**Shortcomings / limitations**
- Inverse exists only when $\det(\mathbf{M}) \neq 0$; degenerate configurations (e.g. parallel planes, collinear points) break direct solutions.
- 2×2 formulas do not generalise directly to $n \times n$ matrices; larger inverses require Gaussian elimination or LU/SVD decomposition (see [[singular-value-decomposition]]).
- Cross product is strictly 3D; no direct analogue in 2D or higher dimensions.
- Floating-point arithmetic introduces rounding; normalisation before dot/plane computations is essential.

> [!warning]
> $d$ in $ax + by + cz = d$ equals the distance from the plane to the origin **only when $\mathbf{n} = (a, b, c)^T$ is already a unit vector**. Always normalise first.

### Likely exam questions

**Q:** Given $\mathbf{u} = (1, 2, -1)^T$, normalise $\mathbf{u}$.
**A:** $|\mathbf{u}| = \sqrt{1 + 4 + 1} = \sqrt{6}$, so $\hat{\mathbf{u}} = \frac{1}{\sqrt{6}}(1, 2, -1)^T$.

---

**Q:** Compute the scalar projection of $\mathbf{u} = (6, 15, 10)^T$ onto $\mathbf{v} = (5, 14, 9)^T$.

> [!example]
> $\mathbf{v} \cdot \mathbf{u} = 30 + 210 + 90 = 330$; $|\mathbf{v}| = \sqrt{25 + 196 + 81} = \sqrt{302} \approx 19$; scalar projection $= 330/19 \approx 17.37$.

---

**Q:** Find $\mathbf{M}^{-1}$ for $\mathbf{M} = \begin{pmatrix}2 & 3 \\ 3 & 3\end{pmatrix}$.

> [!example]
> $\det \mathbf{M} = 6 - 9 = -3$. $\mathbf{M}^{-1} = \frac{1}{-3}\begin{pmatrix}3 & -3 \\ -3 & 2\end{pmatrix} = \frac{1}{3}\begin{pmatrix}-3 & 3 \\ 3 & -2\end{pmatrix}$.

---

**Q:** A ray has origin $\mathbf{o} = (0, 1, -1)^T$ and direction $\mathbf{d} = (1, 1, 1)^T$. Find where it intersects the plane $x + y + z = 2$.

> [!example]
> Parametric: $x = t$, $y = 1 + t$, $z = t - 1$. Substitute: $t + (1+t) + (t-1) = 2 \Rightarrow 3t = 2 \Rightarrow t = 2/3$. Intersection: $p(2/3) = (2/3,\ 5/3,\ -1/3)^T$.

---

**Q:** The plane $3x + y - 2z = 5$. Find the distance from the plane to point $Q = (3, 4, 2)^T$.

> [!example]
> $|\mathbf{n}| = \sqrt{9 + 1 + 4} = \sqrt{14}$. Distance $= \frac{\mathbf{q} \cdot \mathbf{n} - d}{|\mathbf{n}|} = \frac{9 + 4 - 4 - 5}{\sqrt{14}} = \frac{4}{\sqrt{14}}$.

---

**Q:** What is the dot product of $\mathbf{u} = (-1, -2, 3)^T$ and $\mathbf{v} = (1, -1, 1)^T$, and what does the sign tell you?
**A:** $\mathbf{u} \cdot \mathbf{v} = -1 + 2 + 3 = 4 > 0$, so the angle between them is acute ($\theta < 90°$).

---

## A4. Directions of reasoning

### Forward: inputs → geometric quantity
- **Given** two vectors $\mathbf{u}$, $\mathbf{v}$; **asked** for the angle between them.
  - Compute $\mathbf{u} \cdot \mathbf{v}$, compute $|\mathbf{u}|$ and $|\mathbf{v}|$, then $\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{|\mathbf{u}||\mathbf{v}|}$.
- **Given** a point $S$ on a plane and a unit normal $\mathbf{n}$; **asked** for the plane equation.
  - Compute $d = \mathbf{n} \cdot \mathbf{s}$ and write $\mathbf{n} \cdot \mathbf{p} = d$.
- **Given** a ray and a plane; **asked** for the intersection point.
  - Substitute the parametric components of $p(t) = \mathbf{o} + \mathbf{d}\,t$ into $\mathbf{n} \cdot \mathbf{p} = d$, solve for $t$, back-substitute.

### Reverse: output/constraint → recover inputs or verify a step
- **Given** the result of a matrix-vector product $\mathbf{M}\mathbf{x} = \mathbf{b}$; **asked** for $\mathbf{x}$.
  - $\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}$ (exists iff $\det \mathbf{M} \neq 0$).
- **Given** a plane equation in un-normalised form $ax + by + cz = d$; **asked** for the distance to origin.
  - Normalise: distance $= d / \sqrt{a^2 + b^2 + c^2}$.
- **Given** the plane and a point; **asked** which side the point lies on.
  - Evaluate $\mathbf{q} \cdot \mathbf{n} - d$: positive = same side as $\mathbf{n}$ points, negative = opposite.

---

## A5. Standard implementation

### A5a. Key definitions & notation

> [!note]
> Spec notation used throughout: scalars italic ($d$, $\lambda$, $t$); vectors bold lowercase ($\mathbf{u}$, $\mathbf{v}$, $\mathbf{n}$, $\mathbf{o}$, $\mathbf{d}$); matrices bold uppercase ($\mathbf{M}$, $\mathbf{R}$). W2T slides use $\hat{\mathbf{u}}$ for a unit vector, $|\mathbf{u}|$ for magnitude, $\mathbf{u}_v$ for orthogonal projection of $\mathbf{u}$ onto $\mathbf{v}$.

**Input/output conventions:**
- Vectors are **column vectors** unless transposed.
- Matrix-vector product $\mathbf{M}\mathbf{x}$: $\mathbf{M}$ is $(m \times n)$, $\mathbf{x}$ is $(n \times 1)$, result is $(m \times 1)$.
- Matrix product $\mathbf{M}\mathbf{N}$: inner dimensions must match; result shape is (rows of $\mathbf{M}$) × (cols of $\mathbf{N}$).

### A5b. Operation formulas

**1. Magnitude**
$$|\mathbf{u}| = \sqrt{u_1^2 + u_2^2 + \cdots + u_n^2}$$

**2. Normalisation**
$$\hat{\mathbf{u}} = \frac{\mathbf{u}}{|\mathbf{u}|}$$

> [!example]
> $\mathbf{u} = (1, 2, -1)^T$: $|\mathbf{u}| = \sqrt{6}$, $\hat{\mathbf{u}} = \frac{1}{\sqrt{6}}(1, 2, -1)^T$.

**3. Vector addition / subtraction / scaling**
$$\mathbf{u} \pm \mathbf{v} = (u_1 \pm v_1,\ u_2 \pm v_2,\ \ldots)^T, \qquad c\,\mathbf{u} = (cu_1, cu_2, \ldots)^T$$

**4. Transpose**
$$(\mathbf{M}^T)_{ij} = M_{ji}; \qquad (\mathbf{u}^T\mathbf{v}) = \mathbf{u} \cdot \mathbf{v} \text{ (row-vec times col-vec)}$$

**5. Dot product & angle**
$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T\mathbf{v} = \sum_i u_i v_i = |\mathbf{u}||\mathbf{v}|\cos\theta$$
$$\theta = \cos^{-1}\!\left(\frac{\mathbf{u}\cdot\mathbf{v}}{|\mathbf{u}||\mathbf{v}|}\right)$$

> [!example]
> $\mathbf{u} = (5,3,-2)^T$, $\mathbf{v} = (3,-1,2)^T$: $\mathbf{u}\cdot\mathbf{v} = 15 - 3 - 4 = 8$.

**6. Orthogonal projection of $\mathbf{u}$ onto $\mathbf{v}$**

Scalar (magnitude):
$$|\mathbf{u}_v| = \frac{\mathbf{v} \cdot \mathbf{u}}{|\mathbf{v}|}$$

Vector form:
$$\mathbf{u}_v = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{v} \cdot \mathbf{v}}\,\mathbf{v}$$

> [!example]
> $\mathbf{u} = (1,1,2)^T$, $\mathbf{v} = (4,0,3)^T$: $|\mathbf{u}_v| = \frac{4+0+6}{\sqrt{16+9}} = \frac{10}{5} = 2$.
>
> $\mathbf{u} = (1,2,0)^T$, $\mathbf{v} = (5,3,1)^T$: $\mathbf{u}_v = \frac{5+6+0}{25+9+1}(5,3,1)^T = \frac{11}{35}(5,3,1)^T$.

**7. Matrix multiplication**
$$(\mathbf{M}\mathbf{N})_{ij} = \sum_k M_{ik}\,N_{kj}$$

> [!note]
> Order matters: $\mathbf{M}\mathbf{N} \neq \mathbf{N}\mathbf{M}$ in general.

> [!example]
> $\mathbf{M} = \begin{pmatrix}-2&3\\3&3\end{pmatrix}$, $\mathbf{N} = \begin{pmatrix}2&-3\\3&-3\end{pmatrix}$: $\mathbf{M}\mathbf{N} = \begin{pmatrix}5&-3\\15&-18\end{pmatrix}$.
>
> Non-square: $\mathbf{M}_{2\times3} = \begin{pmatrix}4&2&1\\-3&1&5\end{pmatrix}$, $\mathbf{N}_{3\times2} = \begin{pmatrix}1&-2\\0&-3\\4&0\end{pmatrix}$: $\mathbf{M}\mathbf{N} = \begin{pmatrix}8&-14\\17&3\end{pmatrix}$.

**8. 2×2 determinant**
$$\det\begin{pmatrix}a & b \\ c & d\end{pmatrix} = ad - bc$$

> [!example]
> $\mathbf{M} = \begin{pmatrix}1&-3\\5&3\end{pmatrix}$: $\det = 3 - (-15) = 18$.
>
> $\mathbf{M} = \begin{pmatrix}2&3\\3&3\end{pmatrix}$: $\det = 6 - 9 = -3$.

**9. 2×2 matrix inverse**
$$\mathbf{M}^{-1} = \frac{1}{ad - bc}\begin{pmatrix}d & -b \\ -c & a\end{pmatrix}, \quad \det(\mathbf{M}) \neq 0$$

- $\mathbf{M}^{-1}\mathbf{M} = \mathbf{M}\mathbf{M}^{-1} = \mathbf{I}$; $(\mathbf{M}^{-1})^{-1} = \mathbf{M}$.

> [!example]
> $\mathbf{M} = \begin{pmatrix}1&2\\2&4\end{pmatrix}$: $\det = 0$ → **no inverse**.
>
> $\mathbf{M} = \begin{pmatrix}2&4\\-1&5\end{pmatrix}$: $\det = 14$; $\mathbf{M}^{-1} = \frac{1}{14}\begin{pmatrix}5&-4\\1&2\end{pmatrix}$.
>
> $\mathbf{M} = \begin{pmatrix}3&2\\-3&3\end{pmatrix}$: $\det = 15$; $\mathbf{M}^{-1} = \frac{1}{15}\begin{pmatrix}3&-2\\3&3\end{pmatrix}$.

**10. Cross product (3D only)**
$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix}\mathbf{i} & \mathbf{j} & \mathbf{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3\end{vmatrix} = \begin{pmatrix}u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1\end{pmatrix}$$

Result is perpendicular to both $\mathbf{u}$ and $\mathbf{v}$; used to find plane normals.

> [!example]
> $\mathbf{u} = (-1,-1,1)^T$, $\mathbf{v} = (1,0,1)^T$: $\mathbf{u}\times\mathbf{v} = (-1\cdot1 - 1\cdot0,\ 1\cdot1 - (-1)\cdot1,\ (-1)\cdot0 - (-1)\cdot1) = (-1, 2, 1)^T$.
>
> $\mathbf{u} = (-1,1,3)^T$, $\mathbf{v} = (2,1,0)^T$: $\mathbf{u}\times\mathbf{v} = (-3, 6, -3)^T$.

**11. Plane equation (normal form)**

Given a unit normal $\mathbf{n}$ and any point $\mathbf{s}$ on the plane:
$$\mathbf{n} \cdot (\mathbf{p} - \mathbf{s}) = 0 \implies \mathbf{n} \cdot \mathbf{p} = d, \quad d = \mathbf{n} \cdot \mathbf{s}$$
$$ax + by + cz = d \quad \text{where } \mathbf{n} = (a, b, c)^T$$

**12. Distance from plane to origin**

When $|\mathbf{n}| = 1$: distance $= d$.

When $|\mathbf{n}| \neq 1$: distance $= \dfrac{d}{|\mathbf{n}|}$.

> [!example]
> Plane $10x + 10y - z = 109$: $|\mathbf{n}| = \sqrt{100+100+1} = \sqrt{201}$; distance $= \dfrac{109}{\sqrt{201}}$.
>
> Plane $52x + 429y - 832z = 0$: distance $= 0$ (passes through origin).

**13. Distance from plane to a point $Q$**

(Unit normal form, $|\mathbf{n}| = 1$):
$$\text{distance} = \mathbf{q} \cdot \mathbf{n} - d$$

(Un-normalised $\mathbf{n}$):
$$\text{distance} = \frac{\mathbf{q} \cdot \mathbf{n} - d}{|\mathbf{n}|}$$

> [!example]
> $Q = (3,4,2)^T$, plane $3x + y - 2z = 5$: $|\mathbf{n}| = \sqrt{14}$; distance $= \dfrac{(9+4-4) - 5}{\sqrt{14}} = \dfrac{4}{\sqrt{14}}$.
>
> $Q = (1,2,3)^T$, plane $x + y + z = 1$: $|\mathbf{n}| = \sqrt{3}$; distance $= \dfrac{(1+2+3)-1}{\sqrt{3}} = \dfrac{5}{\sqrt{3}}$.

**14. Ray equation**
$$p(t) = \mathbf{o} + \mathbf{d}\,t, \quad t \in \mathbb{R}$$

$\mathbf{o}$: ray origin; $\mathbf{d}$: direction vector; $t$: parameter ($t > 0$ = forward, $t < 0$ = behind).

**Ray–plane intersection:** substitute parametric components into plane equation $\mathbf{n}\cdot\mathbf{p} = d$, solve for $t$, back-substitute.

> [!example]
> $\mathbf{o} = (0,1,-1)^T$, $\mathbf{d} = (1,1,1)^T$, plane $x+y+z=2$:
> $t + (1+t) + (t-1) = 2 \Rightarrow 3t = 2 \Rightarrow t = 2/3$.
> $p(2/3) = (2/3,\ 5/3,\ -1/3)^T$.

See [[backward-projection-and-ray-intersection]] for how this is applied to camera rays.

---

## A6. Variations & related topics

### Homogeneous coordinates
Appending a 1 (or any scalar $w \neq 0$) to a vector embeds it in projective space, allowing translations to be expressed as matrix multiplications. Cartesian $(x,y,z) \to$ homogeneous $(x,y,z,1)$; reverse by dividing by $w$ and dropping it. See [[homogeneous-coordinates-and-transformations]] for the full affine/projective transform machinery.

### Eigenvalues & eigenvectors
A special case of matrix-vector multiplication where $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$ — the vector is only scaled, not rotated. The determinant condition $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ uses the 2×2 determinant formula directly. See [[eigenvalues-eigenvectors]].

### Singular value decomposition (SVD)
Generalises the inverse/determinant to non-square and rank-deficient matrices; decomposes $\mathbf{M} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ into orthonormal factors. Used in rigid-transformation fitting and DLT calibration. See [[singular-value-decomposition]].

### Orthogonal (rotation) matrices
A rotation matrix $\mathbf{R}$ has orthonormal columns: $\mathbf{r}_i \cdot \mathbf{r}_j = 0\ (i \neq j)$, $|\mathbf{r}_i| = 1$, and right-hand rule $\mathbf{r}_1 \times \mathbf{r}_2 = \mathbf{r}_3$. Consequence: $\mathbf{R}^{-1} = \mathbf{R}^T$. This is exploited in the extrinsic matrix inverse: $(\mathbf{R}|\mathbf{t})^{-1} = \begin{pmatrix}\mathbf{R}^T & -\mathbf{R}^T\mathbf{t} \\ \mathbf{0}^T & 1\end{pmatrix}$.

### Dot product as a similarity measure
When both vectors are unit vectors, $\mathbf{u}\cdot\mathbf{v} = \cos\theta \in [-1, 1]$ is directly interpretable as a normalised similarity. Related to NCC in [[patch-similarity-measures]].
