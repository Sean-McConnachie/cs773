# Harris Corner Detector

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

The Harris Corner Detector (Harris & Stephens, 1988) efficiently identifies **corners** — image locations where intensity changes sharply in all directions — for use as repeatable keypoints in matching, tracking, and reconstruction. It replaces the naive, computationally intractable SSD auto-correlation $E(u,v)$ (see [[feature-points-and-autocorrelation]]) with a local quadratic approximation, reducing complexity from $O(W^2 S^2 N^2)$ to $O(N^2)$. Its output is a scalar **cornerness map** $C$ over the image, thresholded and non-maximum-suppressed to yield a sparse set of corner keypoints used as input to [[feature-descriptors]].

---

## A2. High-level overview

**Steps (no math):**
1. Smooth the input image with a Gaussian to suppress noise.
2. Compute horizontal and vertical image gradients $I_x$, $I_y$.
3. Form the three gradient-product images $I_x^2$, $I_y^2$, $I_x I_y$.
4. Smooth each product image with a Gaussian (weighted windowing).
5. Assemble the **structure tensor** $\mathbf{H}$ at every pixel from the smoothed products.
6. Compute the cornerness score $C = \det\mathbf{H} - \alpha\,(\operatorname{tr}\mathbf{H})^2$ at every pixel.
7. Threshold: keep only pixels where $C > T$.
8. Non-maximum suppression (NMS): retain only local maxima of $C$ in a neighbourhood.

**Key definitions:**

| Term | Definition |
|---|---|
| $E(u,v)$ | SSD auto-correlation error — measures intensity change in a window shifted by $(u,v)$ |
| Taylor linearisation | First-order approximation $I(x{+}u,y{+}v)\approx I(x,y)+uI_x+vI_y$, valid for small shifts |
| Structure tensor $\mathbf{H}$ | 2×2 symmetric matrix summarising the local gradient distribution in a window |
| $I_x = \partial I/\partial x$, $I_y = \partial I/\partial y$ | Horizontal/vertical image derivatives |
| $\lambda_1, \lambda_2$ | Eigenvalues of $\mathbf{H}$; characterise the shape of the local $E$ surface |
| Cornerness score $C$ | Scalar per-pixel measure; avoids computing eigenvalues explicitly |
| $\alpha$ | Harris sensitivity parameter; $\alpha \in [0.04, 0.06]$ |
| NMS | Non-maximum suppression — removes redundant detections near a true corner |

> [!warning]
> **Notation clash.** In this course, $\mathbf{H}$ denotes the **structure tensor** (Harris matrix). The same bold symbol $\mathbf{H}$ is used elsewhere for a **homography** matrix (see [[homography-and-dlt]]). Context always disambiguates: in Harris derivations $\mathbf{H}$ is 2×2 and built from image gradients; in homography contexts $\mathbf{H}$ is 3×3 and relates image planes.

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Computationally efficient: $O(N^2)$ after gradient pre-computation, no explicit eigenvalue decomposition needed.
- Rotation-invariant: $C$ depends only on eigenvalues of $\mathbf{H}$, which are invariant to image rotation.
- Tunable sensitivity via $\alpha$ and threshold $T$.
- Well-understood: clear geometric interpretation via the $E$-surface ellipse.

**Shortcomings / Limitations:**
- **Not scale-invariant:** a corner at one scale may appear as an edge or flat region at another (contrast with SIFT/SURF which add scale selection).
- **Not fully illumination-invariant:** gradient magnitudes depend on absolute intensity contrast.
- **Sensitive to blur:** pre-smoothing $\sigma$ affects which structures are detected.
- **Parameter dependence:** results vary with window size, $\sigma$, $\alpha$, and threshold $T$.
- **Planar assumption:** the Taylor linearisation assumes small shifts; large $(u,v)$ may be poorly approximated.

---

### Likely exam questions

**Q:** Write the Harris cornerness score $C$ in terms of $\det\mathbf{H}$ and $\operatorname{tr}\mathbf{H}$. What values of $C$ indicate a corner, edge, and flat region?
**A:** $C = \det\mathbf{H} - \alpha\,(\operatorname{tr}\mathbf{H})^2$, $\alpha \in [0.04, 0.06]$. Corner: $C > 0$ (both $\lambda$ large). Edge: $C < 0$ (one $\lambda$ dominates). Flat: $|C|$ small (both $\lambda \approx 0$).

**Q:** Starting from the SSD error $E(u,v)$, derive how the structure tensor $\mathbf{H}$ arises.
**A:** Apply the first-order Taylor approximation $I(x{+}u,y{+}v) \approx I(x,y)+uI_x+vI_y$ to the shifted term. Substitute into $E$: the squared bracket becomes $(I_x u + I_y v)^2$. Expand and collect: $E \approx Au^2 + 2Buv + Cv^2 = \begin{bmatrix}u&v\end{bmatrix}\mathbf{H}\begin{bmatrix}u\\v\end{bmatrix}$ where $\mathbf{H}=\sum w\bigl(\begin{smallmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{smallmatrix}\bigr)$.

**Q:** Given $\mathbf{H}=\bigl(\begin{smallmatrix}33&6\\6&12\end{smallmatrix}\bigr)$, compute $C$ with $\alpha=0.04$. Classify the region.
**A:** $\det\mathbf{H}=33\times12-36=360$. $\operatorname{tr}\mathbf{H}=45$. $C=360-0.04\times45^2=360-81=279>0$ → **corner**.

**Q:** Given $\mathbf{H}=\bigl(\begin{smallmatrix}181847.63&0\\0&0\end{smallmatrix}\bigr)$, what region type is this and why?
**A:** $\det\mathbf{H}=0$, $\operatorname{tr}\mathbf{H}=181847.63$, $C=0-0.04\times181847.63^2\ll 0$ → **edge**. One eigenvalue is large ($181847.63$), the other is zero — intensity changes only in one direction (vertical edge: $I_y\approx0$, dominant gradient is $I_x$).

**Q:** Why does the Harris detector use $C = \det\mathbf{H} - \alpha\,(\operatorname{tr}\mathbf{H})^2$ instead of directly checking $\lambda_1$ and $\lambda_2$?
**A:** $\det\mathbf{H}=\lambda_1\lambda_2$ and $\operatorname{tr}\mathbf{H}=\lambda_1+\lambda_2$ (linear-algebra identities), so $C$ recovers the eigenvalue classification without computing eigenvalues explicitly, saving cost and avoiding numerical instability.

**Q:** What are the last two steps of the Harris pipeline, and why are they both necessary?
**A:** (7) Threshold $C>T$ removes low-response pixels; (8) NMS (see [[non-maximum-suppression]]) removes duplicate detections by keeping only local maxima. Thresholding alone leaves clusters of high-$C$ pixels at each corner; NMS collapses each cluster to a single representative point.

---

## A4. Directions of reasoning

### Forward (given image → detect corners)

**Given:** Grayscale image $I$.
**Asked:** Set of corner pixel locations.
**Inference:** Run the 8-step pipeline (A5b) to produce $C$ map → threshold → NMS → corner list.

### Reverse (given $\mathbf{H}$ → classify region and compute $C$)

**Given:** A 2×2 matrix $\mathbf{H}$ (possibly with numeric entries).
**Asked:** What region type does this pixel belong to? What is $C$?
**Inference:**
1. Inspect off-diagonal and diagonal entries to identify structure (e.g., zero off-diagonal and one zero diagonal entry → edge).
2. Compute $\det\mathbf{H}=H_{11}H_{22}-H_{12}^2$ and $\operatorname{tr}\mathbf{H}=H_{11}+H_{22}$.
3. Compute $C=\det\mathbf{H}-\alpha(\operatorname{tr}\mathbf{H})^2$.
4. Classify: $C>0$ → corner; $C<0$ → edge; $|C|\approx 0$ → flat.

> [!example]
> **Chessboard examples (W3T):**
>
> | Region | $\mathbf{H}$ | $C$ ($\alpha=0.04$) | Classification |
> |---|---|---|---|
> | Flat | $\bigl(\begin{smallmatrix}0&0\\0&0\end{smallmatrix}\bigr)$ | $0$ | Flat |
> | Vertical edge | $\bigl(\begin{smallmatrix}181847.63&0\\0&0\end{smallmatrix}\bigr)$ | $-1{,}330{,}485{,}740.6$ | Edge |
> | Horizontal edge | $\bigl(\begin{smallmatrix}0&0\\0&125885.7\end{smallmatrix}\bigr)$ | $-630{,}885{,}699.8$ | Edge |
> | Corner | $\bigl(\begin{smallmatrix}25578.8&7197.8\\7197.8&25578.8\end{smallmatrix}\bigr)$ | $497{,}783{,}558.17$ | Corner |
>
> For the flat case: $H_{11}=0,\ H_{22}=0$ → both gradients zero everywhere in the window → $E$ is flat.
> For the vertical edge: $I_y\approx 0$ so $H_{22}=0$; intensity varies only horizontally ($I_x$ large) → $C\ll 0$.
> For the corner: both diagonal entries comparable and large → both $\lambda$ large → $C\gg 0$.

---

## A5. Standard implementation

### A5a. Setup

- **Input:** Grayscale image $I(x,y)$ (pixel values).
- **Output:** Set of $(x,y)$ corner locations (after threshold + NMS).
- **Parameters:**
  - Pre-smoothing $\sigma_1$ (typically 1–2 px Gaussian).
  - Gradient kernel (e.g., $[-1\ 0\ 1]$ / $[-1\ 0\ 1]^\top$, or Sobel).
  - Gaussian window $\sigma_2$ for gradient-product smoothing (implements the window function $w$).
  - Harris sensitivity $\alpha \in [0.04, 0.06]$ (typically $0.04$).
  - Threshold $T$ on $C$.
  - NMS window size.
- **Notation:**
  - $I_x = \partial I/\partial x$, $I_y = \partial I/\partial y$ — gradient images.
  - $g(\cdot)$ — Gaussian filtering operator.
  - $a \circ b$ — Hadamard (element-wise) product.
  - $\mathbf{H}$ — structure tensor (2×2, per-pixel).
  - $C$ — cornerness map (scalar image).

### A5b. Steps

**Step 1 — Gaussian blur.**
$$I \leftarrow g_{\sigma_1}(I)$$
Suppress noise before differentiation.

**Step 2 — Compute image gradients.**
$$I_x = \frac{\partial I}{\partial x}, \qquad I_y = \frac{\partial I}{\partial y}$$
Typically via convolution with $[-1\ 0\ 1]$ (horizontal) and $[-1\ 0\ 1]^\top$ (vertical). Boundary pixels set to 0.

**Step 3 — Compute gradient-product images.**
$$I_x^2 = I_x \circ I_x, \qquad I_y^2 = I_y \circ I_y, \qquad I_xI_y = I_x \circ I_y$$
Element-wise products, one image per entry of $\mathbf{H}$.

**Step 4 — Gaussian-filter each product (weighted windowing).**
$$\tilde{A} = g(I_x^2), \qquad \tilde{B} = g(I_xI_y), \qquad \tilde{C} = g(I_y^2)$$
This implements $\sum_{x',y'} w(x',y')(\cdot)$ with $w$ a Gaussian.

**Step 5 — Assemble the structure tensor $\mathbf{H}$ at every pixel.**
$$\mathbf{H} = \begin{bmatrix}\tilde{A} & \tilde{B} \\ \tilde{B} & \tilde{C}\end{bmatrix}$$
$\mathbf{H}$ is symmetric by construction ($H_{12}=H_{21}=\tilde{B}$). See [[eigenvalues-eigenvectors]] for properties of symmetric matrices.

**Step 6 — Compute cornerness map $C$.**
$$C = \det\mathbf{H} - \alpha\,(\operatorname{tr}\mathbf{H})^2 = \tilde{A}\tilde{C} - \tilde{B}^2 - \alpha(\tilde{A}+\tilde{C})^2$$
Equivalently $C = \lambda_1\lambda_2 - \alpha(\lambda_1+\lambda_2)^2$ using $\det=\lambda_1\lambda_2$, $\operatorname{tr}=\lambda_1+\lambda_2$.

**Step 7 — Threshold.**
$$\text{Keep pixel }(x,y) \iff C(x,y) > T$$

**Step 8 — Non-maximum suppression (NMS).**
Within a local window around each candidate pixel, retain $(x,y)$ only if $C(x,y)$ is the maximum in the neighbourhood. See [[non-maximum-suppression]].

---

## A6. Variations

### Gradient kernel choices

The standard kernel pair $[-1\ 0\ 1]$ / $[-1\ 0\ 1]^\top$ is a simple finite difference. Alternatives:

| Kernel | Description | When used |
|---|---|---|
| $[-1\ 0\ 1]$ (central diff.) | Simplest, 3-tap | Default in course examples; fast |
| Sobel ($\frac{1}{8}\bigl[\begin{smallmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{smallmatrix}\bigr]$) | Weighted 3×3; smooths perpendicular direction | More robust to noise; common in practice |
| Prewitt | Uniform 3×3 weights | Less common; less noise suppression than Sobel |
| Scharr | Optimised for rotational isotropy | When rotation invariance of gradient magnitude is critical |

Kernel choice affects the entries of $\mathbf{H}$ and therefore $C$ (see [[image-filtering-and-edge-detection]] for filter details).

### Eigenvalue-based classification (direct, without $C$)

Instead of computing $C$, directly inspect $\lambda_1, \lambda_2$ (the eigenvalues of $\mathbf{H}$):
- Both $\lambda_1,\lambda_2 \ll 1$: flat.
- $\lambda_1 \gg \lambda_2$ or $\lambda_2 \gg \lambda_1$: edge.
- Both large and comparable: corner.

The det/trace shortcut (Step 6) is preferred to avoid the cost of eigendecomposition. See [[eigenvalues-eigenvectors]].

### Window function choice

| Window $w(x',y')$ | Effect |
|---|---|
| Uniform (box) | Equal weight; simple; may include irrelevant pixels |
| Gaussian | Smooth, distance-weighted; preferred; reduces ringing |

Gaussian window makes $\mathbf{H}$ less sensitive to exact pixel position and is the standard choice.

### Shi-Tomasi (Good Features to Track) variant

Replace cornerness $C$ with $\min(\lambda_1, \lambda_2)$. A pixel is a corner iff $\min(\lambda_1,\lambda_2) > T$. Simpler classification; requires explicit eigenvalues but avoids the $\alpha$ parameter. Closer to the eigenvalue classification boundary directly.

### Scale-invariant extensions

Harris alone is not scale-invariant. Extensions (e.g., **Harris-Laplace**, **SIFT**) detect corners at multiple scales by additionally searching for scale-space extrema. See [[feature-descriptors]] for scale-invariant descriptor pipelines.

---

> [!note]
> **Link to auto-correlation.** The structure tensor $\mathbf{H}$ is derived by substituting the Taylor linearisation of $I$ into the SSD auto-correlation $E(u,v)$ from [[feature-points-and-autocorrelation]]. The ellipse $[u\ v]\mathbf{H}[u\ v]^\top = \text{const}$ is a level set of the approximate $E$ surface. Semi-axes of the ellipse are $\lambda_{\max}^{-1/2}$ (short, fast-change direction) and $\lambda_{\min}^{-1/2}$ (long, slow-change direction) — an **inverse** relationship: larger eigenvalue → steeper $E$ → shorter ellipse axis.

---

> [!example]
> **Full worked example (W3T / W6T quiz): $\mathbf{H}=(2219,84;\,84,115)$**
>
> **Given** 5×5 image patch, 3×3 analysis window, gradient kernels $[-1\ 0\ 1]$ and $[-1\ 0\ 1]^\top$, uniform weights ($w=1$), $\alpha=0.04$.
>
> **$I_x$ values (inner 3×3):**
> $$\begin{pmatrix}-14&-17&-16\\-15&-16&-17\\-14&-16&-16\end{pmatrix}$$
>
> **$I_y$ values (inner 3×3):**
> $$\begin{pmatrix}-2&-5&-6\\-2&0&-1\\2&4&5\end{pmatrix}$$
>
> **Assemble $\mathbf{H}$:**
> $$\sum w\,I_x^2 = 14^2+17^2+16^2+15^2+16^2+17^2+14^2+16^2+16^2 = 2219$$
> $$\sum w\,I_y^2 = 2^2+5^2+6^2+2^2+0^2+1^2+2^2+4^2+5^2 = 115$$
> $$\sum w\,I_xI_y = (-14)(-2)+(-17)(-5)+(-16)(-6)+(-15)(-2)+(-16)(0)+(-17)(-1)+(-14)(2)+(-16)(4)+(-16)(5) = 84$$
> $$\mathbf{H} = \begin{pmatrix}2219 & 84 \\ 84 & 115\end{pmatrix}$$
>
> **Cornerness:**
> $$\det\mathbf{H} = 2219\times115 - 84^2 = 255185 - 7056 = 248129$$
> $$\operatorname{tr}\mathbf{H} = 2219 + 115 = 2334$$
> $$C = 248129 - 0.04\times2334^2 = 248129 - 217902.24 = 30226.76$$
>
> $C = 30226.76 > 0$ → **corner**.
