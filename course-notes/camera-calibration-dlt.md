# Camera Calibration — DLT (Generic Linear Method)

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

Given a set of known 3D world points and their corresponding 2D pixel positions in an image, estimate the full $3\times4$ projection matrix $\mathbf{P}$ that encodes the entire imaging chain. This is the **generic** (DLT-style) approach: it treats $\mathbf{P}$ as 11 free parameters and solves a linear system directly — no separation of intrinsics from extrinsics. It is used when you need a quick projection model but do not require interpretable camera parameters; [[tsai-calibration]] is preferred when you need $f$, $s_x$, $\mathbf{R}$, $\mathbf{t}$ explicitly.

---

## A2. High-level overview

**Coordinate frames and named quantities:**

| Symbol | Frame / Space | Meaning |
|---|---|---|
| $(X,Y,Z)$ | WRF (world, mm) | 3D world point coordinates |
| $(x,y,z)$ | CRF (camera, mm) | 3D camera-frame coordinates |
| $(x_p, y_p)$ | Image plane (mm, centred at principal point) | Projected point before pixel scaling |
| $(u,v)$ | Pixel coordinates | Final 2D image measurement |
| $\mathbf{R}_{3\times3}$, $\mathbf{t}_{3\times1}$ | Extrinsic | Rotation and translation WRF→CRF |
| $\mathbf{K}_{3\times3}$ | Intrinsic | Focal length + principal point + pixel size |
| $\mathbf{P}_{3\times4}$ | Projection | Combined camera matrix $\mathbf{P}=\mathbf{K}[\mathbf{R}\mid\mathbf{t}]$ |
| $f$ | Camera | Focal length (mm) |
| $d_x, d_y$ | Camera | Pixel size in mm (x and y) |
| $f_x = f/d_x$, $f_y = f/d_y$ | Intrinsic | Focal length in pixels |
| $(c_x, c_y)$ | Intrinsic | Principal point (pixels) |
| $s$ | Scale | Homogeneous scale; equals $z$ (depth in CRF) |
| $\mathbf{A}$ | Linear system | Measurement matrix ($2n\times11$) |
| $\mathbf{h}$ | Linear system | Vector of 11 unknowns (entries of $\mathbf{P}$) |
| $\mathbf{b}$ | Linear system | Observed pixel measurements ($2n\times1$) |

**Steps (no math):**

1. Place a calibration object with known 3D geometry in the scene; measure $n\geq6$ 3D↔2D correspondences.
2. Write the 4-step imaging chain: WRF→CRF (extrinsic), CRF→image-plane mm (perspective), mm→pixels (scaling), combined as $\mathbf{P}=\mathbf{K}[\mathbf{R}\mid\mathbf{t}]$.
3. Parameterise $\mathbf{P}$ as a generic $3\times4$ matrix with the last entry normalised to 1 (11 unknowns).
4. Eliminate the scale $s$ from the homogeneous equations to get two linear equations per correspondence.
5. Stack all correspondences into $\mathbf{A}\mathbf{h}=\mathbf{b}$ and solve via Moore–Penrose pseudoinverse.

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Entirely linear — closed-form solution, no iterative optimisation needed.
- Simple to implement; applies to any calibration object with known 3D geometry.
- Same DLT principle as [[homography-and-dlt]] (just extended to 3D→2D).

**Shortcomings / limitations:**
- Produces a "black-box" $\mathbf{P}$: intrinsics and extrinsics are entangled — you cannot directly read off $f$, $\mathbf{R}$, $\mathbf{t}$ from the raw entries.
- Does not model lens distortion; distortion must be handled separately (see [[lens-distortion-and-removal]]).
- Sensitive to noise; accuracy degrades if correspondences are poorly distributed or nearly coplanar.
- Requires at least 6 non-coplanar point correspondences (11 unknowns, 2 equations per point).
- Normalisation of the last entry to 1 is an arbitrary choice; the scale of $\mathbf{P}$ is ambiguous.

> [!warning]
> You **cannot** directly extract $f$, $s_x$, $(c_x,c_y)$, $\mathbf{R}$, $\mathbf{t}$ by inspection of the raw $\mathbf{P}$ entries — decomposition (e.g. RQ decomposition as in Tsai) is required. This is the primary motivation for [[tsai-calibration]].

### Likely exam questions

**Q:** Why does the generic calibration method require at least 6 point correspondences?
**A:** $\mathbf{P}$ has 12 entries; normalising the $(3,4)$ entry to 1 leaves 11 unknowns. Each correspondence gives 2 linear equations (one for $u$, one for $v$), so $\lceil 11/2 \rceil = 6$ correspondences are the minimum to make the system determined.

**Q:** Write the two linear equations obtained from a single 3D↔2D correspondence $(X,Y,Z)\leftrightarrow(u,v)$ after eliminating the scale $s$.
**A:**
$$u = aX + bY + cZ + d - iXu - jYu - kZu$$
$$v = eX + fY + gZ + h - iXv - jYv - kZv$$
(with 11 unknowns $a,b,c,d,e,f,g,h,i,j,k$).

**Q:** What is the Moore–Penrose pseudoinverse solution to $\mathbf{A}\mathbf{h}=\mathbf{b}$, and when is it applicable?
**A:** $\mathbf{h} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$; applicable when $\mathbf{A}$ is tall (more rows than columns, i.e. $n\geq6$) and has full column rank (correspondences not degenerate/coplanar).

**Q:** List the 4 steps in the full camera calibration chain and identify which step accounts for each of: rotation/translation, focal length, and pixel size.
**A:**
1. WRF→CRF: rotation $\mathbf{R}$ and translation $\mathbf{t}$ (extrinsic, $4\times4$).
2. CRF→image plane (mm): $3\times4$ matrix with focal length $f$ on diagonal (perspective projection).
3. mm→pixels: $3\times3$ matrix with $1/d_x$, $1/d_y$ and principal point $(c_x,c_y)$ (pixel scaling).
4. Combine: $\mathbf{P} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]$ ($3\times4$).

**Q:** What is the difference between Case 1 and Case 2 image frame conventions, and how does it affect $\mathbf{K}$?
**A:** Case 1: origin top-left, $v$ increases downward → $-1/d_y$ in the scaling matrix. Case 2: origin bottom-left, $v$ increases upward → $+1/d_y$. The sign of the $v$ scale term in $\mathbf{K}$ flips.

**Q:** Given $\mathbf{P}$, how do you project a new 3D point $(X,Y,Z)$ to pixel coordinates $(u,v)$?
**A:** Form $\tilde{\mathbf{M}}=(X,Y,Z,1)^\top$, compute $\tilde{\mathbf{p}} = \mathbf{P}\tilde{\mathbf{M}} = (su, sv, s)^\top$, then $u = (su)/s$, $v = (sv)/s$.

---

## A4. Directions of reasoning

### Forward (calibration): 3D correspondences → $\mathbf{P}$

- **Given:** $n\geq6$ pairs $\{(X_i,Y_i,Z_i),(u_i,v_i)\}$.
- **Asked:** the $3\times4$ matrix $\mathbf{P}$ (encoded as $\mathbf{h}$, 11 unknowns).
- **Key steps:** expand homogeneous projection equations, eliminate $s$, stack into $\mathbf{A}\mathbf{h}=\mathbf{b}$, solve with pseudoinverse.

### Reverse (projection): $\mathbf{P}$ → projected pixel

- **Given:** a calibrated $\mathbf{P}$ and a new 3D point $(X,Y,Z)$.
- **Asked:** its 2D pixel location $(u,v)$.
- **Key inference:** $s(u,v,1)^\top = \mathbf{P}(X,Y,Z,1)^\top$; divide the first two components by the third to dehomogenise.

> [!note]
> The reverse direction does NOT let you recover intrinsics/extrinsics from $\mathbf{P}$ by inspection. Decomposing $\mathbf{P}$ into $\mathbf{K}$, $\mathbf{R}$, $\mathbf{t}$ requires further algebraic decomposition (RQ factorisation), which is what [[tsai-calibration]] provides.

### Inferential: understanding why ≥6 points are needed

- **Given:** the generic matrix has 12 entries; last entry normalised to 1 → 11 unknowns.
- **Each point yields:** 2 independent linear equations (after eliminating $s$).
- **Conclusion:** $\lceil 11/2 \rceil = 6$ points minimum; more points give an overdetermined system, improving robustness via least squares.

---

## A5. Standard implementation

### A5a. Setup

- **Inputs:** $n\geq6$ 3D↔2D correspondences $\{(X_i,Y_i,Z_i)\leftrightarrow(u_i,v_i)\}_{i=1}^n$ from a calibration object with known geometry.
- **Output:** vector $\mathbf{h}=(a,b,c,d,e,f,g,h,i,j,k)^\top$ — the 11 free entries of $\mathbf{P}$.
- **Assumption:** last entry of $\mathbf{P}$ normalised to 1; points not all coplanar; $s=z\neq0$ (no point at camera centre).
- **Notation:** WRF coords $(X,Y,Z)$; pixel coords $(u,v)$; scale $s=z$ (depth in CRF).

### A5b. Steps

**Step 1 — The 4-step imaging chain (conceptual)**

Chain three transforms to get $\mathbf{P} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]$:

$$\begin{pmatrix}su\\sv\\s\end{pmatrix} = \underbrace{\begin{pmatrix}\tfrac{1}{d_x}&0&c_x\\0&\pm\tfrac{1}{d_y}&c_y\\0&0&1\end{pmatrix}}_{\text{Step 3: mm→px}} \underbrace{\begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}}_{\text{Step 2: CRF→mm}} \underbrace{\begin{pmatrix}\mathbf{R}_{3\times3}&\mathbf{t}_{3\times1}\\0&1\end{pmatrix}}_{\text{Step 1: WRF→CRF}} \begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$

Sign of $1/d_y$: $-$ for Case 1 (top-left origin), $+$ for Case 2 (bottom-left origin).

**Step 2 — Generic parameterisation (11 unknowns)**

Write the combined $\mathbf{P}$ as a generic $3\times4$ matrix with last entry = 1:

$$\begin{pmatrix}su\\sv\\s\end{pmatrix} = \begin{pmatrix}a&b&c&d\\e&f&g&h\\i&j&k&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}, \quad s\neq0$$

Expanding the three rows:
$$su = aX+bY+cZ+d, \quad sv = eX+fY+gZ+h, \quad s = iX+jY+kZ+1$$

**Step 3 — Eliminate $s$ to linearise**

Substitute $s = iX+jY+kZ+1$ into the first two equations:

$$u = aX+bY+cZ+d - iXu - jYu - kZu$$
$$v = eX+fY+gZ+h - iXv - jYv - kZv$$

These are linear in the 11 unknowns $(a,b,c,d,e,f,g,h,i,j,k)$.

**Step 4 — Assemble the measurement matrix $\mathbf{A}$**

Stack two rows per correspondence; for point $i$ with $(X_i,Y_i,Z_i,u_i,v_i)$:

$$\text{row}_{2i-1}: \begin{pmatrix}X_i&Y_i&Z_i&1&0&0&0&0&-X_iu_i&-Y_iu_i&-Z_iu_i\end{pmatrix}$$
$$\text{row}_{2i}: \begin{pmatrix}0&0&0&0&X_i&Y_i&Z_i&1&-X_iv_i&-Y_iv_i&-Z_iv_i\end{pmatrix}$$

Full system:

$$\underbrace{\begin{pmatrix}X_1&Y_1&Z_1&1&0&0&0&0&-X_1u_1&-Y_1u_1&-Z_1u_1\\0&0&0&0&X_1&Y_1&Z_1&1&-X_1v_1&-Y_1v_1&-Z_1v_1\\X_2&Y_2&Z_2&1&0&0&0&0&-X_2u_2&-Y_2u_2&-Z_2u_2\\0&0&0&0&X_2&Y_2&Z_2&1&-X_2v_2&-Y_2v_2&-Z_2v_2\\\vdots&&&&&&&&&&\vdots\end{pmatrix}}_{\mathbf{A}\;(2n\times11)}\begin{pmatrix}a\\b\\c\\d\\e\\f\\g\\h\\i\\j\\k\end{pmatrix}=\begin{pmatrix}u_1\\v_1\\u_2\\v_2\\\vdots\end{pmatrix}$$

Compactly: $\mathbf{A}\mathbf{h}=\mathbf{b}$.

**Step 5 — Solve via Moore–Penrose pseudoinverse**

$$\mathbf{h} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$$

Reconstruct $\mathbf{P}$ by reshaping $\mathbf{h}$ into the first two rows of $\mathbf{P}$ and appending $(i,j,k,1)$ as the third row.

> [!note]
> The pseudoinverse minimises the least-squares residual $\|\mathbf{A}\mathbf{h}-\mathbf{b}\|^2$ and is equivalent to solving the normal equations. For numerical stability, [[singular-value-decomposition]] can be used instead; when the last entry is NOT normalised to 1 the homogeneous form $\mathbf{A}\mathbf{h}=\mathbf{0}$ is solved by the right singular vector corresponding to the smallest singular value.

---

## A6. Variations

### Homogeneous DLT (no normalisation constraint)

- Instead of fixing the $(3,4)$ entry to 1, treat all 12 entries as unknowns and enforce $\|\mathbf{h}\|=1$.
- Reformulated as $\mathbf{A}\mathbf{h}=\mathbf{0}$; solved by SVD as $\mathbf{h} = $ last right singular vector of $\mathbf{A}$.
- Same idea as used in [[homography-and-dlt]] (2D→2D case with $3\times3$ $\mathbf{H}$) — the DLT there has 8 DOF (9 entries, scale-normalised), requiring ≥4 correspondences.
- See [[singular-value-decomposition]] for the mechanics.

### 2D→2D DLT (Homography)

- Special case where $Z=0$ (planar calibration object) or all world points are coplanar.
- The $3\times4$ $\mathbf{P}$ degenerates to a $3\times3$ homography $\mathbf{H}$ (one column collapses).
- See [[homography-and-dlt]].

### Tsai Calibration (structured decomposition)

- After obtaining $\mathbf{P}$ (or directly from correspondences), **decomposes** $\mathbf{P}$ into interpretable parameters: $f$, $s_x$ (horizontal scale), $\mathbf{R}$, $\mathbf{t}$, plus radial distortion.
- Preferred in practice because it gives physically meaningful parameters.
- See [[tsai-calibration]].

### Case 1 vs Case 2 pixel-frame conventions

- **Case 1** (top-left origin, $v$ downward): pixel-scaling matrix uses $-1/d_y$; $c_x=W/2$, $c_y=H/2$.
- **Case 2** (bottom-left origin, $v$ upward): pixel-scaling matrix uses $+1/d_y$.
- The choice affects the sign convention inside $\mathbf{K}$ but not the DLT setup itself — correspondences $(u_i,v_i)$ just need to be measured consistently in one convention.

---

## Related topics

- [[pinhole-camera-model]] — geometric basis (similar triangles, focal length, optical centre)
- [[homogeneous-coordinates-and-transformations]] — why scale $s$ appears; how to dehomogenise
- [[homography-and-dlt]] — same DLT principle in the 2D→2D setting
- [[tsai-calibration]] — decomposes $\mathbf{P}$ into interpretable intrinsic/extrinsic parameters
- [[singular-value-decomposition]] — alternative/more stable solver for the homogeneous DLT
- [[calibration-error-analysis]] — how to assess residual back-projection error
- [[lens-distortion-and-removal]] — distortion effects not captured by the linear $\mathbf{P}$ model
