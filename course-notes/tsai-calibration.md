# Tsai Single-Camera Calibration

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-Level Overview]]
  - [[#Key Definitions]]
- [[#A3. Strengths, Shortcomings & Limitations]]
  - [[#Likely Exam Questions]]
- [[#A4. Directions of Reasoning]]
- [[#A5. Standard Implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Tsai calibration recovers the physically interpretable **intrinsic** parameters ($f$, $s_x$, $c_x$, $c_y$, $d_x$, $d_y$) and **extrinsic** parameters ($\mathbf{R}$, $\mathbf{t}$) of a single camera by decomposing the projection, rather than returning a raw opaque $3\times4$ matrix $\mathbf{P}$. Use it when you need individual, meaningful camera parameters — for stereo rectification, depth reconstruction, or robotic grasping — and you have access to $\geq7$ **non-coplanar** 3D reference points with known world coordinates and measured pixel correspondences. It is a two-stage analytical method (linear solve + nonlinear refinement), making it faster to initialise than purely iterative approaches. See [[camera-calibration-dlt]] for the simpler DLT which does not decompose $\mathbf{P}$, and [[pinhole-camera-model]] for the underlying projection model.

---

## A2. High-Level Overview

The method proceeds in two phases (linear then nonlinear):

1. **Convert** pixel coords $(u,v)$ to centred metric image-plane coords $(x_p,y_p)$ in mm.
2. **Form** an $n\times7$ linear system $\mathbf{M}\mathbf{L}=\mathbf{x}_p$ from all point correspondences; solve for the 7-vector $\mathbf{L}$ via Moore–Penrose pseudoinverse.
3. **Recover** $|t_y|$ from the orthonormality of row 2 of $\mathbf{R}$.
4. **Recover** $s_x$ from the orthonormality of row 1 of $\mathbf{R}$.
5. **Sign-disambiguate** $t_y$ by projecting a far reference point and comparing signs.
6. **Reconstruct** rows 1 & 2 of $\mathbf{R}$ and $t_x$ from $\mathbf{L}$ with the correct $s_x$, $t_y$.
7. **Compute** row 3 of $\mathbf{R}$ via cross product + $\lambda$ normalisation.
8. **Recover** $f$ and $t_z$ from a second linear system (Moore–Penrose again).
9. **Refine** all parameters jointly with Levenberg–Marquardt to minimise reprojection error.

### Key Definitions

| Term | Definition |
|------|-----------|
| **WRF** | World Reference Frame — 3D world coords $(X,Y,Z)$ in mm |
| **CRF** | Camera Reference Frame — 3D camera coords $(x,y,z)$ in mm |
| Image plane | 2D coords $(x_p,y_p)$ in mm, origin at principal point |
| Pixels | $(u,v)$; $u$ horizontal (right), $v$ vertical (down or up — convention matters) |
| $f$ | Focal length in mm |
| $s_x$ | Horizontal pixel scaling factor (uncertainty in $x$ pixel pitch); initially $1$ |
| $d_x,\,d_y$ | Physical pixel sizes in mm (hardware spec) |
| $(c_x,c_y)$ | Principal point in pixels; assumed $= (W/2,\,H/2)$ |
| $\mathbf{R}$ | $3\times3$ rotation matrix, **WRF** $\to$ **CRF**; $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$, $\det(\mathbf{R})=1$ |
| $\mathbf{t}=(t_x,t_y,t_z)^\top$ | Translation vector in mm |
| $\mathbf{L}=[a_1,\ldots,a_7]^\top$ | 7-vector of unknowns in the first linear system (scaled combinations of $\mathbf{R}$, $\mathbf{t}$, $s_x$) |
| $\mathbf{m}^\top$ | Row of the design matrix $\mathbf{M}$ for one correspondence |
| $\lambda$ | Normalisation scalar for row 3 of $\mathbf{R}$ |
| $U_{y,i},\,U_{z,i}$ | Intermediate scalars in Step 8 |
| Reprojection error | $e_i=\|\mathbf{p}_m^{(i)}-\tilde{\mathbf{p}}^{(i)}\|_2$ — Euclidean distance (pixels) between measured and projected image point |
| Moore–Penrose pseudoinverse | Least-squares solution $\hat{\mathbf{x}}=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$ for overdetermined $\mathbf{A}\mathbf{x}=\mathbf{b}$ |

---

## A3. Strengths, Shortcomings & Limitations

**Strengths**
- Produces **physically interpretable** parameters ($f$, $s_x$, $\mathbf{R}$, $\mathbf{t}$ individually), unlike raw DLT.
- **Closed-form analytical initialisation** — no need to guess starting values for the nonlinear step.
- Works well with as few as 7 correspondences; robust when overdetermined with many points.
- $\mathbf{R}$ orthonormality is enforced explicitly (cross-product step + normalisation).

**Shortcomings / Limitations**
- Requires $\geq7$ **non-coplanar** 3D reference points — a planar target is insufficient; cannot use a flat checkerboard alone (see Zhang).
- Assumes the principal point is at the image centre (no decentering distortion).
- No radial or tangential lens distortion model in the basic form (apply distortion correction as a preprocessing step, or extend the model).
- Sign ambiguity in the linear system requires a separate geometric test (Step 5).
- Numerical precision is critical: **do not round intermediate values** — the error compounds severely through Steps 3–8.
- The nonlinear refinement (LM) can diverge if the analytical initialisation is poor (rare but possible with degenerate point configurations).

> [!warning]
> **Non-coplanar requirement is an exam trap.** The 7+ points must span 3D space. Points all lying on one plane make the system rank-deficient. The GoPro tutorial uses a 3-panel chessboard corner target specifically to guarantee non-coplanarity.

### Likely Exam Questions

**Q:** What is the minimum number of calibration points required by Tsai, and what geometric constraint must they satisfy?

**A:** At least **7 non-coplanar 3D reference points**, each with known world coordinates $(X,Y,Z)$ and measured pixel coordinates $(u,v)$. They must be non-coplanar so the linear system $\mathbf{M}\mathbf{L}=\mathbf{x}_p$ has full rank; coplanar points make the system rank-deficient and $\mathbf{L}$ unrecoverable.

---

**Q:** How is $|t_y|$ recovered from the solution vector $\mathbf{L}=[a_1,\ldots,a_7]^\top$?

**A:** From the definition $a_5=r_{21}/t_y$, $a_6=r_{22}/t_y$, $a_7=r_{23}/t_y$, and the orthonormality constraint $r_{21}^2+r_{22}^2+r_{23}^2=1$:
$$|t_y|=\frac{1}{\sqrt{a_5^2+a_6^2+a_7^2}}$$
GoPro H3 numeric result: $|t_y|=9.9527$ mm.

---

**Q:** How is $s_x$ recovered?

**A:** From $a_1=s_x r_{11}/t_y$, $a_2=s_x r_{12}/t_y$, $a_3=s_x r_{13}/t_y$ and $r_{11}^2+r_{12}^2+r_{13}^2=1$:
$$s_x=|t_y|\sqrt{a_1^2+a_2^2+a_3^2}$$
GoPro H3 numeric result: $s_x=1.0018$.

---

**Q:** How is row 3 of $\mathbf{R}$ computed, and why?

**A:** Via the cross product $\mathbf{r}_3 = \mathbf{r}_1\times\mathbf{r}_2$ (right-hand rule, since $\det(\mathbf{R})=1$). In practice:
$$r_{31}=\lambda(r_{12}r_{23}-r_{13}r_{22}),\quad r_{32}=-\lambda(r_{11}r_{23}-r_{13}r_{21}),\quad r_{33}=\lambda(r_{11}r_{22}-r_{12}r_{21})$$
where $\lambda=1/\sqrt{r_{31}^2+r_{32}^2+r_{33}^2}$ normalises to unit length. This ensures $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$ exactly.

---

**Q:** Why does a sign ambiguity arise in $t_y$, and how is it resolved?

**A:** The linear system $x_p=\mathbf{m}^\top\mathbf{L}$ is invariant to a global negation of all entries of $\mathbf{L}$, so both $+|t_y|$ and $-|t_y|$ satisfy the system equally. Resolution: pick the 3D reference point whose image is furthest from the principal point; compute provisional $S_x=r_{11}X+r_{12}Y+r_{13}Z+t_x$ and $S_y=r_{21}X+r_{22}Y+r_{23}Z+|t_y|$. If $\text{sign}(S_x)=\text{sign}(x_p)$ and $\text{sign}(S_y)=\text{sign}(y_p)$, keep $t_y=+|t_y|$; otherwise flip to $t_y=-|t_y|$.

---

**Q:** Describe the two-stage linear solve structure of Tsai calibration.

**A:** **Stage 1 (Steps 2–7):** Solve the $n\times7$ system $\mathbf{M}\mathbf{L}=\mathbf{x}_p$ for $\mathbf{L}$, then extract $|t_y|$, $s_x$, rows 1–2 of $\mathbf{R}$, $t_x$, and row 3 of $\mathbf{R}$. **Stage 2 (Step 8):** Form an $n\times2$ system in $(f,t_z)$ from the $y$-projection equation; solve via pseudoinverse. Both are overdetermined least-squares problems solved by $\hat{\mathbf{x}}=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$.

---

## A4. Directions of Reasoning

### Forward / Standard: pixel correspondences → camera parameters

**Given:** $n\geq7$ pairs $\{(X_i,Y_i,Z_i),\,(u_i,v_i)\}$ (non-coplanar), plus $d_x,d_y,c_x,c_y$.

**Asked:** $f$, $s_x$, $\mathbf{R}$, $\mathbf{t}=(t_x,t_y,t_z)^\top$.

**Key inference:** Run Steps 1–8 exactly as given below. The orthonormality of $\mathbf{R}$ ($\mathbf{R}^\top\mathbf{R}=\mathbf{I}$) provides the two scalar equations that disentangle $|t_y|$ and $s_x$ from the composite parameters in $\mathbf{L}$. Once $\mathbf{R}$ and $\mathbf{t}$ are known up to sign, the sign test resolves the ambiguity, and a second linear solve yields $f,t_z$.

### Reverse / Inferential A: calibrated camera → project 3D point to pixel (reprojection)

**Given:** Calibrated $f,s_x,d_x,d_y,c_x,c_y,\mathbf{R},\mathbf{t}$ and a new 3D point $(X,Y,Z)$.

**Asked:** Predicted pixel $(u,v)$ — used to compute reprojection error.

**Key manipulation:** Apply the three-stage pipeline:

$$\begin{pmatrix}s\tilde{u}\\s\tilde{v}\\s\end{pmatrix} = \underbrace{\begin{pmatrix}s_x/d_x & 0 & c_x \\ 0 & 1/d_y & c_y \\ 0 & 0 & 1\end{pmatrix}}_{\text{plane}\to\text{pixel}} \underbrace{\begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}}_{\text{CRF}\to\text{plane}} \underbrace{\begin{pmatrix}\mathbf{R}&\mathbf{t}\\\mathbf{0}^\top&1\end{pmatrix}}_{\text{WRF}\to\text{CRF}} \begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$

Dehomogenise: $(u,v) = (\tilde{u}/s,\tilde{v}/s)$. Then $e_i=\|\mathbf{p}_m^{(i)}-(u,v)^\top\|_2$.

> [!note]
> Signs of $f$ and $t_z$ depend on the Step 1 convention chosen. With the $y_p=d_y(v-c_y)$ (upward) convention used in the GoPro tutorial, both $f=-2.709$ mm and $t_z=-58.69$ mm are negative. With the downward convention $y_p=d_y(c_y-v)$ the signs may differ.

### Reverse / Inferential B: recover sign after solving

**Given:** $\mathbf{L}$ solved; $|t_y|$ and $s_x$ computed; provisional $r_{ij}$ with $|t_y|$.

**Asked:** True sign of $t_y$.

**Key manipulation:** Geometric projection test — see Step 5 below.

---

## A5. Standard Implementation

### A5a. Setup

| Item | Value |
|------|-------|
| Inputs | $n\geq7$ non-coplanar triples $(X_i,Y_i,Z_i)$ + measured pixels $(u_i,v_i)$ |
| Known camera params | $d_x,d_y$ (mm/px), $c_x=W/2$, $c_y=H/2$, initial $s_x=1$ |
| Unknowns | $f,s_x,\mathbf{R}\in SO(3),\mathbf{t}\in\mathbb{R}^3$ |
| Output | All parameters + reprojection error |
| GoPro H3 hardware | $d_x=d_y=0.00155$ mm, image $3000\times2250$ px, $c_x=1500$, $c_y=1125$ |
| Assumption | No lens distortion (apply correction beforehand if needed); principal point at image centre |

> [!warning]
> Do **not** round intermediate results. Full floating-point precision must be carried through all steps. The GoPro example shows values to 16 significant figures.

### A5b. Steps

**Step 1 — Pixel to centred metric image-plane coords**

Two sign conventions exist (both appear in lecture slides):

$$\text{(downward } y_C\text{):}\quad x_p = s_x\,d_x(u-c_x),\qquad y_p = d_y(c_y-v)$$

$$\text{(upward } y_C\text{):}\quad x_p = s_x\,d_x(u-c_x),\qquad y_p = d_y(v-c_y)$$

> [!note]
> The GoPro tutorial (W10T) uses the **upward** convention $y_p=d_y(v-c_y)$, which causes $f$ and $t_z$ to be **negative** in Step 8. The W9L_pt2 slides also show the downward form $y_p=d_y(c_y-v)$. The convention must be **consistent** throughout — pick one and stick with it.

*GoPro H3 example (point 1):* $(u,v)=(1487.917,1439.352)$, $s_x=1$:
$$x_p=1\times0.00155\times(1487.917-1500)=-0.019\text{ mm}$$
$$y_p=0.00155\times(1439.352-1125)=0.487\text{ mm}$$

---

**Step 2 — Build and solve the linear system for $\mathbf{L}$**

Starting from the perspective projection (after dividing out the homogeneous scale and cross-multiplying to eliminate the denominator), each correspondence yields one linear equation $x_{p,i}=\mathbf{m}_i^\top\mathbf{L}$:

$$\mathbf{m}_i^\top = \bigl[y_{p,i}X_i,\;y_{p,i}Y_i,\;y_{p,i}Z_i,\;y_{p,i},\;-x_{p,i}X_i,\;-x_{p,i}Y_i,\;-x_{p,i}Z_i\bigr]$$

$$\mathbf{L} = \frac{1}{t_y}\bigl[s_xr_{11},\;s_xr_{12},\;s_xr_{13},\;s_xt_x,\;r_{21},\;r_{22},\;r_{23}\bigr]^\top = [a_1,\ldots,a_7]^\top$$

Stack $n$ equations: $\underbrace{\mathbf{M}}_{n\times7}\,\mathbf{L}=\underbrace{\mathbf{x}_p}_{n\times1}$. Solve by Moore–Penrose:

$$\mathbf{L} = (\mathbf{M}^\top\mathbf{M})^{-1}\mathbf{M}^\top\mathbf{x}_p$$

*GoPro H3 numeric $\mathbf{L}$ (from $n=140$ points):*
$$\mathbf{L}=\begin{bmatrix}0.06982\\-0.07231\\0.00537\\0.20333\\0.01364\\0.00616\\-0.09935\end{bmatrix}$$

---

**Step 3 — Recover $|t_y|$**

Use $r_{21}^2+r_{22}^2+r_{23}^2=1$ (unit-norm row of $\mathbf{R}$) and $a_5=r_{21}/t_y$, $a_6=r_{22}/t_y$, $a_7=r_{23}/t_y$:

$$\boxed{|t_y|=\frac{1}{\sqrt{a_5^2+a_6^2+a_7^2}}}$$

*GoPro H3:* $|t_y|=1/\sqrt{(0.01364)^2+(0.00616)^2+(-0.09935)^2}=\mathbf{9.9527}$ mm.

*(Calibration-cube example from W9L_pt2: $|t_y|=176.2675$ mm.)*

---

**Step 4 — Recover $s_x$**

Use $r_{11}^2+r_{12}^2+r_{13}^2=1$ and $a_i=s_xr_{1i}/t_y$ for $i=1,2,3$:

$$\boxed{s_x=|t_y|\sqrt{a_1^2+a_2^2+a_3^2}}$$

*GoPro H3:* $s_x=9.9527\times\sqrt{(0.06982)^2+(-0.07231)^2+(0.00537)^2}=\mathbf{1.0018}$.

*(Calibration-cube example: $s_x=1.072051$.)*

---

**Step 5 — Sign disambiguation of $t_y$**

Compute provisional $r_{ij}$ using $|t_y|$ (formula as in Step 6, but with $|t_y|$). Choose the reference point $(X,Y,Z)$ whose image lies furthest from the principal point. Compute:

$$S_x=r_{11}X+r_{12}Y+r_{13}Z+t_x,\qquad S_y=r_{21}X+r_{22}Y+r_{23}Z+|t_y|$$

If $\text{sign}(S_x)=\text{sign}(x_p)$ **and** $\text{sign}(S_y)=\text{sign}(y_p)$: set $t_y=+|t_y|$. Otherwise: $t_y=-|t_y|$.

*GoPro H3 example:* Furthest point $(X,Y,Z)=(0.0,24.4,31.5)$, $(x_p,y_p)=(-1.006,-1.430)$.
$$S_x=0+(-0.717)(24.4)+(0.05)(31.5)+2.020=-13.9\quad[\text{sign}(-)=\text{sign}(x_p)\ \checkmark]$$
$$S_y=0+(0.060)(24.4)+(-0.985)(31.5)+9.953=-19.6\quad[\text{sign}(-)=\text{sign}(y_p)\ \checkmark]$$
No flip: $t_y=+9.9527$ mm.

---

**Step 6 — Recover rows 1–2 of $\mathbf{R}$ and $t_x$**

With the true signed $t_y$ and $s_x$:

$$r_{11}=a_1\frac{t_y}{s_x},\quad r_{12}=a_2\frac{t_y}{s_x},\quad r_{13}=a_3\frac{t_y}{s_x}$$
$$r_{21}=a_5\,t_y,\quad r_{22}=a_6\,t_y,\quad r_{23}=a_7\,t_y$$
$$t_x=a_4\frac{t_y}{s_x}$$

*GoPro H3 ($t_y/s_x=9.9527/1.0018=9.933$):*
$$r_{11}=0.695,\;r_{12}=-0.715,\;r_{13}=0.050,\;r_{21}=0.139,\;r_{22}=0.060,\;r_{23}=-0.985,\;t_x=2.016$$

---

**Step 7 — Recover row 3 of $\mathbf{R}$ via cross product**

$\mathbf{r}_3=\mathbf{r}_1\times\mathbf{r}_2$, computed using $2\times2$ determinants:

$$r_{31}=\lambda\begin{vmatrix}r_{12}&r_{13}\\r_{22}&r_{23}\end{vmatrix},\quad r_{32}=-\lambda\begin{vmatrix}r_{11}&r_{13}\\r_{21}&r_{23}\end{vmatrix},\quad r_{33}=\lambda\begin{vmatrix}r_{11}&r_{12}\\r_{21}&r_{22}\end{vmatrix}$$

where $\lambda=1/\sqrt{r_{31}^2+r_{32}^2+r_{33}^2}$ (normalise to unit length).

*GoPro H3:*
$$r_{31}=\lambda\bigl((-0.715)(-0.985)-(0.050)(0.060)\bigr)=0.701\lambda$$
$$r_{32}=-\lambda\bigl((0.695)(-0.985)-(0.050)(0.139)\bigr)=0.692\lambda$$
$$r_{33}=\lambda\bigl((0.695)(0.060)-(-0.715)(0.139)\bigr)=0.058\lambda$$
$$\lambda=1/\sqrt{0.701^2+0.692^2+0.058^2}=1.013$$

$$\mathbf{R}=\begin{pmatrix}0.695&-0.715&0.050\\0.139&0.060&-0.985\\0.710&0.701&0.059\end{pmatrix}$$

---

**Step 8 — Recover $f$ and $t_z$**

From the $y$-projection equation for each point $i$:

$$y_{p,i}=f\,\frac{r_{21}X_i+r_{22}Y_i+r_{23}Z_i+t_y}{r_{31}X_i+r_{32}Y_i+r_{33}Z_i+t_z}$$

Define:
$$U_{y,i}=r_{21}X_i+r_{22}Y_i+r_{23}Z_i+t_y,\qquad U_{z,i}=r_{31}X_i+r_{32}Y_i+r_{33}Z_i$$

Rearranged as a linear equation in $(f,\,t_z)$:
$$(U_{y,i},\;-y_{p,i})\begin{pmatrix}f\\t_z\end{pmatrix}=U_{z,i}\,y_{p,i}$$

Stack $n\geq2$ equations into $\mathbf{M}_{n\times2}\,(f,t_z)^\top=\mathbf{b}_{n\times1}$ and solve:
$$\begin{pmatrix}f\\t_z\end{pmatrix}=(\mathbf{M}^\top\mathbf{M})^{-1}\mathbf{M}^\top\mathbf{b}$$

*GoPro H3 results:*
$$f=\mathbf{-2.709}\text{ mm},\qquad t_z=\mathbf{-58.69}\text{ mm}$$

(Negative signs are correct for the upward $y_p=d_y(v-c_y)$ convention.)

---

**Step 9 — Nonlinear refinement**

Use the analytical solution from Steps 1–8 as initialisation for **Levenberg–Marquardt** (gradient descent fallback) minimising total reprojection error $\sum_i e_i^2$. SVD-based pseudoinverse is preferred over direct $(\mathbf{M}^\top\mathbf{M})^{-1}$ for numerical stability. See [[singular-value-decomposition]].

---

**Post-calibration sanity checks (W11T)**

$$f>0\quad\text{(or }<0\text{ if sign convention says so)}$$
$$(c_x,c_y)\approx(W/2,\,H/2)$$
$$\mathbf{R}^\top\mathbf{R}\approx\mathbf{I},\quad\det(\mathbf{R})\approx1$$
$$\|\mathbf{t}\|_2\approx d(O_c,O_w)$$

---

## A6. Variations

### Unknown pixel dimensions ($d_x,d_y$ not available)

When the sensor pixel pitch is unknown, work in pixel units. Define:

$$x_p'=\frac{x_p}{d_x}=s_x(u-c_x),\qquad y_p'=\frac{y_p}{d_y}=(v-c_y)$$

and replace $f$ with the pixel-unit focal lengths:

$$f_x=\frac{f}{d_x},\qquad f_y=\frac{f}{d_y}$$

Step 1 becomes: $x_p'=s_x(u-c_x)$, $y_p'=(v-c_y)$. Steps 2–7 are structurally identical with $(x_p',y_p')$ in place of $(x_p,y_p)$. Step 8 recovers $f_x$ and $f_y$ (possibly different) instead of $f$. This variant naturally handles asymmetric sensor pixels and is the standard formulation for the intrinsic matrix:

$$\mathbf{K}=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}$$

See also [[camera-calibration-dlt]] which always works in this pixel-unit space.

---

### Zhang calibration (comparison)

| Property | Tsai | Zhang |
|----------|------|-------|
| Target | $\geq7$ **non-coplanar** 3D points | Planar checkerboard ($Z=0$), multiple views |
| Output | $f,s_x,(c_x,c_y),\mathbf{R},\mathbf{t}$ | $f_x,f_y,(c_x,c_y),\mathbf{R},\mathbf{t}$, distortion coeffs |
| Distortion | Minimal (pre-correct separately) | Full radial + tangential model |
| Usage | Sparse industrial setups; stereo rigs | OpenCV `calibrateCamera`; consumer apps |
| Weakness | Non-coplanar constraint; no distortion | Needs multiple images; corner detection sensitive |

---

### DeepCalib (comparison)

| Property | Tsai | DeepCalib |
|----------|------|-----------|
| Approach | Analytical + LM refinement | CNN regression/classification |
| Input | Known 3D–2D correspondences | Single image |
| Output | Full intrinsics + extrinsics | $f$ + distortion coefficient |
| Strengths | Interpretable; fast; no training data | Handles complex fisheye distortions; online use |
| Weaknesses | Non-coplanar points needed; limited distortion | Needs large labelled dataset; lower interpretability; GPU cost |

See [[deep-learning-for-stereo-and-calibration]] for DeepCalib architecture details.

---

### Related topics

- [[pinhole-camera-model]] — the projection model Tsai calibrates
- [[camera-calibration-dlt]] — simpler DLT without parameter decomposition
- [[singular-value-decomposition]] — preferred numerics for pseudoinverse
- [[calibration-error-analysis]] — reprojection error metrics and outlier removal
- [[backward-projection-and-ray-intersection]] — inverse use of calibrated camera
- [[stereo-rectification]] — downstream use of $\mathbf{K},\mathbf{R},\mathbf{t}$
- [[deep-learning-for-stereo-and-calibration]] — DeepCalib and CREStereo
