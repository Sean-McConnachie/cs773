# Calibration Error Analysis

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Key definitions]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps — reprojection (image) error]]
  - [[#A5b-ii. Steps — cube (3D) back-projection error]]
  - [[#A5b-iii. Steps — stereo calibration error]]
- [[#A6. Variations]]

---

## A1. Purpose

After running [[tsai-calibration]] (or any camera calibration), calibration-error analysis quantifies how well the recovered parameters $(\mathbf{K}, \mathbf{R}, \mathbf{t}, \kappa_1)$ explain the observed data. Three complementary error types are computed — in pixels (reprojection), in mm (cube back-projection), and in mm via stereo ray intersection — so that different failure modes become visible. The results also drive sanity checks that flag implausible parameter values before the camera is used in a downstream application such as [[depth-from-disparity]].

---

## A2. High-level overview

1. **Reprojection / image error** — forward-project each known WRF point through $\mathbf{P}$ to $(\tilde{u},\tilde{v})$; compare to measured $(u,v)$ in pixels.
2. **Cube / 3D error** — back-project each measured pixel onto the calibration-cube plane (via ray-plane intersection; see [[backward-projection-and-ray-intersection]]); compare estimated WRF point to true WRF point in mm.
3. **Stereo error** — trace one ray per camera per match; intersect them; compare midpoint to true 3D in mm.
4. **Aggregate metrics** — compute RMSE, AE, SD, PE over all $n$ points.
5. **Sanity checks** — verify intrinsic, extrinsic, and stereo parameter plausibility.

### Key definitions

| Term | Definition |
|---|---|
| WRF | World Reference Frame — 3D scene coords $(X,Y,Z)$ in mm |
| CRF | Camera Reference Frame — 3D camera coords $(x,y,z)$ in mm |
| $(u,v)$ | Measured pixel coordinates |
| $(\tilde{u},\tilde{v})$ | Projected pixel coordinates (from calibration) |
| $\mathbf{p}_i = (u_i,v_i)^\top$ | Measured image point |
| $\tilde{\mathbf{p}}_i = (\tilde{u}_i,\tilde{v}_i)^\top$ | Forward-projected estimate |
| $\mathbf{M}_w = (X,Y,Z)^\top$ | True WRF calibration point (mm) |
| $\tilde{\mathbf{M}}_w = (\tilde{X},\tilde{Y},\tilde{Z})^\top$ | Back-projected WRF estimate (mm) |
| $e_i$ | Per-point calibration error (pixels or mm) |
| $b$ | Stereo baseline $= \|{}^W\mathbf{O}_L - {}^W\mathbf{O}_R\|$ (mm) |
| Systematic error | Consistent bias, correctable by a model (e.g. [[lens-distortion-and-removal]]) |
| Gross error | Blunder; must be detected and eliminated |
| Random error | Stochastic noise; reduced by statistical models |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Reprojection error is cheap to compute (only forward projection needed) — good for optimisation objective in [[tsai-calibration]].
- Cube / 3D error is physically interpretable (mm in scene space).
- Stereo error isolates the inter-camera geometry separately from each camera's intrinsics.
- Error-distribution plots (histogram + scatter of $(u,v)$ positions) reveal spatial patterns.

**Shortcomings / limitations**
- Reprojection error can be low even with physically wrong parameters (e.g. compensating $f$ and $\|\mathbf{t}\|$ errors).
- Cube error requires knowing the plane equation; error grows if the plane is misidentified.
- Stereo rays rarely intersect exactly in 3D — midpoint approximation introduces its own error.
- All three metrics are degraded by gross errors (outliers); outlier removal is necessary before reporting.
- Errors are largest near image borders due to unmodelled higher-order distortion.

### Likely exam questions

**Q:** Define reprojection error and state its formula.
**A:** $e_i = \|\mathbf{p}_i - \tilde{\mathbf{p}}_i\|_2$ — the Euclidean distance in pixels between the measured point $(u_i,v_i)$ and the forward-projected estimate $(\tilde{u}_i,\tilde{v}_i)$ obtained from the calibration parameters. Example: $\mathbf{p}_i=(420,315)$, $\tilde{\mathbf{p}}_i=(425,310)$ gives $e_i = \sqrt{5^2+5^2} = 7.07$ px.

---

**Q:** What is the RMSE for 2D image error and for 3D cube error?
**A:**
$$RMSE_{2D} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl[(u_i-\tilde{u}_i)^2+(v_i-\tilde{v}_i)^2\bigr]}$$
$$RMSE_{3D} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl[(X_i-\tilde{X}_i)^2+(Y_i-\tilde{Y}_i)^2+(Z_i-\tilde{Z}_i)^2\bigr]}$$
The lecture example gives $RMSE_{3D} = 0.0465$ mm.

---

**Q:** Name the three types of measurement error and explain how each is handled.
**A:** (1) **Systematic** — predictable, correctable by a model (e.g. radial distortion via $\kappa_1$). (2) **Gross** — a blunder (misidentified corner, wrong correspondence); must be detected and removed as an outlier. (3) **Random** — stochastic noise; reduced by stochastic modelling (e.g. Gaussian smoothing, increasing correspondence count).

---

**Q:** List four calibration sanity checks and what each verifies.
**A:** (1) $f > 0$ — focal length physically meaningful. (2) $(c_x,c_y) \approx (W/2, H/2)$ — principal point near image centre. (3) $\mathbf{R}^\top\mathbf{R} \approx \mathbf{I}$, $\det\mathbf{R} \approx 1$ — rotation matrix is orthonormal. (4) $\|\mathbf{t}\| \approx$ camera-to-object distance — translation magnitude plausible. Additional: baseline $b = \|{}^W\mathbf{O}_L - {}^W\mathbf{O}_R\| \approx$ measured ruler distance.

---

**Q:** How is the stereo baseline $b$ computed from calibration outputs?
**A:** $b = \|{}^W\mathbf{O}_L - {}^W\mathbf{O}_R\|$ where ${}^W\mathbf{O}_k = (\mathbf{R}|\mathbf{t})_k^{-1}\mathbf{O}_c$ and $\mathbf{O}_c = (0,0,0,1)^\top$ in CRF, and $(\mathbf{R}|\mathbf{t})^{-1} = (\mathbf{R}^\top \mid -\mathbf{R}^\top\mathbf{t})$. Lecture worked example: $b = 120$ mm.

---

**Q:** Why do stereo back-projected rays rarely intersect exactly, and how is the 3D point estimated anyway?
**A:** Due to noise and calibration error, left and right rays are skew lines in 3D. The estimated 3D point is the midpoint of the shortest segment connecting the two rays, obtained by minimising $\|\mathbf{p}_1(t_{12}) - \mathbf{p}_2(t_{34})\|^2$ (a $2\times2$ least-squares system in $t_{12}, t_{34}$).

---

## A4. Directions of reasoning

### Forward / standard (calibration → error)

- **Given:** calibration parameters $(\mathbf{K},\mathbf{R},\mathbf{t})$, true WRF points $\mathbf{M}_w^{(i)}$, measured pixels $\mathbf{p}_i$.
- **Asked:** reprojection error per point, RMSE over all points.
- **Key step:** $\tilde{\mathbf{p}}_i = \pi(\mathbf{K}[\mathbf{R}|\mathbf{t}]\,\tilde{\mathbf{M}}_w^{(i)})$, then $e_i = \|\mathbf{p}_i - \tilde{\mathbf{p}}_i\|_2$.

### Reverse / inferential (error pattern → diagnosis)

- **Given:** an observed error pattern (e.g. errors larger at image borders; barrel-shaped residual vectors; systematic offset of principal point).
- **Asked:** what calibration problem caused it?
- **Key inferences:**
  - High error near borders, low at centre → unmodelled radial [[lens-distortion-and-removal]] (systematic). Fix: include $\kappa_1$ (or $\kappa_2$ for mustache distortion).
  - Uniform large error everywhere → gross error in corner detection or wrong world-point labelling (blunder).
  - Random scatter with no spatial pattern → random noise; acceptable if below 1–2 px.
  - $(c_x,c_y)$ far from $(W/2,H/2)$ → misconfigured image crop or lens shift.
  - $\|\mathbf{t}\|$ inconsistent with physical distance → wrong scale of the calibration object (mm vs. cm confusion).
  - $\mathbf{R}^\top\mathbf{R} \not\approx \mathbf{I}$ → numerical issue in calibration solver; re-run or increase correspondences.
  - Stereo error much larger than mono error → $T_{RtoL}$ or baseline estimate is wrong; re-check $(\mathbf{R}|\mathbf{t})_{L}$, $(\mathbf{R}|\mathbf{t})_{R}$.

---

## A5. Standard implementation

### A5a. Setup

| Item | Detail |
|---|---|
| Inputs | $n$ point correspondences $\{(\mathbf{M}_w^{(i)}, \mathbf{p}_i)\}$; calibrated $\mathbf{K}$, $\mathbf{R}$, $\mathbf{t}$, $\kappa_1$ |
| Intrinsics | $f$ (mm), $d_x, d_y$ (mm/px), $(c_x,c_y)$ (px) |
| Extrinsics | $\mathbf{R}$ ($3\times3$), $\mathbf{t}=(t_x,t_y,t_z)^\top$ (mm) |
| Convention | C1: $(u,v)$ origin bottom-left, $y_p = d_y(v-c_y)$; C2: origin top-left, $y_p = d_y(c_y-v)$. Assignment 2 uses C2. |
| Output | $\{e_i\}$, RMSE, AE, SD, PE; pass/fail for each sanity check |

### A5b. Steps — reprojection (image) error

1. **Forward-project each WRF point** to pixel space:

$$s\begin{pmatrix}\tilde{u}_i\\\tilde{v}_i\\1\end{pmatrix} = \underbrace{\begin{pmatrix}\tfrac{1}{d_x}&0&c_x\\0&\tfrac{1}{d_y}&c_y\\0&0&1\end{pmatrix}}_{\mathbf{K}}\underbrace{\begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}}_{\text{proj}}\underbrace{\begin{pmatrix}\mathbf{R}&\mathbf{t}\\0&1\end{pmatrix}}_{[\mathbf{R}|\mathbf{t}]}\begin{pmatrix}X_i\\Y_i\\Z_i\\1\end{pmatrix}$$

(Use $-\tfrac{1}{d_y}$ in the C2 convention.)

2. **Dehomogenise** to get $(\tilde{u}_i, \tilde{v}_i)$.

3. **Compute per-point error:**

$$e_i = \|\mathbf{p}_i - \tilde{\mathbf{p}}_i\|_2 = \sqrt{(u_i-\tilde{u}_i)^2+(v_i-\tilde{v}_i)^2}$$

4. **Aggregate:**

$$AE = \frac{1}{n}\sum_{i=1}^n e_i, \qquad SD = \sqrt{\frac{1}{n-1}\sum_{i=1}^n(e_i-AE)^2}$$

$$RMSE_{2D} = \sqrt{\frac{1}{n}\sum_{i=1}^n\bigl[(u_i-\tilde{u}_i)^2+(v_i-\tilde{v}_i)^2\bigr]}$$

$$PE = 100\cdot\frac{\text{estimated}-\text{actual}}{\text{actual}}$$

> [!example]
> Measured $\mathbf{p}_i=(420,315)$, projected $\tilde{\mathbf{p}}_i=(425,310)$:
> $$e_i = \sqrt{(420-425)^2+(315-310)^2} = \sqrt{25+25} = 7.07 \text{ px}$$

### A5b-ii. Steps — cube (3D) back-projection error

Uses the 7-step [[backward-projection-and-ray-intersection]] procedure.

1. **Pixel to metric image coords** (C1: positive sign; C2: negative $y_p$ sign):

$$x_{p,i} = s_x d_x(u_i - c_x), \qquad y_{p,i} = d_y(v_i - c_y)$$

2. **Extend to CRF 3D point** (image plane at $z = -f$):

$$(x_{p,i},\; y_{p,i},\; -f)^\top$$

3. **Map to WRF** using extrinsic inverse $(\mathbf{R}|\mathbf{t})^{-1} = (\mathbf{R}^\top \mid -\mathbf{R}^\top\mathbf{t})$:

$$\mathbf{M}_i^W = (\mathbf{R}|\mathbf{t})^{-1}(x_{p,i},\; y_{p,i},\; -f,\; 1)^\top$$

4. **Find optical centre in WRF:**

$${}^W\mathbf{O}_c = (\mathbf{R}|\mathbf{t})^{-1}(0,0,0,1)^\top$$

5. **Form parametric ray in WRF:**

$$\mathbf{P}(t) = {}^W\mathbf{O}_c + t\!\left(\mathbf{M}_i^W - {}^W\mathbf{O}_c\right)$$

6. **Intersect with cube plane** (e.g. $X_w = 0$): solve for $t$:

$$t = \frac{-X_c^W}{X_i^W - X_c^W}$$

For a general plane $AX+BY+CZ+D=0$: $t = \frac{-(A X_c^W + B Y_c^W + C Z_c^W + D)}{A(X_i^W-X_c^W)+B(Y_i^W-Y_c^W)+C(Z_i^W-Z_c^W)}$

7. **Back-projected WRF point:**

$$\tilde{X}_w = X_c^W + t(X_i^W - X_c^W), \quad \tilde{Y}_w = Y_c^W + t(Y_i^W - Y_c^W), \quad \tilde{Z}_w = Z_c^W + t(Z_i^W - Z_c^W)$$

$$\tilde{\mathbf{M}}_w = (\tilde{X}_w, \tilde{Y}_w, \tilde{Z}_w)^\top$$

8. **Aggregate 3D error:**

$$RMSE_{3D} = \sqrt{\frac{1}{n}\sum_{i=1}^n\bigl[(X_i-\tilde{X}_i)^2+(Y_i-\tilde{Y}_i)^2+(Z_i-\tilde{Z}_i)^2\bigr]}$$

> [!example]
> Lecture worked example: $RMSE_{3D} = 0.0465$ mm (`tsai.cubeError = 0.04645539611732293`).

> [!note]
> Ray-plane worked example from slides: ${}^W\mathbf{O}_c=(0,0,0)^\top$, direction $\mathbf{d}=(2,1,5)^\top$, plane $Z_w=100$. Then $5t=100 \Rightarrow t=20$, intersection $(40,20,100)^\top$.

### A5b-iii. Steps — stereo calibration error

1. **Back-project left-image pixel** to left-camera WRF ray $r_L = [{}^W\mathbf{O}_{cL},\; \mathbf{M}_i^{W_L}]$ (steps 1–5 above, for left camera).

2. **Back-project right-image pixel** to right-camera WRF ray $r_R$ (same procedure with $(\mathbf{R}|\mathbf{t})_R$).

3. **Compute left-to-right transform:**

$$T_{RtoL} = (\mathbf{R}|\mathbf{t})_L \cdot \bigl((\mathbf{R}|\mathbf{t})_R\bigr)^{-1}$$

4. **Intersect (or approximate) two 3D rays** via least-squares minimisation:

$$\min_{t_{12},t_{34}}\|\mathbf{P}_L(t_{12}) - \mathbf{P}_R(t_{34})\|^2$$

Solved via $2\times2$ system (with $\Delta_{ij} = \mathbf{p}_i - \mathbf{p}_j$):

$$\begin{bmatrix}\Delta_{21}^\top\Delta_{21} & \Delta_{21}^\top\Delta_{43}\\\Delta_{43}^\top\Delta_{21} & \Delta_{43}^\top\Delta_{43}\end{bmatrix}\begin{bmatrix}t_{21}\\t_{43}\end{bmatrix} = \begin{bmatrix}-\Delta_{13}^\top\Delta_{21}\\-\Delta_{13}^\top\Delta_{43}\end{bmatrix}$$

5. **Estimated 3D point** $\tilde{\mathbf{M}}_i$ = midpoint of shortest inter-ray segment.

6. **3D stereo error per point:** $\|\tilde{\mathbf{M}}_i - \mathbf{M}_{3D,i}\|_2$.

7. **Aggregate:**

$$RMSE_{stereo} = \sqrt{\frac{1}{n}\sum_{i=1}^n\bigl[(X_i-\tilde{X}_i)^2+(Y_i-\tilde{Y}_i)^2+(Z_i-\tilde{Z}_i)^2\bigr]}$$

> [!note]
> **Stereo baseline computation.** ${}^W\mathbf{O}_k = (\mathbf{R}|\mathbf{t})_k^{-1}(0,0,0,1)^\top$.
> $$b = \|{}^W\mathbf{O}_L - {}^W\mathbf{O}_R\|$$
> Lecture worked example: $b = 120$ mm.

**Additional stereo error variants:**
- **Left-to-right:** project the back-projected left-ray intersection into the right image; error $= \|{}^R\tilde{\mathbf{m}}_l - \mathbf{m}_r\|$.
- **Right-to-left:** project the back-projected right-ray intersection into the left image; error $= \|{}^L\tilde{\mathbf{m}}_r - \mathbf{m}_l\|$.

---

## A6. Variations

### Sanity-check protocol

Run immediately after calibration, before any error measurement:

| Check | Criterion | Failure implication |
|---|---|---|
| Focal length | $f > 0$ | Degenerate calibration |
| Principal point | $(c_x,c_y) \approx (W/2,\, H/2)$ | Image crop / lens shift |
| Rotation validity | $\mathbf{R}^\top\mathbf{R} \approx \mathbf{I}$, $\det\mathbf{R} \approx 1$ | Solver numerical error |
| Translation magnitude | $\|\mathbf{t}\| \approx$ camera-to-object dist | Scale error (mm vs cm) |
| Baseline | $b \approx$ physical ruler measurement | Stereo extrinsic error |
| Error spatial distribution | Errors uniform, not border-concentrated | If border-high: unmodelled distortion |

### Without distortion correction

If $\kappa_1$ is not estimated, reprojection error will show a characteristic pattern — largest at the corners (barrel) or edges (pincushion). Remedy: re-run [[tsai-calibration]] including $\kappa_1$; for complex optics, add $\kappa_2$ for mustache distortion. See [[lens-distortion-and-removal]].

### K-fold cross-validation of error

Split correspondences into $k$ folds; calibrate on $k-1$, evaluate on the holdout fold. RMSE on the held-out fold better estimates generalisation than training-set error alone. Reported alongside AE and SD.

### Stereo depth accuracy propagation

Calibration error propagates to depth via $Z = fb/d$ (see [[depth-from-disparity]]). A baseline error $\Delta b$ or focal-length error $\Delta f$ creates a multiplicative depth bias; disparity noise $\Delta d$ creates depth uncertainty that grows as $Z^2$ (since $\partial Z/\partial d = -fb/d^2 = -Z^2/(fb)$).
