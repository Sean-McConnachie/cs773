# Backward Projection & Ray Intersection

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps — the 7-step back-projection procedure]]
- [[#A6. Variations]]
  - [[#Ray equation & ray-plane intersection (supporting geometry)]]
  - [[#2D line intersection]]
  - [[#3D line intersection (stereo, least-squares midpoint)]]
  - [[#Stereo 3D calibration error]]

---

## A1. Purpose

Backward projection reconstructs a **3D world point** $\widetilde{\mathbf{M}}_w = (\widetilde{X}_w, \widetilde{Y}_w, \widetilde{Z}_w)^\top$ from a **measured 2D pixel** $(u, v)^\top$, given full camera calibration and knowledge of the **plane** the point lies on (e.g. $X_w = 0$ or $Z_w = 0$). It is the inverse of forward projection and is used to evaluate **3D cube calibration error** (compare $\widetilde{\mathbf{M}}_w$ to the true corner), **stereo 3D error** (intersect two back-projected rays), and to answer "where in the world is this pixel?" given a depth constraint.

---

## A2. High-level overview

**Steps (no math):**

1. Convert pixel $(u,v)$ to metric image-plane coordinates $(x_p, y_p)$ (mm).
2. Extend to a 3D point on the CRF virtual image plane by appending $-f$.
3. Apply the inverse extrinsic transform to lift the point into the WRF — this gives one point $\mathbf{M}_i^W$ on the ray.
4. Apply the inverse extrinsic transform to the CRF optical centre $\mathbf{O}_c = (0,0,0,1)^\top$ to find its WRF location $\mathbf{O}_c^W$ — the ray origin.
5. Write the parametric ray through $\mathbf{O}_c^W$ and $\mathbf{M}_i^W$.
6. Intersect the ray with the known constraint plane; solve for parameter $t$.
7. Substitute $t$ back to read off $\widetilde{\mathbf{M}}_w$.

**Key definitions:**

| Term | Definition |
|---|---|
| **WRF** | World Reference Frame — global 3D space, coordinates $(X_w, Y_w, Z_w)$ in mm |
| **CRF** | Camera Reference Frame — 3D space centred at optical centre $O_c$, coordinates $(x, y, z)$ in mm |
| Image plane (mm) | 2D plane at $z = -f$ in CRF; coordinates $(x_p, y_p)$ centred at the principal point |
| Pixel space | $(u, v)$ in pixels; origin at image corner |
| $f$ | Focal length (mm); note $f < 0$ in Tsai calibration |
| $d_x, d_y$ | Physical pixel size (mm/px) in $x$ and $y$ |
| $(c_x, c_y)$ | Principal point in pixels |
| $s_x$ | $x$-pixel uncertainty scale factor (close to 1; from Tsai calibration) |
| $\mathbf{R}$ | $3\times3$ rotation matrix (WRF $\to$ CRF); orthonormal, $\mathbf{R}^{-1} = \mathbf{R}^\top$ |
| $\mathbf{t}$ | Translation vector (mm); completes the extrinsic transform $(\mathbf{R}\mid\mathbf{t})$ |
| $(\mathbf{R}\mid\mathbf{t})^{-1}$ | Inverse extrinsic: $\begin{pmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}$ |
| $\mathbf{M}_i^W$ | One point on the back-projected ray expressed in WRF |
| $\mathbf{O}_c^W$ | Optical centre expressed in WRF (the other point defining the ray) |
| $t$ (ray param.) | Scalar along the parametric ray; not to be confused with translation $\mathbf{t}$ |
| $\widetilde{\mathbf{M}}_w$ | Estimated world point after ray-plane intersection |

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Gives physically meaningful error in 3D world units (mm), unlike 2D reprojection error (pixels).
- Works for any plane constraint — $X_w = 0$, $Z_w = 0$, or any $AX + BY + CZ + D = 0$.
- Exploits that $\mathbf{R}^{-1} = \mathbf{R}^\top$, so no matrix inversion is needed beyond a transpose + dot product.

**Shortcomings / limitations:**
- Requires a known constraint plane — underdetermined otherwise (infinitely many 3D points project to the same pixel).
- Sensitive to calibration errors: small error in $\mathbf{R}, \mathbf{t}, f$ shifts the ray direction, causing proportionally larger errors at greater depth.
- In stereo, two rays rarely intersect exactly (3D lines in general position are skew); must approximate with the midpoint of the shortest segment.
- Coordinate convention (C1 vs C2) changes the sign of $y_p$ and $f$; must be consistent.

### Likely exam questions

**Q:** Write out the 7-step backward projection procedure.
**A:** (1) Pixel $\to$ metric: $(x_p, y_p)$. (2) Append $-f$: CRF point $(x_p, y_p, -f)^\top$. (3) Apply $(\mathbf{R}\mid\mathbf{t})^{-1}$ to the homogeneous 4-vector to get $\mathbf{M}_i^W$. (4) Apply $(\mathbf{R}\mid\mathbf{t})^{-1}$ to $\mathbf{O}_c = (0,0,0,1)^\top$ to get $\mathbf{O}_c^W$. (5) Parametric ray $\mathbf{P}(t) = \mathbf{O}_c^W + t(\mathbf{M}_i^W - \mathbf{O}_c^W)$. (6) Intersect with known plane — for $X_w = 0$: $t = -X_c^W/(X_i^W - X_c^W)$. (7) Substitute to find $\widetilde{\mathbf{M}}_w$.

---

**Q:** Using the lecture worked example, pixel $(u,v) = (1487.917, 1439.352)$ on the $X_w = 0$ plane. Calibration: $f = -2.709$ mm, $s_x = 1.0018$, $d_x = d_y = 0.00155$ mm/px, $c_x = 1500$, $c_y = 1125$. The computed $\mathbf{O}_c^W \approx (36.885, 17.094, 39.684)^\top$ and $\mathbf{M}_i^W \approx (-0.01876, 0.48725, 2.70869)^\top$. What is $t$ and $\widetilde{\mathbf{M}}_w$?
**A:** $t = -36.885 / (-0.01876 - 36.885) \approx 20.807$. Then $\widetilde{Y}_w = 17.094 + 20.807(0.48725 - 17.094) \approx 0.014$ mm, $\widetilde{Z}_w \approx 3.357$ mm. So $\widetilde{\mathbf{M}}_w \approx (0, 0.014, 3.357)^\top$ vs. true $(0, 0, 3.4)^\top$.

---

**Q:** Quiz — $f = -5$ mm, $c = (119, 194)$ px, $d_x = d_y = 0.001$ mm/px, $s_x = 1$, pixel $\mathbf{p} = (174, 73)^\top$, plane $Z_w = 0$. Given $\mathbf{O}_c^W = (23.72, -2.0, -1.545)^\top$ and $\mathbf{M}_i^W = (19.249, -2.121, 0.694)^\top$, find $\widetilde{\mathbf{M}}_w$.
**A:** $t = -(-1.545)/(0.694 - (-1.545)) = 1.545/2.239 \approx 0.69$. $\widetilde{X}_w = 23.72 + 0.69(19.249 - 23.72) \approx 20.6$. $\widetilde{Y}_w = -2.0 + 0.69(-2.121 - (-2.0)) \approx -2.1$. $\widetilde{Z}_w = 0$. So $\widetilde{\mathbf{M}}_w \approx (20.6, -2.1, 0)^\top$.

---

**Q:** Give the formula for $(\mathbf{R}\mid\mathbf{t})^{-1}$ and explain why it is exact.
**A:** $(\mathbf{R}\mid\mathbf{t})^{-1} = \begin{pmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}$. It is exact because $\mathbf{R}$ is orthonormal ($\mathbf{R}^{-1} = \mathbf{R}^\top$ exactly), so no numerical inversion is needed.

---

**Q:** For a ray from $\mathbf{O}_c = (0,0,0)^\top$ in direction $\mathbf{d} = (2,1,5)^\top$, find the point on the ray where $Z = 100$.
**A:** $r(t) = t\mathbf{d}$; the $Z$ component is $5t = 100 \Rightarrow t = 20$. Point $= (40, 20, 100)^\top$.

---

**Q:** Why must backward projection use a constraint plane, and what happens in stereo when no such plane is assumed?
**A:** A single pixel defines a ray, not a unique 3D point — the problem is underdetermined. In stereo, two rays (one per camera) are expected to intersect, but in practice they are skew. The estimated 3D point is taken as the **midpoint of the shortest line segment** between the two rays, found by least-squares minimisation.

---

## A4. Directions of reasoning

**Forward (standard): $\mathbf{M}_w \to (u, v)$**
- Given: 3D world point $\mathbf{M}_w$, calibration $(\mathbf{K}, \mathbf{R}, \mathbf{t}, f)$.
- Asked: pixel $(u, v)$.
- Key steps: apply $(\mathbf{R}\mid\mathbf{t})$ to go WRF $\to$ CRF; perspective divide; apply $\mathbf{K}$ (pixel conversion).
- This is covered in [[pinhole-camera-model]] and [[tsai-calibration]].

**Reverse (backward): $(u, v) + \text{plane} \to \widetilde{\mathbf{M}}_w$**
- Given: pixel $(u, v)$, calibration, plane equation.
- Asked: 3D world point $\widetilde{\mathbf{M}}_w$.
- Key inference: invert $(\mathbf{R}\mid\mathbf{t})$ to bring two known points (metric image point at $-f$, and the optical centre) into WRF; form a ray; intersect with the plane.
- The plane constraint breaks the underdeterminacy. Without it, only a ray is recovered.

> [!note]
> This topic is inherently the "A4 reverse" of forward projection. Every application of backward projection is an instance of recovering the world-space input from the image-space output.

---

## A5. Standard implementation

### a. Setup

| | |
|---|---|
| **Inputs** | Pixel $(u, v)^\top$; calibration params $f, s_x, d_x, d_y, c_x, c_y$; extrinsic $(\mathbf{R}\mid\mathbf{t})$; plane equation |
| **Output** | Estimated world point $\widetilde{\mathbf{M}}_w = (\widetilde{X}_w, \widetilde{Y}_w, \widetilde{Z}_w)^\top$ |
| **Convention** | C1 unless stated: origin bottom-left, $v$ up, $y_p = d_y(v - c_y)$; image plane at $z = -f$ in CRF |
| **Assumption** | The point lies on a known plane in WRF |

### b. Steps — the 7-step back-projection procedure

**Step 1 — Pixel to metric image coordinates**

$$x_p = s_x\, d_x\,(u - c_x), \qquad y_p = d_y\,(v - c_y)$$

> [!note]
> C2 convention (origin top-left, $v$ down): $y_p = d_y(c_y - v)$. The sign flip propagates through all subsequent steps — be consistent.

**Step 2 — Extend to CRF virtual image plane**

Append the focal length (with sign) to get a 3D point on the virtual image plane inside the camera:

$$\mathbf{q} = (x_p,\; y_p,\; -f)^\top \in \mathbb{R}^3$$

The image plane sits at $z = -f$ in CRF. If $f < 0$ (Tsai convention), then $-f > 0$, so the point is in front of the optical centre.

**Step 3 — Lift to WRF (one point on the ray)**

Apply the inverse extrinsic transform to the homogeneous vector:

$$\mathbf{M}_i^W = (\mathbf{R}\mid\mathbf{t})^{-1}\begin{pmatrix}x_p \\ y_p \\ -f \\ 1\end{pmatrix} = (X_i^W,\; Y_i^W,\; Z_i^W,\; 1)^\top$$

where $(\mathbf{R}\mid\mathbf{t})^{-1} = \begin{pmatrix}\mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}$.

**Step 4 — Optical centre in WRF (ray origin)**

$$\mathbf{O}_c^W = (\mathbf{R}\mid\mathbf{t})^{-1}\begin{pmatrix}0\\0\\0\\1\end{pmatrix} = -\mathbf{R}^\top\mathbf{t} \in \mathbb{R}^3$$

(The $4\times1$ result; discard the trailing $1$.)

**Step 5 — Parametric ray in WRF**

$$\mathbf{P}(t) = \mathbf{O}_c^W + t\!\left(\mathbf{M}_i^W - \mathbf{O}_c^W\right), \quad t \in \mathbb{R}$$

At $t = 0$: the ray is at the optical centre. At $t = 1$: the ray is at $\mathbf{M}_i^W$. The 3D world point lies at some $t > 1$ (scene is behind the image plane).

**Step 6 — Intersect with the constraint plane**

For plane $X_w = 0$ (left face of calibration cube):

$$X_c^W + t(X_i^W - X_c^W) = 0 \implies \boxed{t = \frac{-X_c^W}{X_i^W - X_c^W}}$$

For plane $Z_w = 0$ (bottom face):

$$Z_c^W + t(Z_i^W - Z_c^W) = 0 \implies t = \frac{-Z_c^W}{Z_i^W - Z_c^W}$$

General plane $AX + BY + CZ + D = 0$: substitute the ray and solve the resulting scalar linear equation for $t$.

**Step 7 — Back-projected world point**

$$\widetilde{X}_w = X_c^W + t(X_i^W - X_c^W)$$
$$\widetilde{Y}_w = Y_c^W + t(Y_i^W - Y_c^W)$$
$$\widetilde{Z}_w = Z_c^W + t(Z_i^W - Z_c^W)$$
$$\widetilde{\mathbf{M}}_w = (\widetilde{X}_w,\; \widetilde{Y}_w,\; \widetilde{Z}_w)^\top$$

> [!example]
> **Worked example (W11L pp. 22–26, chessboard cube, plane $X_w = 0$):**
> - Input: $(u, v) = (1487.917, 1439.352)$; $f = -2.709$ mm, $s_x = 1.0018$, $d_x = d_y = 0.00155$ mm/px, $c_x = 1500$, $c_y = 1125$.
> - Step 1: $x_p = 1.0018 \times 0.00155 \times (1487.917 - 1500) = -0.019$ mm; $y_p = 0.00155 \times (1439.352 - 1125) = 0.487$ mm.
> - Step 2: $(x_p, y_p, -f) = (-0.019, 0.487, 2.709)$ mm.
> - Step 3: $\mathbf{M}_i^W \approx (-0.01876, 0.48725, 2.70869, 1)^\top$.
> - Step 4: $\mathbf{O}_c^W \approx (36.885, 17.094, 39.684, 1)^\top$.
> - Step 6: $t = -36.885 / (-0.01876 - 36.885) \approx 20.807$.
> - Step 7: $\widetilde{\mathbf{M}}_w \approx (0.0,\; 0.014,\; 3.357)^\top$. True point: $(0, 0, 3.4)^\top$. Close agreement.

> [!example]
> **Quiz (W11L p. 27, plane $Z_w = 0$):**
> - $f = -5$ mm, $c = (119, 194)$ px, $d_x = d_y = 0.001$ mm/px, $s_x = 1$, pixel $(174, 73)^\top$.
> - $x_p = 0.001(174 - 119) = 0.055$ mm; $y_p = 0.001(73 - 194) = -0.121$ mm.
> - $\mathbf{O}_c^W = (23.72, -2.0, -1.545)^\top$; $\mathbf{M}_i^W = (19.249, -2.121, 0.694)^\top$.
> - $t = -(-1.545)/(0.694 - (-1.545)) = 1.545/2.239 \approx 0.69$.
> - $\widetilde{\mathbf{M}}_w \approx (20.6,\; -2.1,\; 0.0)^\top$.

> [!warning]
> Do not round intermediate results. The quiz slides explicitly flag this as a common mistake in exams.

---

## A6. Variations

### Ray equation & ray-plane intersection (supporting geometry)

The general ray from origin $\mathbf{o}$ in direction $\mathbf{d}$:

$$r(t) = \mathbf{o} + t\,\mathbf{d}$$

> [!example]
> **W10L worked example:** $\mathbf{o} = (0,0,0)^\top$, $\mathbf{d} = (2,1,5)^\top$, plane $Z = 100$.
> $5t = 100 \Rightarrow t = 20$. Point $= r(20) = (40, 20, 100)^\top$.

For a plane $\mathbf{n}^\top \mathbf{x} + D = 0$ (normal $\mathbf{n}$, offset $D$), the ray hits the plane at:

$$t = \frac{-D - \mathbf{n}^\top \mathbf{o}}{\mathbf{n}^\top \mathbf{d}}$$

(No intersection if $\mathbf{n}^\top \mathbf{d} = 0$, i.e. ray is parallel to plane.)

---

### 2D line intersection

Two 2D parametric line segments: $\mathbf{p}(t_{12}) = \mathbf{p}_1 + t_{12}(\mathbf{p}_2 - \mathbf{p}_1)$ and $\mathbf{q}(t_{34}) = \mathbf{p}_3 + t_{34}(\mathbf{p}_4 - \mathbf{p}_3)$.

Setting $\mathbf{p}(t_{12}) = \mathbf{q}(t_{34})$ yields the $2\times2$ linear system:

$$\begin{bmatrix} x_2 - x_1 & -(x_4 - x_3) \\ y_2 - y_1 & -(y_4 - y_3) \end{bmatrix} \begin{bmatrix} t_{12} \\ t_{34} \end{bmatrix} = \begin{bmatrix} x_3 - x_1 \\ y_3 - y_1 \end{bmatrix}$$

- Intersection exists iff the $2\times2$ matrix is non-singular.
- Intersection point: $\mathbf{p}_1 + t_{12}(\mathbf{p}_2 - \mathbf{p}_1)$.
- Within both segments iff $0 \le t_{12} \le 1$ and $0 \le t_{34} \le 1$.
- Normal form of a 2D line: $x\sin\theta - y\cos\theta + d = 0$; unit normal $\mathbf{n} = (\sin\theta, -\cos\theta)^\top$.

---

### 3D line intersection (stereo, least-squares midpoint)

Two 3D rays $\mathbf{p}_1 + t_{21}(\mathbf{p}_2 - \mathbf{p}_1)$ and $\mathbf{p}_3 + t_{43}(\mathbf{p}_4 - \mathbf{p}_3)$ almost never intersect exactly in practice (skew lines). Minimise the squared distance between points on the two rays:

$$\min_{t_{21},\, t_{43}} \|\mathbf{p}_1 + t_{21}\Delta_{21} - \mathbf{p}_3 - t_{43}\Delta_{43}\|^2$$

where $\Delta_{ij} = \mathbf{p}_i - \mathbf{p}_j$.

This gives the $2\times2$ normal-equation system:

$$\begin{bmatrix} \Delta_{21}^\top\Delta_{21} & \Delta_{21}^\top\Delta_{43} \\ \Delta_{43}^\top\Delta_{21} & \Delta_{43}^\top\Delta_{43} \end{bmatrix} \begin{bmatrix} t_{21} \\ t_{43} \end{bmatrix} = \begin{bmatrix} -\Delta_{13}^\top\Delta_{21} \\ -\Delta_{13}^\top\Delta_{43} \end{bmatrix}$$

where $\Delta_{13} = \mathbf{p}_1 - \mathbf{p}_3$.

Solved as:

$$\begin{bmatrix} t_{21} \\ t_{43} \end{bmatrix} = -\begin{bmatrix} \Delta_{21}^\top\Delta_{21} & \Delta_{21}^\top\Delta_{43} \\ \Delta_{21}^\top\Delta_{43} & \Delta_{43}^\top\Delta_{43} \end{bmatrix}^{-1} \begin{bmatrix} \Delta_{13}^\top\Delta_{21} \\ \Delta_{13}^\top\Delta_{43} \end{bmatrix}$$

The estimated 3D intersection point is the **midpoint of the shortest segment** between the two rays.

See also [[vector-matrix-algebra]], [[homogeneous-coordinates-and-transformations]].

---

### Stereo 3D calibration error

Apply backward projection from both the left and right cameras to produce two rays in WRF:

$$\mathbf{r}_L = [\mathbf{O}_{cL}^W,\; \mathbf{M}_{iL}^W], \qquad \mathbf{r}_R = [\mathbf{O}_{cR}^W,\; \mathbf{M}_{iR}^W]$$

Intersect $\mathbf{r}_L$ and $\mathbf{r}_R$ using the 3D least-squares midpoint method above to get $\widetilde{\mathbf{M}}_w$.

Compare to the true 3D point $\mathbf{M}_w$; aggregate over $n$ correspondences:

$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (X_i - \widetilde{X}_i)^2 + (Y_i - \widetilde{Y}_i)^2 + (Z_i - \widetilde{Z}_i)^2}$$

**Left-to-right error:** back-project each left-camera pixel to WRF, forward-project into the right camera, compare to the right-camera measured pixel.

**Right-to-left error:** the symmetric operation.

See [[calibration-error-analysis]] for the full error framework, [[tsai-calibration]] for parameter estimation, and [[pinhole-camera-model]] for the forward projection pipeline.
