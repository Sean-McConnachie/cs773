# Pinhole Camera Model

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Key definitions]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
  - [[#Forward: 3D point → pixels]]
  - [[#Reverse: pixels → 3D (and why it fails)]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps]]
- [[#A6. Variations]]
  - [[#Case 1 vs Case 2 image-frame conventions]]
  - [[#Intrinsic matrix with skew]]

---

## A1. Purpose

The pinhole camera model is the standard geometric model that describes how a 3D point in the world maps to a 2D pixel position in an image. It is used whenever you need to relate world coordinates to image coordinates — the foundation of camera calibration, stereo vision, structure-from-motion, and augmented reality. You reach for it as soon as you need a quantitative relationship between scene geometry and pixel measurements.

---

## A2. High-level overview

**The projection chain (4 stages):**

1. Express the 3D point in the **World Reference Frame (WRF)**: $(X, Y, Z)$ in mm.
2. Apply the **extrinsic transform** $(\mathbf{R} \mid \mathbf{t})$ to get the point in the **Camera Reference Frame (CRF)**: $(x, y, z)$ in mm.
3. Apply **perspective projection** to get the point on the **image plane** in mm: $(x_p, y_p)$.
4. Apply the **mm-to-pixel transform** (intrinsic matrix $\mathbf{K}$) to get **pixel coordinates**: $(u, v)$.

### Key definitions

| Term | Definition |
|---|---|
| **WRF** | World Reference Frame; coordinates $(X, Y, Z)$ in physical units (mm). |
| **CRF** | Camera Reference Frame; coordinates $(x, y, z)$ in mm, origin at optical centre. |
| **Image plane** (mm) | Plane at distance $f$ from $C$; coordinates $(x_p, y_p)$ centred at principal point. |
| **Pixel coordinates** | $(u, v)$; origin at image corner (Case 1: top-left, Case 2: bottom-left). |
| **Optical centre** $C$ | The single point through which all light rays pass in the pinhole model. |
| **Focal length** $f$ | Distance (mm) from $C$ to the image plane along the optical axis. |
| **Principal point** $(c_x, c_y)$ | Intersection of the optical axis with the image plane, in pixels. |
| **Pixel size** $d_x, d_y$ | Physical size of one pixel (mm/pixel) in $x$ and $y$ directions. |
| **Extrinsic parameters** | $\mathbf{R}$ (3×3 rotation) and $\mathbf{t}$ (3×1 translation); encode camera pose in the world. |
| **Intrinsic matrix** $\mathbf{K}$ | 3×3 matrix encoding $f_x, f_y, c_x, c_y$; maps CRF → pixels. |
| **Projection matrix** $\mathbf{P}$ | 3×4 matrix $\mathbf{K}[\mathbf{R} \mid \mathbf{t}]$; maps WRF homogeneous point directly to pixel. |
| **Homogeneous coords** | A point $(x,y)$ represented as $(x,y,1)^\top$ up to scale; enables matrix-multiply for perspective. |
| **Perspective projection** | The geometric operation mapping 3D CRF point to image plane via similar triangles. |
| **Inverted image** | Physical consequence of pinhole: image is upside-down on the back screen. |

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Simple linear model (in homogeneous coordinates); analytically tractable.
- Captures the essential geometry of real cameras to engineering accuracy.
- Enables calibration, rectification, triangulation, and all of stereo vision.
- Perspective projection expressed as a single matrix multiply: $s\tilde{\mathbf{p}} = \mathbf{P}\tilde{\mathbf{M}}$.

**Shortcomings / limitations:**
- **Not invertible from one camera**: depth $Z$ is lost in projection. A 2D pixel corresponds to an entire ray, not a unique 3D point. Motivates stereo.
- **No lens distortion**: real lenses introduce radial/tangential distortion not captured by the ideal pinhole model. See [[lens-distortion-and-removal]].
- **No skew in the basic model**: real sensors can have non-orthogonal pixel axes (skew $\alpha$). The basic $\mathbf{K}$ assumes $\alpha = 0$.
- **Assumes pinhole (infinite depth of field)**: real cameras have finite aperture and blur.
- **Single focal length**: square-pixel assumption ($f_x = f_y$) is needed for canonical stereo but may not hold for arbitrary cameras.

### Likely exam questions

**Q:** State the similar-triangles relation for pinhole projection and derive $x_p = f\,x/z$.

**A:** In the CRF, a point at $(x, y, z)$ is projected onto the image plane at focal distance $f$. By similar triangles (the triangle formed by the optical centre, the image point, and the principal axis, is similar to the triangle formed by the scene point):
$$\frac{x_p}{f} = \frac{x}{z} \implies x_p = f\frac{x}{z}, \qquad y_p = f\frac{y}{z}$$
The image plane is conventionally drawn in *front* of the focal point (virtual image, upright) so the image is not inverted in the diagram.

---

**Q:** Write out the full projection chain $\mathbf{P} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]$ as a product of three matrices applied to a WRF homogeneous point $\tilde{\mathbf{M}} = (X, Y, Z, 1)^\top$.

**A:**
$$\begin{pmatrix}su\\sv\\s\end{pmatrix} = \underbrace{\begin{pmatrix}\tfrac{1}{d_x}&0&c_x\\0&\tfrac{1}{d_y}&c_y\\0&0&1\end{pmatrix}}_{\mathbf{K}} \underbrace{\begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}\begin{pmatrix}r_{11}&r_{12}&r_{13}&t_x\\r_{21}&r_{22}&r_{23}&t_y\\r_{31}&r_{32}&r_{33}&t_z\\0&0&0&1\end{pmatrix}}_{[\mathbf{R}\mid\mathbf{t}]}\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$
Dividing by $s$ gives the actual pixel $(u, v)$.

---

**Q:** What are the intrinsic parameters $f_x$ and $f_y$, and what are their units?

**A:** $f_x = f / d_x$ and $f_y = f / d_y$ where $f$ is focal length in mm and $d_x, d_y$ are pixel sizes in mm/pixel. Units of $f_x, f_y$ are **pixels** (dimensionless in the sense of "pixels per mm" × mm = px). They encode focal length in the pixel coordinate system.

---

**Q:** Why can you not recover the 3D position of a scene point from a single 2D image? What extra information is required?

**A:** The projection $x_p = f\,x/z$, $y_p = f\,y/z$ is many-to-one: any point on the ray through $C$ and $(x_p, y_p)$ maps to the same pixel. Depth $Z$ is irreversibly lost. To recover 3D position you need either: (a) a **second camera** (stereo triangulation — see [[epipolar-geometry]], [[depth-from-disparity]]), or (b) known geometry / additional constraints (e.g., known object size, structured light, depth sensor).

---

**Q:** What is the difference between Case 1 and Case 2 image-frame conventions, and how does it affect the intrinsic matrix?

**A:** Case 1: origin top-left, $v$ axis pointing **down**. The mm-to-pixel transform has $-1/d_y$ (negated because $y_p$ increases upward in CRF but $v$ increases downward). Case 2: origin bottom-left, $v$ axis pointing **up**. The transform has $+1/d_y$. The sign of the $f_y = f/d_y$ entry in $\mathbf{K}$ flips between conventions.

---

**Q:** How many unknowns does the generic $3\times4$ projection matrix $\mathbf{P}$ have, and how many point correspondences are required to solve for them?

**A:** Normalising the last entry to 1 leaves **11 unknowns**. Each 3D–2D correspondence provides 2 equations (for $u$ and $v$). Minimum correspondences needed: $\lceil 11/2 \rceil = 6$ (giving 12 equations for 11 unknowns — overdetermined, solved via least squares). See [[camera-calibration-dlt]].

---

## A4. Directions of reasoning

### Forward: 3D point → pixels

**Given:** a 3D point $\tilde{\mathbf{M}} = (X, Y, Z, 1)^\top$ in WRF; camera matrices $\mathbf{R}$, $\mathbf{t}$, $\mathbf{K}$.

**Asked:** pixel coordinates $(u, v)$.

**Key steps:**
1. Transform to CRF: $(x, y, z)^\top = \mathbf{R}(X, Y, Z)^\top + \mathbf{t}$.
2. Project to image plane (mm): $x_p = f\,x/z$, $y_p = f\,y/z$.
3. Convert to pixels: $u = x_p/d_x + c_x$, $v = \pm y_p/d_y + c_y$ (sign depends on convention).
4. **Or in one step:** $s(u, v, 1)^\top = \mathbf{P}\,(X, Y, Z, 1)^\top$ with $s = z$; divide out $s$.

> [!example]
> **Worked example (from slides):** Given $f = 8$ mm, pixel size $d_x = d_y = 0.004$ mm/pixel, image $1000 \times 1000$ px, principal point $(c_x, c_y) = (500, 500)$ px, and a CRF point $(x, y, z)$.
>
> $f_x = f_y = 8/0.004 = 2000$ px.
>
> For a point at CRF $(x, y, z)$: $u = f_x \cdot x/z + c_x = 2000\,x/z + 500$, $v = f_y \cdot y/z + c_y$.

### Reverse: pixels → 3D (and why it fails)

**Given:** pixel $(u, v)$; calibrated $\mathbf{K}$, $\mathbf{R}$, $\mathbf{t}$.

**Asked:** 3D point $(X, Y, Z)$.

**Key inference:** You can compute a **ray** from the optical centre through the pixel — the direction is determined. But $Z$ (depth along the ray) is a free parameter. The map $\mathbf{P}: \mathbb{R}^4 \to \mathbb{R}^3$ has a 1D null space (all points on the same ray project to the same pixel). $\mathbf{P}$ is a $3\times4$ matrix; its pseudo-inverse gives a point on the ray but not the unique 3D location.

**Extra info needed to resolve $Z$:**
- A second camera image + epipolar geometry → triangulation (stereo). See [[epipolar-geometry]], [[backward-projection-and-ray-intersection]].
- Known depth from a range sensor or structured light.
- Geometric constraints (known plane, known object dimensions).

> [!warning]
> Back-projection gives a ray, not a point. The phrase "inverting the pinhole model" means recovering $Z$, which requires stereo or another depth cue.

---

## A5. Standard implementation

### a. Setup

**Goal:** project a 3D WRF point $\tilde{\mathbf{M}} = (X, Y, Z, 1)^\top$ to pixel $\tilde{\mathbf{p}} = (u, v, 1)^\top$ (homogeneous).

**Parameters:**
- Focal length $f$ (mm).
- Pixel sizes $d_x, d_y$ (mm/pixel).
- Principal point $(c_x, c_y)$ (pixels); typically $(W/2,\, H/2)$ for Case 1.
- Rotation $\mathbf{R}$ (3×3, orthonormal), translation $\mathbf{t}$ (3×1 in mm).
- Image-frame convention (Case 1 or Case 2).

**Notation:** scalars italic ($f$, $s$, $Z$); vectors bold lowercase ($\mathbf{t}$, $\mathbf{x}$); matrices bold uppercase ($\mathbf{R}$, $\mathbf{K}$, $\mathbf{P}$); homogeneous coords with tilde $\tilde{\mathbf{p}}$.

**Input:** $\tilde{\mathbf{M}} \in \mathbb{R}^4$.  **Output:** pixel $(u, v)$.

### b. Steps

**Step 1 — WRF to CRF (extrinsic transform).**

$$\begin{pmatrix}x\\y\\z\\1\end{pmatrix} = \begin{pmatrix}\mathbf{R}_{3\times3} & \mathbf{t}_{3\times1}\\\mathbf{0}_{1\times3} & 1\end{pmatrix}\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$

Result: $(x, y, z)$ in CRF (mm).

**Step 2 — CRF to image plane, mm (perspective projection).**

$$\begin{pmatrix}sx_p\\sy_p\\s\end{pmatrix} = \begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}\begin{pmatrix}x\\y\\z\\1\end{pmatrix}, \quad s = z \neq 0$$

Divide by $s$: $x_p = f\,x/z$, $y_p = f\,y/z$.

**Step 3 — Image plane (mm) to pixels (intrinsic transform).**

*Case 2 (bottom-left origin, $y$ up):*
$$\begin{pmatrix}u\\v\\1\end{pmatrix} = \underbrace{\begin{pmatrix}1/d_x & 0 & c_x \\ 0 & 1/d_y & c_y \\ 0 & 0 & 1\end{pmatrix}}_{\mathbf{K}}\begin{pmatrix}x_p\\y_p\\1\end{pmatrix}$$

*Case 1 (top-left origin, $y$ down):* replace $1/d_y$ with $-1/d_y$ (see [[#A6. Variations]]).

**Step 4 — Combine into the full projection matrix $\mathbf{P}$.**

$$\mathbf{P} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}], \quad \mathbf{P} \in \mathbb{R}^{3\times4}$$

$$s\tilde{\mathbf{p}} = \mathbf{P}\,\tilde{\mathbf{M}}, \quad \tilde{\mathbf{p}} = (u,v,1)^\top, \quad \tilde{\mathbf{M}} = (X,Y,Z,1)^\top, \quad s = z_{\text{CRF}}$$

> [!note]
> The slides express the intrinsic matrix as:
> $$\mathbf{K} = \begin{pmatrix}f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{pmatrix}, \quad f_x = \frac{f}{d_x},\ f_y = \frac{f}{d_y}$$
> The perspective $3\times4$ matrix (Step 2) and $\mathbf{K}$ (Step 3) multiply to give $\mathbf{K}[\mathbf{R}\mid\mathbf{t}]$ as the combined $3\times4$ matrix $\mathbf{P}$.

---

## A6. Variations

### Case 1 vs Case 2 image-frame conventions

Both conventions are used in computer vision; confusion between them causes sign errors in $v$.

| | Case 1 | Case 2 |
|---|---|---|
| **Origin** | Top-left corner | Bottom-left corner |
| **$v$-axis direction** | Downward ($v$ increases downward) | Upward ($v$ increases upward) |
| **Sign of $f_y$ in $\mathbf{K}$** | $-f/d_y$ (negative) | $+f/d_y$ (positive) |
| **Principal point** | $c_x = W/2$, $c_y = H/2$ | $c_x$, $c_y$ at optical axis intersection |

**Case 1 full mm-to-pixel transform:**
$$\begin{pmatrix}u\\v\\1\end{pmatrix} = \begin{pmatrix}1/d_x & 0 & c_x \\ 0 & -1/d_y & c_y \\ 0 & 0 & 1\end{pmatrix}\begin{pmatrix}x_p\\y_p\\1\end{pmatrix}, \quad c_x = W/2,\ c_y = H/2$$

**Case 2 full mm-to-pixel transform:**
$$\begin{pmatrix}u\\v\\1\end{pmatrix} = \begin{pmatrix}1/d_x & 0 & c_x \\ 0 & +1/d_y & c_y \\ 0 & 0 & 1\end{pmatrix}\begin{pmatrix}x_p\\y_p\\1\end{pmatrix}$$

> [!note]
> The sign flip arises because the CRF $y$-axis points upward (following a right-hand coordinate system), whereas Case 1 pixel coordinates have $v$ increasing downward. Case 2 aligns with the CRF $y$-direction, so no sign flip is needed.

**When to use which:** most image processing libraries (OpenCV, PIL) use Case 1 (top-left, $y$ down). Photogrammetry and some calibration tools use Case 2. Always check the convention before applying calibration parameters. See [[camera-calibration-dlt]] and [[tsai-calibration]].

### Intrinsic matrix with skew

In the basic model, pixels are assumed to have orthogonal axes (no skew). A more general $\mathbf{K}$ includes a skew parameter $\alpha$:

$$\mathbf{K} = \begin{pmatrix}f_x & \alpha & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{pmatrix}$$

where $\alpha$ encodes the angle between the $u$- and $v$-axes of the sensor if they are not exactly perpendicular. For modern cameras, $\alpha \approx 0$. Tsai calibration (see [[tsai-calibration]]) can estimate $\alpha$ if needed.

> [!warning]
> The course slides present only the zero-skew $\mathbf{K}$. The skew form is included here for completeness and because some calibration tools report it, but exam questions will likely assume $\alpha = 0$ unless stated otherwise.

---

**Related topics:** [[homogeneous-coordinates-and-transformations]] — basis for the matrix-multiply formulation. [[camera-calibration-dlt]] — how to solve for $\mathbf{P}$ from point correspondences. [[tsai-calibration]] — practical calibration method outputting $\mathbf{K}$, $[\mathbf{R}\mid\mathbf{t}]$, and distortion coefficients. [[epipolar-geometry]] — two-camera geometry that enables depth recovery. [[depth-from-disparity]] — using stereo disparity to compute $Z$. [[backward-projection-and-ray-intersection]] — formal treatment of the back-projection ray. [[lens-distortion-and-removal]] — correction applied before the pinhole model is used.
