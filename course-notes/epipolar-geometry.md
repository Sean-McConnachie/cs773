# Epipolar Geometry

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
- [[#A6. Variations]]

---

## A1. Purpose

Epipolar geometry describes the projective relationship between two views of the same 3D scene. It exploits the constraint that a 3D point $S$, and both camera optical centres $O_l$ and $O_r$, are coplanar — the **epipolar plane** — so that a point observed in one image must correspond to a point lying on a known line (the **epipolar line**) in the other image. This reduces the correspondence search from a 2D problem to a 1D line search, which is the foundational constraint enabling efficient stereo matching. It arises at the **rectification** and **stereo matching** stages of the binocular stereo pipeline.

---

## A2. High-level overview

**Steps (conceptually):**
1. Two cameras with optical centres $O_l$, $O_r$ observe the same 3D point $S$.
2. The plane containing $S$, $O_l$, $O_r$ is the **epipolar plane**.
3. The intersection of the epipolar plane with each image plane gives the **epipolar lines** $l_l$ and $l_r$.
4. All epipolar lines in one image pass through the **epipole** of that image.
5. Apply the epipolar constraint: the match for a pixel in the left image lies on the corresponding epipolar line in the right image.

**Key definitions:**

| Term | Definition |
|---|---|
| $O_l$, $O_r$ | Optical centres (centres of projection) of the left and right cameras |
| $S = (X_{3d}, Y_{3d}, Z_{3d})$ | 3D scene point being observed |
| Epipolar plane | The plane containing $S$, $O_l$, and $O_r$ |
| $e_l$, $e_r$ | Epipoles: projection of the other camera's optical centre onto each image plane; equivalently, the point where the baseline $O_l O_r$ meets each image plane |
| $l_l$, $l_r$ | Epipolar lines: intersection of the epipolar plane with each image plane |
| Baseline $b$ | The distance between $O_l$ and $O_r$ |
| Disparity $d$ | $d = x_l - x_r$; the horizontal shift between corresponding pixel columns in canonical geometry |
| Epipolar constraint | Given pixel $s_l$ in the left image, its correspondence $s_r$ must lie on $l_r$ in the right image |

> [!note]
> The slides use $S_l e_l$ and $S_r e_r$ to denote the epipolar lines in left and right images respectively. This note links notation back to the slide diagrams (W7L_pt1, p. 35; W8L_pt1, p. 6).

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Reduces 2D correspondence search to a 1D search along the epipolar line — large computational saving.
- Enables triangulation and recovery of depth $Z$ from disparity.
- Provides a geometric guarantee independent of scene content.
- In canonical (rectified) geometry, the constraint simplifies further: matches are on the **same row**, so search is purely horizontal.

**Shortcomings & limitations:**
- Requires accurate camera calibration (intrinsics and extrinsics) to compute epipolar lines.
- Physical camera mounting to achieve canonical geometry is impractical — rectification (computational warping) is always needed in practice; see [[stereo-rectification]].
- Epipolar constraint does not resolve the correspondence problem by itself — ambiguity remains on textureless surfaces, occlusions, and repeated patterns.
- General epipolar geometry (non-canonical) requires the fundamental/essential matrix, which adds complexity.
- If cameras are not rigidly mounted (moving rig), calibration becomes dynamic.

### Likely exam questions

**Q:** What is the epipole $e_r$, and where does it come from?
**A:** $e_r$ is the projection of the left camera's optical centre $O_l$ onto the right image plane. It is the point where the baseline $O_l O_r$ intersects the right image plane. Every epipolar line in the right image passes through $e_r$.

---

**Q:** Why do all epipolar lines pass through the epipole?
**A:** All epipolar planes share the baseline $O_l O_r$ as a common edge. Each epipolar plane intersects an image plane in an epipolar line. Since all these planes contain the same line $O_l O_r$, all epipolar lines in a given image must meet at the single point where $O_l O_r$ pierces that image plane — the epipole.

---

**Q:** What is the epipolar constraint, and what does it gain computationally?
**A:** Given a pixel $s_l = (x_l, y_l)$ in the left image, the epipolar constraint states that its correspondence $s_r$ in the right image must lie on the epipolar line $l_r$. This reduces the search from all pixels in the right image (2D) to a single line (1D).

---

**Q:** State the four attributes of canonical epipolar geometry.
**A:** (1) The baseline $b$ between optical centres is aligned with the $X$-axis. (2) The optical axes of both cameras are parallel and point in the $Z$ direction. (3) Both cameras share the same intrinsic parameters. (4) Focal length satisfies $f_x = f_y = f$ (square pixels).

---

**Q:** In canonical geometry, a point in the left image is at pixel $(420, 300)$. Where must its match lie in the right image?
**A:** On row $y = 300$ (same row). The match is at $(x_r, 300)$ for some $x_r < 420$ (since objects project to the left in the right camera when the right camera is to the right of the left), so only a horizontal 1D search is needed.

---

> [!example]
> **Q:** Given $f = 8$ mm, baseline $b = 120$ mm, pixel size $s = 0.004$ mm/px, and disparity $d_{px} = 40$ px, compute depth $Z$.
>
> **A:**
> $d_{mm} = 40 \times 0.004 = 0.16$ mm
>
> $Z = \dfrac{fb}{d} = \dfrac{8 \times 120}{0.16} = 6000$ mm
>
> (From the worked example in W7L_pt1, pp. 44–45.)

---

## A4. Directions of reasoning

### Forward / standard (pixel in left → constraint on right)

**Given:** A pixel $s_l = (x_l, y_l)$ in the left image; camera models $K_l$, $M_l$, $K_r$, $M_r$ (or equivalently, calibrated canonical geometry).

**Asked:** Where must the correspondence $s_r$ lie?

**Inference:** The ray through $O_l$ and $s_l$ intersects a family of 3D points at different depths. Each projects to a different point in the right image, but all lie on the same epipolar line $l_r$. Therefore $s_r \in l_r$ — search along $l_r$ only.

In **canonical geometry** specifically: $l_r$ is the horizontal line $y = y_l$, so $s_r = (x_r, y_l)$ for some $x_r$, and only a horizontal scan over disparities $d = x_l - x_r \geq 0$ is needed.

### Reverse / inferential (given correspondence → recover depth)

**Given:** A correspondence pair $s_l = (x_l, y_l)$, $s_r = (x_r, y_r)$ in canonical geometry; $f$, $b$, pixel size $s$.

**Asked:** What is the 3D depth $Z$?

**Inference:**
1. Compute disparity in pixels: $d_{px} = x_l - x_r$.
2. Convert to metric: $d_{mm} = d_{px} \times s$.
3. Apply $Z = \dfrac{fb}{d_{mm}}$.

The epipolar constraint (same row: $y_l = y_r$ in canonical geometry) must hold — if rows differ, the images are not rectified or the correspondence is wrong.

> [!note]
> If $d_{px} = 0$ (zero disparity), the point is at infinity. If $d_{px}$ is large, the point is close. This inverse relationship ($Z \propto 1/d$) is the core of [[depth-from-disparity]].

---

## A5. Standard implementation

### a. Setup

**Input:** Two calibrated cameras; image pair (left $I_l$, right $I_r$); a query pixel $s_l = (x_l, y_l) \in I_l$.

**Output:** The epipolar line $l_r$ in $I_r$ on which the correspondence must lie.

**Assumptions:** Rigid stereo rig; static scene; cameras calibrated (intrinsics $K_l, K_r$; extrinsics $M_l = [R_l \mid t_l]$, $M_r = [R_r \mid t_r]$).

**Notation (this file):** scalars italic ($f$, $d$, $Z$, $b$); image pixels $(x, y)$ or $(u, v)$; 3D points in WRF $(X, Y, Z)$; optical centres $O_l, O_r$; epipoles $e_l, e_r$; epipolar lines $l_l, l_r$.

### b. Steps

1. **Identify the epipolar plane.** The plane is defined by the 3D point $S$ and the two optical centres $O_l$, $O_r$. Its orientation depends on $S$, so it varies per pixel.

2. **Find the epipole.** $e_r$ = projection of $O_l$ into the right image; $e_l$ = projection of $O_r$ into the left image:
   $$e_r = \mathbf{P}_r \, \tilde{O}_l, \quad e_l = \mathbf{P}_l \, \tilde{O}_r$$
   where $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$ is the $3 \times 4$ projection matrix.

3. **Construct the epipolar line in $I_r$.** In general geometry, the line $l_r$ is determined by the fundamental matrix $\mathbf{F}$:
   $$l_r = \mathbf{F}\,\tilde{\mathbf{s}}_l$$
   where $\tilde{\mathbf{s}}_l = (x_l, y_l, 1)^\top$ is the homogeneous image point.

4. **Apply the epipolar constraint.** The correspondence $s_r = (x_r, y_r)$ satisfies:
   $$\tilde{\mathbf{s}}_r^\top \, l_r = 0$$
   i.e. $s_r$ lies on the line $l_r$ in $I_r$.

5. **Search along $l_r$.** Evaluate a patch similarity measure (SSD, SAD, NCC — see [[stereo-block-matching]]) at each candidate position along $l_r$ to find the best match.

> [!note]
> Steps 3–4 use the fundamental matrix, which is beyond the core syllabus scope. For exam purposes, the canonical-geometry case (A6 below) is the primary treatment; the fundamental matrix is mentioned here for completeness.

---

## A6. Variations

### General epipolar geometry (non-canonical)

**Setup change:** Cameras have different intrinsics $\mathbf{K}_1 \neq \mathbf{K}_2$, different orientations $\mathbf{R}_1 \neq \mathbf{R}_2$, arbitrary placement. The baseline is not aligned with any axis.

**Properties:**
- Epipoles $e_l$, $e_r$ are finite points within (or outside) the image planes.
- Epipolar lines are not horizontal — they fan out from the epipole.
- Correspondences are on the epipolar line but not necessarily on the same row.
- Search direction varies per pixel.

**When used:** Any uncalibrated or non-aligned stereo rig before rectification.

**Transition to canonical:** [[stereo-rectification]] (Fusiello algorithm) computes homographies $\mathbf{H}_1$, $\mathbf{H}_2$ that warp both images to canonical geometry, making all epipolar lines horizontal and parallel.

---

### Canonical (rectified) epipolar geometry

**Setup — four attributes** (all four must hold):
1. Baseline $b$ between $O_l$ and $O_r$ is aligned with the $X$-axis.
2. Optical axes of both cameras are parallel and point in the $Z$ direction.
3. Both cameras share the same intrinsic matrix $\mathbf{K}$.
4. $f_x = f_y = f$ (square pixels).

**Properties** (derived from the attributes):
- Epipoles $e_l$, $e_r$ are **at infinity** — the baseline is parallel to the image planes and never intersects them.
- Epipolar lines are **horizontal and parallel** in both images.
- Corresponding points lie on the **same row**: $y_l = y_r$.
- Corresponding points differ **only in $x$-coordinate**: $x_l \neq x_r$ in general, $y_l = y_r$.
- Disparity $d = x_l - x_r$ encodes depth (see [[depth-from-disparity]]):
  $$Z = \frac{fb}{d}$$
- Disparity map: each pixel $(x, y)$ stores $d(x, y) = x_l - x_r$; close objects → high $d$; far objects → low $d$.

**When used:** After [[stereo-rectification]]; assumed by all standard stereo matching algorithms including [[stereo-block-matching]].

> [!note]
> Physical mounting of cameras in perfect canonical configuration is impractical (sensitivity to alignment errors, requirement for identical hardware). In practice, cameras are mounted approximately and then computationally rectified — see [[stereo-rectification]].

---

### Related topics

- [[pinhole-camera-model]] — the projection model underlying the geometry of each camera.
- [[stereo-rectification]] — how canonical geometry is achieved computationally (Fusiello algorithm).
- [[depth-from-disparity]] — derivation of $Z = fb/d$ and worked numeric examples.
- [[stereo-block-matching]] — search along epipolar lines using SSD/SAD/NCC cost functions.
