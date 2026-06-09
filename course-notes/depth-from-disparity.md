# Depth from Disparity

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Recovers the metric depth $Z$ (distance along the optical axis) of a scene point from a **rectified stereo image pair**, given the cameras' intrinsic parameters and baseline. It is the core computational step of the binocular stereo pipeline — the bridge between the disparity map produced by [[stereo-block-matching]] and the 3D point cloud. It is used whenever 3D structure must be inferred passively (no active depth sensor) from two calibrated cameras.

---

## A2. High-level overview

**Steps (no math):**
1. Acquire the pixel coordinates of a matched point in the left and right rectified images.
2. Compute the horizontal pixel offset (disparity) between the two image coordinates.
3. Convert disparity from pixels to metric units (or keep focal length dimensionless — see A6).
4. Apply the depth formula to recover $Z$.
5. Back-project the 2D image point into 3D to recover $X_w$ and $Y_w$.

**Key definitions:**

| Term | Definition |
|---|---|
| $d$ | **Disparity** — horizontal pixel offset $d = x_L - x_R$ between corresponding points in the left and right rectified images. A discrete integer. |
| $b$ | **Baseline** — physical separation between the two optical centres (mm). Equals $b_1 + b_2$ where $b_1, b_2$ are distances from each camera to the vertical through the scene point. |
| $f$ | **Focal length** — either in mm ($f_{mm}$) or dimensionless pixels ($f_{px} = f_{mm}/s$); must be consistent with the units of $d$. |
| $s$ | **Pixel size** — physical width of one pixel (mm/pixel). |
| $Z$ | **Depth** — distance from the camera optical centre to the scene point along the optical axis (mm). $Z \propto 1/d$. |
| $c_x, c_y$ | **Principal point** — pixel coordinates of the optical axis intersection with the image plane. |
| $X_w, Y_w, Z_w$ | **3D world point** (mm) recovered by back-projection. |
| Disparity map | Image where each pixel stores the disparity $d(x,y)$ of the matched point. |
| Canonical geometry | Rectified stereo configuration: optical axes parallel, baseline along $X$-axis, same intrinsics, $f_x = f_y = f$. Ensures epipolar lines are horizontal and correspondence reduces to a horizontal search. |

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Simple closed-form formula once disparity is known.
- Passive (no active illumination needed).
- Dense depth map possible if a dense disparity map is available.
- Works with any two calibrated, rectified cameras.

**Shortcomings / limitations:**
- Depth accuracy degrades quadratically with $Z$: $\Delta Z \approx Z^2 / (fb)$ for distant objects.
- Disparity is quantised to integer pixels, so depth is quantised — $\Delta Z = Z^2/(fb+Z)$.
- Sensitive to matching errors; a 1-pixel disparity error at 10 m gives ~40 cm depth error (see worked examples).
- Requires prior rectification ([[stereo-rectification]]) and calibration.
- Fails wherever stereo matching fails: textureless regions, occlusions, transparent/specular surfaces, repetitive patterns.
- Deep learning cannot remove the geometric constraint $Z = fb/d$ — it can only improve matching quality.

### Likely exam questions

**Q:** Write the depth formula and explain each symbol.
**A:** $Z = fb/d$. $f$ = focal length (pixels, dimensionless), $b$ = baseline (mm), $d$ = disparity (pixels). Depth is inversely proportional to disparity.

---

**Q:** A point has $x_L = 420$ px, $x_R = 380$ px, $f = 8$ mm, $b = 120$ mm, $s = 0.004$ mm/px. Find $Z$.
**A:** $d_{px} = 420 - 380 = 40$ px; $d_{mm} = 40 \times 0.004 = 0.16$ mm; $Z = (8 \times 120) / 0.16 = \mathbf{6000}$ mm.

---

**Q:** A point has $x_L = 570$ px, $x_R = 400$ px, $f_{mm} = 2$ mm, $b = 65$ mm, $s = 0.002$ mm/px. Find $Z$.
**A:** $d_{px} = 170$ px; $d_{mm} = 170 \times 0.002 = 0.34$ mm; $Z = (2 \times 65)/0.34 = \mathbf{382.35}$ mm.

---

**Q:** Why is depth inversely proportional to disparity? What does this mean for close vs. far objects?
**A:** From $Z = fb/d$: as $d$ increases, $Z$ decreases. Close objects project to widely separated positions in the two images (large $d$); far objects project to nearly the same position (small $d$).

---

**Q:** Given $Z = 6000$ mm, $f_{px} = 2000$, $b = 120$ mm, what is the expected disparity?
**A:** $d = f_{px} \cdot b / Z = 2000 \times 120 / 6000 = \mathbf{40}$ px.

---

**Q:** A pixel at $(x, y) = (520, 310)$ with principal point $(c_x, c_y) = (512, 256)$ and $Z = 3000$ mm, $f_{px} = 1000$ px. Find $(X_w, Y_w, Z_w)$.
**A:** $X_w = (3000/1000)(520 - 512) = 3 \times 8 = \mathbf{24}$ mm; $Y_w = (3000/1000)(310 - 256) = 3 \times 54 = \mathbf{162}$ mm; $Z_w = \mathbf{3000}$ mm.

---

> [!note] Exam hint (from hints.txt)
> "How to calculate disparity manually" is flagged as a key skill. The examiners expect the explicit 3-step procedure: (1) subtract pixel coordinates, (2) convert pixels to mm, (3) apply $Z = fb/d$ — not a shortcut.

---

## A4. Directions of reasoning

### Forward (disparity → depth)

**Given:** $x_L$, $x_R$ (pixel coords), $f_{mm}$, $b$, $s$.  
**Asked:** depth $Z$ of the matched 3D point.  
**Key inference:** compute $d_{px} = x_L - x_R$; convert $d_{mm} = d_{px} \cdot s$; apply $Z = f_{mm} \cdot b / d_{mm}$.

### Reverse (depth → expected disparity)

**Given:** known depth $Z$, $f$ (pixels, dimensionless), $b$ (mm).  
**Asked:** what disparity should a point at depth $Z$ produce?  
**Key manipulation:** rearrange $Z = fb/d$ to $d = fb/Z$.  
This is used to verify calibration, set $d_{\max}$ for the closest expected feature, or check that a returned disparity value is geometrically plausible.

### 3D back-projection (depth → 3D point)

**Given:** image pixel $(x, y)$, depth $Z$, principal point $(c_x, c_y)$, $f_{px}$.  
**Asked:** 3D world point $(X_w, Y_w, Z_w)$.  
**Key manipulation:** undo the pinhole projection — multiply the centred, normalised pixel offset by $Z$.

---

## A5. Standard implementation

### a. Setup

- **Input:** pair of rectified stereo images; matched pixel coordinates $(x_L, y)$ and $(x_R, y)$ (same row after rectification); camera intrinsics: $f_{mm}$, $s$, principal point $(c_x, c_y)$; baseline $b$ (mm).
- **Output:** depth $Z$ (mm); optionally 3D point $(X_w, Y_w, Z_w)$.
- **Assumption:** canonical (rectified) geometry — corresponding points are on the same image row, disparity is purely horizontal.
- **Notation:** $d_{px}$ = disparity in pixels (integer); $d_{mm}$ = disparity in mm; $f_{px} = f_{mm}/s$ = focal length in pixels (dimensionless).

### b. Steps

**Step 1 — Compute disparity in pixels:**
$$d_{px} = x_L - x_R$$

> [!note]
> Convention in some block-matching code is $d = x_2 - x_1$ (right minus left, giving a positive shift leftward). The sign convention must be consistent; the slides use $d = x_L - x_R > 0$ for standard forward-facing stereo.

**Step 2 — Convert disparity to millimetres:**
$$d_{mm} = d_{px} \times s$$

**Step 3 — Compute depth (consistent physical units):**
$$Z = \frac{f_{mm} \cdot b}{d_{mm}}$$

> [!example] W7L worked example
> $f_{mm} = 8$ mm, $b = 120$ mm, $s = 0.004$ mm/px, $x_L = 420$ px, $x_R = 380$ px.
>
> 1. $d_{px} = 420 - 380 = 40$ px
> 2. $d_{mm} = 40 \times 0.004 = 0.16$ mm
> 3. $Z = (8 \times 120) / 0.16 = \mathbf{6000}$ mm

> [!example] W8T Example-1
> $f_{mm} = 2$ mm, $b = 65$ mm, $s = 0.002$ mm/px, $x_L = 570$ px, $x_R = 400$ px.
>
> 1. $d_{px} = 570 - 400 = 170$ px
> 2. $d_{mm} = 170 \times 0.002 = 0.34$ mm
> 3. $Z = (2 \times 65) / 0.34 = \mathbf{382.35}$ mm

**Step 4 (optional) — Recover the 3D point by back-projection:**

Using $f_{px} = f_{mm}/s$:

$$X_w = \frac{Z}{f_{px}}\,(x - c_x)$$

$$Y_w = \frac{Z}{f_{px}}\,(y - c_y)$$

$$Z_w = Z$$

This inverts the [[pinhole-camera-model]] projection, using the recovered $Z$ to disambiguate the depth-lost image projection. See [[backward-projection-and-ray-intersection]].

**Depth range limits** (for a sensor with $n$ pixels per row):

$$d_{\max} = (n - 1) \cdot s \quad \Rightarrow \quad Z_{\min} = \frac{f_{mm} \cdot b}{d_{\max}}$$

$$d_{\min} = 1 \cdot s \quad \Rightarrow \quad Z_{\max} = \frac{f_{mm} \cdot b}{d_{\min}}$$

> [!example] W7L range example (same parameters)
> $n = 1000$, $d_{\max} = 999 \times 0.004 = 3.996$ mm $\Rightarrow Z_{\min} = 960/3.996 \approx 240$ mm;
> $d_{\min} = 0.004$ mm $\Rightarrow Z_{\max} = 960/0.004 = 240{,}000$ mm.

---

## A6. Variations

### Dimensionless focal length ($f_{px}$) convention

**Change:** divide focal length by pixel size once to get a dimensionless quantity $f_{px} = f_{mm}/s$, then use disparity directly in pixels without Step 2.

**Steps with this convention:**

$$d_{px} = x_L - x_R \qquad Z = \frac{f_{px} \cdot b}{d_{px}}$$

Both forms are equivalent: $f_{mm} \cdot b / d_{mm} = (f_{mm}/s) \cdot b / (d_{px}) = f_{px} \cdot b / d_{px}$.

**When used:** W7L_pt2 and W8L_pt2 consistently use $f_{px}$; W8T uses $f_{px}$ for the $\Delta Z$ derivation. The W7L slides show the 3-step procedure with explicit unit conversion to reinforce unit handling.

> [!note]
> After rectification, the reported focal length "2248" in the slides is already $f_{mm}/s$ (dimensionless pixels), not a value in mm. Check which convention the question uses before plugging in numbers.

---

### Derivation of $Z = fb/d$ from triangulation (similar triangles)

Using the canonical stereo geometry with the scene point $P$ at depth $Z$, baseline $b = b_1 + b_2$, and projection angles $\theta_1$, $\theta_2$:

$$\frac{X_L}{f} = \frac{b_1}{Z}, \qquad \frac{-X_R}{f} = \frac{b_2}{Z}$$

Adding:

$$\frac{X_L - X_R}{f} = \frac{b_1 + b_2}{Z} = \frac{b}{Z}$$

Substituting $d = X_L - X_R$:

$$\boxed{Z = \frac{fb}{d}}$$

This derivation holds in any consistent unit system (both $f$ and $d$ in mm, or both dimensionless pixels).

---

### Depth resolution $\Delta Z$

The quantisation of $d$ to integer pixels limits depth precision. The depth resolution is the change in $Z$ for a 1-pixel change in $d$:

$$\Delta Z = \frac{fb}{d} - \frac{fb}{d+1} = \frac{fb}{d(d+1)}$$

Substituting $d = fb/Z$ (dimensionless convention):

$$\boxed{\Delta Z = \frac{Z^2}{fb + Z}}$$

For large $Z$: $\Delta Z \approx Z^2 / (fb)$ — depth error grows quadratically. See [[stereo-depth-accuracy]] for full treatment and improvement strategies.

---

### Epipolar and rectification prerequisites

$Z = fb/d$ assumes **canonical (rectified) epipolar geometry**: correspondences share the same image row, disparity is purely horizontal, both cameras have equal focal length. Without [[stereo-rectification]] the formula does not apply directly. The fundamental geometry is described in [[epipolar-geometry]].

---

### Related topics

- [[epipolar-geometry]] — why corresponding points lie on the same row after rectification.
- [[stereo-rectification]] — the warping step that creates the canonical geometry required here.
- [[stereo-block-matching]] — produces the disparity map $d(x,y)$ consumed by this formula.
- [[stereo-depth-accuracy]] — $\Delta Z$ derivation, $Z_{\min}$/$Z_{\max}$, baseline/pixel-size trade-offs.
- [[pinhole-camera-model]] — the projection model that back-projection (Step 4) inverts.
