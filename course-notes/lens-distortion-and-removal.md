# Lens Distortion and Removal

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Key definitions]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Lens distortion is a systematic geometric error that displaces image pixels radially from the principal point — outward (barrel) or inward (pincushion). Because it is systematic, it can be modelled with one or two scalar coefficients and corrected algebraically before or during calibration. Distortion removal maps each distorted pixel $(x_d, y_d)$ to its undistorted position $(x_u, y_u)$, making the corrected image consistent with the ideal pinhole model assumed by $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$. It is a necessary pre-processing step whenever sub-pixel accuracy is required in calibration, stereo, or feature matching.

---

## A2. High-level overview

1. Identify distortion type (barrel / pincushion / mustache) from visual inspection or calibration residuals.
2. Determine the distortion coefficient $\kappa_1$ (and optionally $\kappa_2$) — either from Tsai calibration or from a design-time constraint on maximum allowable displacement $\Delta$.
3. Convert $\kappa_1$ to consistent units (metric mm$^{-2}$ or pixel px$^{-2}$) using the sensor pixel size.
4. For every distorted pixel $(x_d, y_d)$, apply the inverse-model correction formula to obtain $(x_u, y_u)$.
5. Use the corrected image for all subsequent processing (calibration, rectification, matching).

### Key definitions

| Term | Definition |
|---|---|
| Principal point $(c_x, c_y)$ | Image centre in pixels; origin of the centred coordinate system. |
| $r_d$ | Radial distance of a distorted point from the principal point (pixels or mm). |
| $r_u$ | Radial distance of the corrected (undistorted) point from the principal point. |
| $\kappa_1$ | First-order radial distortion coefficient; units px$^{-2}$ or mm$^{-2}$ depending on domain. |
| $\kappa_2$ | Second-order radial distortion coefficient; used for mustache/higher-order distortion. |
| $\kappa_1^p$ | $\kappa_1$ expressed in pixel units (px$^{-2}$). |
| $\kappa_1^m$ | $\kappa_1$ expressed in metric units (mm$^{-2}$). |
| pixel\_size | Physical size of one pixel: sensor dimension / image resolution (mm/px). |
| $\Delta$ | Maximum allowable displacement (in pixels or mm) between distorted and undistorted position. |
| Inverse model | Maps distorted $\to$ undistorted directly; avoids solving a nonlinear equation per pixel. |
| Barrel distortion | Radial outward bow; image magnified less at edges; $\kappa_1 > 0$ in the inverse-model convention. |
| Pincushion distortion | Radial inward pinch; image magnified more at edges; $\kappa_1 < 0$ in the inverse-model convention. |
| Mustache distortion | Distortion that reverses direction with radius; requires $\kappa_2 \neq 0$ with $\text{sign}(\kappa_1) \neq \text{sign}(\kappa_2)$. |

> [!warning]
> **Sign-convention caution.** The statement "$\kappa_1 > 0$ corrects barrel" holds in the **inverse-model** convention used in this course ($r_u = r_d(1 + \kappa_1 r_d^2)$). Some other sources (e.g., OpenCV's forward distortion model) use the opposite sign assignment. Always check which convention a given tool or paper uses before comparing $\kappa_1$ values.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Computationally cheap: a single multiply-add per pixel once $\kappa_1$ is known.
- The first-order model ($\kappa_1$ only) handles most consumer lenses adequately.
- The inverse model avoids per-pixel nonlinear root-finding.
- $\kappa_1$ can be derived analytically from a single design constraint ($\Delta$, $r_d$).

**Shortcomings & limitations**
- The inverse model is an approximation; for very large distortions it may not be exact.
- Only radial distortion is modelled; tangential (decentring) distortion is ignored.
- A single $\kappa_1$ cannot model distortion that reverses direction (mustache) — need $\kappa_2$.
- Pixel-unit vs. metric-unit confusion is a frequent error source; unit conversion is mandatory.
- The formula is centred on the principal point; an inaccurate $(c_x, c_y)$ propagates error everywhere.

### Likely exam questions

**Q1:** Write the first-order radial distortion correction formula (inverse model) for a point $(x_d, y_d)$ centred on the principal point $(c_x, c_y)$.

**A1:**
$$x_u = (x_d - c_x)\!\left(1 + \kappa_1^p\,r_d^2\right) + c_x, \qquad y_u = (y_d - c_y)\!\left(1 + \kappa_1^p\,r_d^2\right) + c_y$$
where $r_d^2 = (x_d - c_x)^2 + (y_d - c_y)^2$ and $\kappa_1^p$ is in px$^{-2}$.

---

**Q2:** A camera sensor is $5.0 \times 3.75$ mm at $5000 \times 3750$ px with $\kappa_1^m = -1.2 \times 10^{-3}$ mm$^{-2}$. Find the undistorted pixel for the distorted pixel $(0, 0)$ (origin at bottom-left).

**A2:** *(Distortion Example 1 — pincushion)*
- $c_x = 2500$, $c_y = 1875$; pixel\_size $= 5.0/5000 = 0.001$ mm/px.
- $\kappa_1^p = -1.2\times10^{-3} \times (0.001)^2 = -1.2\times10^{-9}$ px$^{-2}$.
- $r_d^2 = (0-2500)^2 + (0-1875)^2 = 9{,}765{,}625$.
- Factor $= 1 + \kappa_1^p r_d^2 = 1 + (-1.2\times10^{-9})(9{,}765{,}625) \approx 0.98828$.
- $x_u = (-2500)(0.98828) + 2500 \approx 29$, $y_u = (-1875)(0.98828) + 1875 \approx 22$.

> [!example]
> **Result:** $(x_u, y_u) \approx (29, 22)$ — the corner shifts inward (pincushion correction moves corners toward centre).

---

**Q3:** What is the formula for $\kappa_1$ given a maximum allowable pixel displacement $\Delta$ at radial distance $r_d$ from the principal point?

**A3:**
$$\kappa_1 = \frac{\Delta}{r_d^3}$$
This follows from $r_u - r_d = r_d \cdot \kappa_1 r_d^2 = \kappa_1 r_d^3 = \Delta$.

---

**Q4:** A camera has centre $(1500, 1125)$ and pixel\_size $= 0.002$ mm/px. A distorted pixel at $(2351, 1155)$ must not be displaced by more than $\Delta = 1$ px. Find $\kappa_1^m$.

**A4:** *(Distortion Example 2 — barrel)*
- $r_d = \sqrt{(2351-1500)^2 + (1155-1125)^2} = \sqrt{851^2 + 30^2} = 851.528$ px $= 851.528 \times 0.002 = 1.703$ mm.
- **Pixel route:** $\kappa_1^p = 1/(851.528)^3 = 1.619\times10^{-9}$ px$^{-2}$; then $\kappa_1^m = 1.619\times10^{-9}/(0.002)^2 = 4.04\times10^{-4}$ mm$^{-2}$.
- **Metric route:** $\Delta = 1 \times 0.002 = 0.002$ mm; $\kappa_1^m = 0.002/(1.703)^3 = 4.04\times10^{-4}$ mm$^{-2}$.

---

**Q5:** What is mustache distortion and when does it occur?

**A5:** Mustache distortion is a higher-order radial distortion where the direction of displacement reverses at some intermediate radius: barrel near the centre transitions to pincushion at the edges (or vice versa). It occurs in complex wide-angle and zoom lenses. The model is $r_u = r_d(1 + \kappa_1 r_d^2 + \kappa_2 r_d^4)$ with the condition $\text{sign}(\kappa_1) \neq \text{sign}(\kappa_2)$.

---

**Q6:** A $4000 \times 3000$ px camera has sensor $8.0 \times 6.0$ mm. Find the maximum tolerable $\kappa_1^m$ if the worst-case corner must not be displaced by more than 1 px.

**A6:** *(Distortion Example 3 — max $\kappa_1$)*
- Worst corner: $(4000, 3000)$; $c_x = 2000$, $c_y = 1500$.
- $r_d = \sqrt{2000^2 + 1500^2} = 2500$ px; pixel\_size $= 8.0/4000 = 0.002$ mm/px.
- $\kappa_1^p = 1/(2500)^3 = 6.4\times10^{-11}$ px$^{-2}$.
- $\kappa_1^m = 6.4\times10^{-11}/(0.002)^2 = 1.6\times10^{-5}$ mm$^{-2}$.

---

## A4. Directions of reasoning

### Forward / standard (given $\kappa_1$, correct a distorted image)

- **Given:** distorted pixel $(x_d, y_d)$, principal point $(c_x, c_y)$, $\kappa_1^p$.
- **Asked:** undistorted pixel $(x_u, y_u)$.
- **Key step:** compute $r_d^2$, apply $x_u = (x_d - c_x)(1 + \kappa_1^p r_d^2) + c_x$.

### Reverse / inferential (given allowable error, find $\kappa_1$)

- **Given:** maximum allowable displacement $\Delta$ at a known worst-case point $(x_d, y_d)$ (e.g., image corner), principal point, pixel\_size.
- **Asked:** maximum tolerable $\kappa_1^m$ (or $\kappa_1^p$).
- **Key inference:** From $\Delta = \kappa_1 r_d^3$, rearrange to $\kappa_1 = \Delta / r_d^3$. Compute $r_d$ at the worst corner, apply the formula in consistent units, convert $\kappa_1^p \to \kappa_1^m$ if needed. This is the A4 design formula used in Examples 2 and 3.

> [!note]
> The worst-case point is always the farthest corner from the principal point, since $r_d$ is largest there and hence distortion displacement $\kappa_1 r_d^3$ is largest.

---

## A5. Standard implementation

### A5a. Setup

| Item | Detail |
|---|---|
| **Input** | Distorted pixel image; principal point $(c_x, c_y)$ px; $\kappa_1^m$ in mm$^{-2}$ from calibration. |
| **Output** | Undistorted pixel image. |
| **Parameters** | Sensor size $cs_x \times cs_y$ mm; image resolution $w \times h$ px. |
| **Assumption** | Radial distortion only; principal point known; square (or known aspect) pixels. |

### A5b. Steps

1. **Compute pixel size.**
   $$\text{pixel\_size\_x} = \frac{cs_x}{w}, \quad \text{pixel\_size\_y} = \frac{cs_y}{h}, \quad \text{pixel\_size} = \frac{\text{pixel\_size\_x} + \text{pixel\_size\_y}}{2}$$

2. **Convert $\kappa_1$ to pixel units.**
   $$\kappa_1^p = \kappa_1^m \times (\text{pixel\_size})^2$$

3. **For each distorted pixel $(x_d, y_d)$, compute the squared radial distance.**
   $$r_d^2 = (x_d - c_x)^2 + (y_d - c_y)^2$$

4. **Apply the inverse first-order correction.**
   $$x_u = (x_d - c_x)\!\left(1 + \kappa_1^p\,r_d^2\right) + c_x$$
   $$y_u = (y_d - c_y)\!\left(1 + \kappa_1^p\,r_d^2\right) + c_y$$

5. **Write $(x_u, y_u)$ to the output image** (interpolate if the result is non-integer).

> [!note]
> **Equivalent radial form.** Steps 3–4 can be written compactly as $r_u = r_d(1 + \kappa_1^p r_d^2)$, where the centred coordinates act as the radial direction vector. The pixel-centred form in Step 4 is directly implementable.

> [!note]
> **Inverse vs. forward model.** The formula above is the *inverse model* (distorted $\to$ undistorted). The *forward model* would be $r_d = r_u(1 - \kappa_1 r_u^2 + \ldots)$ and requires iterative inversion. The inverse model is preferred in practice for efficiency.

---

## A6. Variations

### First-order model only ($\kappa_2 = 0$)

- Standard case; single coefficient $\kappa_1$.
- Sufficient for most prime lenses with moderate distortion.
- Formula: $r_u = r_d(1 + \kappa_1 r_d^2)$.

### Second-order (mustache) model ($\kappa_1$, $\kappa_2$)

- Required when distortion reverses direction across the image radius.
- Visual signature: barrel near centre, pincushion near edges (or vice versa); apparent in wide-angle and zoom lenses.
- Formula:
  $$r_u = r_d\!\left(1 + \kappa_1 r_d^2 + \kappa_2 r_d^4\right), \quad r_d > 0$$
- Condition: $\text{sign}(\kappa_1) \neq \text{sign}(\kappa_2)$.
- Pixel-centred form extends naturally: replace $1 + \kappa_1^p r_d^2$ with $1 + \kappa_1^p r_d^2 + \kappa_2^p r_d^4$.

### Metric vs. pixel formulation

- **Metric:** work in mm; use $\kappa_1^m$ (mm$^{-2}$), $r_d$ in mm. Conversion: $r_d^m = r_d^{px} \times \text{pixel\_size}$.
- **Pixel:** work in pixels; use $\kappa_1^p = \kappa_1^m \times (\text{pixel\_size})^2$.
- Both give identical physical corrections; choose whichever unit the calibration output provides.

### Design-time $\kappa_1$ bound (A4 reverse direction)

- Used when specifying a camera system: given a maximum allowable distortion error $\Delta$ at the worst corner, find the tightest $\kappa_1^m$ the lens must satisfy.
- Formula: $\kappa_1 = \Delta / r_d^3$ in consistent units (all px or all mm).
- Worst-case corner: $(0,0)$, $(w,0)$, $(0,h)$, or $(w,h)$ — whichever is farthest from $(c_x, c_y)$; for a centred principal point, all four corners are equidistant.

---

## Related topics

- [[tsai-calibration]] — $\kappa_1$ is one of the intrinsic parameters estimated by Tsai calibration; distortion removal is applied as part of the Tsai pipeline.
- [[calibration-error-analysis]] — reprojection and back-projection errors increase if distortion is not corrected; distortion removal is Step 1 of the calibration error workflow.
- [[backward-projection-and-ray-intersection]] — back-projection of undistorted pixels to 3D rays; requires distortion removal first.
- [[deep-learning-for-stereo-and-calibration]] — learned calibration models can implicitly account for lens distortion without an explicit $\kappa_1$ model.
