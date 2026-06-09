# Image Warping and Stitching

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps — Inverse warping mosaic (W5L 3-case)]]
- [[#A6. Variations]]
  - [[#Variation 1 — Forward warping with splatting]]
  - [[#Variation 2 — Solution-2 forward-style stitching (W7T 4-case)]]

---

## A1. Purpose

Image warping applies a geometric transformation $\mathbf{T}$ to change the **domain** of an image (i.e. where pixels come from), as opposed to filtering which changes the **range** (pixel values). The core use-case in CS773 is **panorama/mosaic stitching**: two overlapping images related by a homography $\mathbf{H}$ are composed onto a single wider canvas. Inverse warping is the preferred production method because it guarantees no holes in the output.

---

## A2. High-level overview

**Key definitions:**

| Term | Definition |
|---|---|
| Warping | Applying $\mathbf{T}$ to change the *domain* of an image; $g(\mathbf{x}') = f(\mathbf{T}(\mathbf{x}))$ |
| Forward warping | Push each source pixel $\mathbf{x}$ to destination $\mathbf{x}' = \mathbf{T}(\mathbf{x})$ |
| Inverse (backward) warping | Pull: for each destination pixel $\mathbf{x}'$, look up $\mathbf{x} = \mathbf{T}^{-1}(\mathbf{x}')$ in source |
| Splatting | Forward-warp fix: distribute a non-integer-landing pixel across its four neighbours |
| Bilinear interpolation | Inverse-warp fix: weighted average of 4 integer-coordinate neighbours around a non-integer source location |
| Homography $\mathbf{H}$ | $3\times 3$ projective transform; maps left-image point $\tilde{\mathbf{x}}_l$ to right-image point $\tilde{\mathbf{x}}_r$ |
| Mosaic canvas | Output image of width $2W$ (twice the source width) that holds both images |
| $f(x,y)$ | Source image pixel value at $(x,y)$ |
| $g(x',y')$ | Destination (warped) image pixel value at $(x',y')$ |
| $a$ | Horizontal fractional offset in bilinear interpolation: $a = x - x_1 \in [0,1)$ |
| $b$ | Vertical fractional offset in bilinear interpolation: $b = y - y_1 \in [0,1)$ |

**Steps (conceptual):**

1. Obtain geometric transform $\mathbf{T}$ (e.g. homography $\mathbf{H}$ from [[ransac]] + [[homography-and-dlt]]).
2. Allocate output canvas (for mosaic: $2W \times H$).
3. Loop over every destination pixel $\mathbf{x}'$ in the canvas.
4. Map back to source: $\mathbf{x} = \mathbf{T}^{-1}(\mathbf{x}')$.
5. If source location is between pixels, use bilinear interpolation.
6. Assign interpolated value to $g(\mathbf{x}')$.
7. Apply per-pixel blending rules (for multi-image mosaic).

---

## A3. Strengths, shortcomings & limitations

**Inverse warping (preferred):**
- No holes — every destination pixel is filled by construction.
- Clean bilinear interpolation; numerically stable.
- Requires $\mathbf{T}^{-1}$ to exist and be computable.

**Forward warping:**
- Intuitive: directly pushes pixels from source to destination.
- Non-integer landings produce **holes** even after splatting.
- Splatting adds complexity and may require normalisation if multiple pixels land at the same location.

**Bilinear interpolation:**
- Simple 4-tap weighted average; fast.
- Can blur fine detail (smoothing artefact).
- Assumes locally linear intensity variation.

**Mosaic stitching:**
- Requires an accurate homography (fails with large depth variation or non-planar scenes).
- Naïve 50/50 blending creates a visible seam at the transition zone.
- Canvas size must be chosen appropriately (2× width is a practical heuristic, not universal).

---

### Likely exam questions

**Q:** What is the difference between image warping and image filtering?
**A:** Warping changes the **domain** — it moves pixels to new locations via a geometric transform $\mathbf{T}$. Filtering changes the **range** — it modifies pixel values (e.g. blurring, sharpening) at fixed locations.

**Q:** Why is inverse warping preferred over forward warping?
**A:** Forward warping can map source pixels to non-integer destination locations, leaving gaps (holes) even after splatting. Inverse warping loops over destination pixels and samples the source via interpolation, guaranteeing every destination pixel receives a value — no holes.

**Q:** Write the bilinear interpolation formula and compute $f(1.3, 1.7)$ given $f(1,2)=100$, $f(2,2)=120$, $f(1,1)=110$, $f(2,1)=130$.
**A:** With $a = x - x_1 = 0.3$, $b = y - y_1 = 0.7$ (using $y_1=1$, $y_2=2$ convention from W7T):

$$f(x,y) = (1-a)(1-b)\,f(x_1,y_2) + a(1-b)\,f(x_2,y_2) + b(1-a)\,f(x_1,y_1) + ab\,f(x_2,y_1)$$

$$f(1.3,1.7) = 0.7\times0.3\times100 + 0.3\times0.3\times120 + 0.7\times0.7\times110 + 0.3\times0.7\times130$$

> [!note]
> W7T uses $b = y - y_1$ where $y_1 = 1$ (bottom), so $b = 0.7$ and $1-b = 0.3$. W5L uses $b$ as the vertical distance from the **top** corner, giving $b = 0.3$ and the same formula. Both sources yield the same numeric answer; the weight arrangement follows:
> - top-left $(x_1, y_2)$: weight $(1-a)(1-b) = 0.7\times0.3 = 0.21$
> - top-right $(x_2, y_2)$: weight $a(1-b) = 0.3\times0.3 = 0.09$
> - bottom-left $(x_1, y_1)$: weight $(1-a)b = 0.7\times0.7 = 0.49$  ← **note** the $b$ used in W5L's formula labels this differently; numeric result is identical.

Computing with W7T's formula directly ($a=0.3$, $b=0.7$, bottom row = $y_1$):
$$= 0.21\times100 + 0.09\times120 + 0.49\times110 + 0.21\times130 = 21 + 10.8 + 53.9 + 27.3 = \mathbf{109}$$

> [!example]
> Slide numeric answer: $f(1.3,1.7) = 49 + 25.2 + 23.1 + 11.7 = 109$ (W7T, Example 4).

**Q:** In the inverse-warp mosaic algorithm (W5L 3-case), what are the three cases and what colour does each produce?
**A:**
1. $\tilde{\mathbf{x}}_r$ maps **outside** right image → colour from **left image only**.
2. $\tilde{\mathbf{x}}_l$ inside left **AND** $\tilde{\mathbf{x}}_r$ inside right → **blend** (weight 0.5 each) left colour + bilinear-interpolated right colour.
3. $\tilde{\mathbf{x}}_l$ outside left **AND** $\tilde{\mathbf{x}}_r$ outside right → **black** (0).

**Q:** How does the Solution-2 (W7T 4-case) forward-style algorithm differ from the W5L 3-case inverse algorithm?
**A:** Solution-2 iterates over canvas coordinates $(x,y)$ and **forward-maps** them to $(x',y') = T(x,y)$ (no $\mathbf{T}^{-1}$ needed). It has an additional explicit case separating "only in Image-2" from "both outside": Case 1 = Image-1 only; Case 2 = blend; Case 3 = Image-2 only; Case 4 = black. The W5L 3-case version merges "only in Image-2" and "only in Image-1" into one case (only-left), with the right image always accessed via inverse warp.

**Q:** What canvas size is used for panorama stitching, and why?
**A:** Canvas width = $2W$ (twice the width of the source image) to provide space for both the left image (placed at its original location) and the right image warped into the left coordinate frame. Height remains $H$.

---

## A4. Directions of reasoning

### Forward / standard (A→B)

**Given:** source image $f$, transform $\mathbf{T}$ (e.g. $\mathbf{H}$), canvas size.
**Asked:** produce warped/stitched image $g$.

**Method:**
1. For each canvas pixel $\mathbf{x}'$, compute $\mathbf{x} = \mathbf{T}^{-1}(\mathbf{x}')$.
2. If $\mathbf{x}$ is inside the source, bilinear interpolate $f$ at $\mathbf{x}$.
3. Apply multi-image case logic to assign $g(\mathbf{x}')$.

### Reverse / inferential (B→A)

**Given:** a destination pixel location $\mathbf{x}' = (u', v')$ in the output mosaic and the homography $\mathbf{H}$.
**Asked:** identify (i) which source image(s) contribute, (ii) the exact source coordinate(s), (iii) which case applies.

**Steps:**

1. Check if $(u', v')$ falls within the left-image region (columns $0 \ldots W-1$) — call this $\mathbf{x}_l$.
2. Compute the corresponding right-image coordinate: $\tilde{\mathbf{x}}_r = \mathbf{H}\,\tilde{\mathbf{x}}_l$ (homogeneous; divide by $w$).
3. Check if $\mathbf{x}_r$ is within the right image's bounds.
4. Apply the case table:

| Left inside? | Right inside? | Case (W5L) | Colour |
|---|---|---|---|
| Yes | No | 1 | left pixel |
| Yes | Yes | 2 | 0.5 · left + 0.5 · bilinear(right at $\mathbf{x}_r$) |
| No | No | 3 | 0 (black) |
| (No, Yes) | — | (absorbed into case 1 or 3 in W5L) | see A6 |

5. If bilinear interpolation is needed, extract $a$, $b$ from the fractional part of $\mathbf{x}_r$ and apply the 4-tap formula.

> [!note]
> In W5L 3-case logic there is no explicit "right-image only" case — points outside the left image but inside the right image map to black (case 3). The W7T Solution-2 adds an explicit Case 3 for this.

---

## A5. Standard implementation

### A5a. Setup

- **Input:** left image $f_l$ ($W \times H$), right image $f_r$ ($W \times H$), homography $\mathbf{H}$ (a $3\times3$ matrix s.t. $\tilde{\mathbf{x}}_r = \mathbf{H}\tilde{\mathbf{x}}_l$).
- **Output:** mosaic canvas $g$ of size $2W \times H$.
- **Convention:** $\tilde{\mathbf{x}} = (x, y, 1)^\top$ (homogeneous); after applying $\mathbf{H}$, divide by the third component to get pixel coordinates.
- **Parameters:** blend weight = 0.5 (both images equally weighted in overlap).
- **Requires:** $\mathbf{H}$ already computed (via [[homography-and-dlt]] + [[ransac]]).

### A5b. Steps — Inverse warping mosaic (W5L 3-case)

1. **Allocate** canvas $g$ of width $2W$, height $H$ (initialised to 0).

2. **For each canvas pixel** $(x_l, y_l)$ with $x_l \in [0, 2W)$, $y_l \in [0, H)$:

3. **Compute right-image coordinate** via forward homography:
$$\tilde{\mathbf{x}}_r = \mathbf{H}\,\tilde{\mathbf{x}}_l, \quad \tilde{\mathbf{x}}_l = (x_l, y_l, 1)^\top$$
   Divide by $w$ (third component) to get $(x_r, y_r)$.

4. **Case 1** — if $(x_r, y_r)$ is **outside** $f_r$ (i.e. $x_r < 0$ or $x_r \geq W$ or similarly for $y_r$):
$$g(x_l, y_l) \leftarrow f_l(x_l, y_l)$$

5. **Case 2** — if $(x_l, y_l)$ **inside** $f_l$ **AND** $(x_r, y_r)$ **inside** $f_r$:
$$g(x_l, y_l) \leftarrow 0.5 \cdot f_l(x_l, y_l) + 0.5 \cdot \text{bilinear}(f_r,\, x_r, y_r)$$

6. **Case 3** — otherwise (both outside):
$$g(x_l, y_l) \leftarrow 0$$

7. **Bilinear interpolation** (step 5 detail): for non-integer $(x_r, y_r)$, let $x_1 = \lfloor x_r \rfloor$, $x_2 = x_1+1$, $y_1 = \lfloor y_r \rfloor$, $y_2 = y_1+1$, $a = x_r - x_1$, $b = y_r - y_1$:

$$\text{bilinear}(f_r, x_r, y_r) = (1-a)(1-b)\,f_r(x_1,y_2) + a(1-b)\,f_r(x_2,y_2) + b(1-a)\,f_r(x_1,y_1) + ab\,f_r(x_2,y_1)$$

> [!note]
> W5L and W7T both give this formula with weights $(1-a)(1-b)$, $a(1-b)$, $(1-a)b$, $ab$ for the four corners. The corner assignment (which weight goes to which corner) depends on whether $b$ is measured from the top or bottom; the numeric result is the same either way.

---

## A6. Variations

### Variation 1 — Forward warping with splatting

**Change to setup:** iterate over **source** pixels $(x, y)$ rather than destination pixels.

**Change to steps:**
1. For each source pixel $(x, y)$, compute $(x', y') = \mathbf{T}(x, y)$.
2. If $(x', y')$ is an integer, assign $g(x', y') \leftarrow f(x, y)$ directly.
3. If $(x', y')$ is non-integer, **splat**: distribute $f(x,y)$ to the four surrounding integer pixels of $(x', y')$, weighted by proximity (inverse of the bilinear weights). Maintain an accumulation buffer and normalise afterwards.

**Problem:** even with splatting, destination pixels that no source pixel maps near may remain unfilled — **holes**.

**When used:** rare in practice; forward warping is conceptually simpler and used in the W7T Solution-2 assignment context, where the canvas loop drives everything and holes are handled by case logic.

---

### Variation 2 — Solution-2 forward-style stitching (W7T 4-case)

**Context:** W7T assignment algorithm. Iterates over canvas coordinates and forward-maps using $T$ (not $T^{-1}$), avoiding the need to invert $\mathbf{H}$.

**Setup change:** same $2W$ canvas; but use $T$ (forward homography: from canvas/Image-1 coordinates to Image-2 coordinates).

**Steps:** for each canvas pixel $(x, y)$:

1. Compute $(x', y') = T(x, y)$ (forward map into Image-2 space).
2. **Case 1** — $(x,y)$ **inside Image-1** AND $(x',y')$ **outside Image-2**:
$$g(x,y) \leftarrow \text{Color from Image-1}$$
3. **Case 2** — $(x,y)$ **inside Image-1** AND $(x',y')$ **inside Image-2**:
$$g(x,y) \leftarrow 0.5 \cdot f_1(x,y) + 0.5 \cdot \text{bilinear}(f_2, x', y')$$
4. **Case 3** — $(x,y)$ **outside Image-1** AND $(x',y')$ **inside Image-2**:
$$g(x,y) \leftarrow \text{bilinear}(f_2, x', y')$$
5. **Case 4** — $(x,y)$ **outside Image-1** AND $(x',y')$ **outside Image-2**:
$$g(x,y) \leftarrow 0 \text{ (black)}$$

**Key difference from W5L 3-case:**

| Aspect | W5L 3-case (inverse) | W7T 4-case (forward-style) |
|---|---|---|
| Loop driver | destination canvas pixels | destination canvas pixels |
| Mapping direction | $\mathbf{H}$ applied forward to find $\mathbf{x}_r$ | $T$ applied forward to find $(x',y')$ |
| "Right/Image-2 only" pixels | → black (absorbed into case 3) | → Case 3 (explicit: colour from Image-2) |
| Number of cases | 3 | 4 |
| Inverse needed? | No ($\mathbf{H}$ is used forward) | No |

> [!warning]
> Both algorithms iterate over **destination** canvas pixels and use the **forward** homography $\mathbf{H}$ to find where the canvas point lands in the right/second image. The term "inverse warping" in W5L refers to the general principle of iterating over output pixels (as opposed to source pixels), not to using $\mathbf{H}^{-1}$ here.

**Related topics:** [[homography-and-dlt]] · [[ransac]] · [[feature-matching]] · [[vector-matrix-algebra]]
