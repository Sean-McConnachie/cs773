# Non-Maximum Suppression

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

After thresholding the corner-response map $C$ of the [[harris-corner-detector]], many adjacent pixels exceed the threshold and would all be reported as separate corners — yet they all belong to the same physical corner. NMS reduces this to exactly one detection per local peak by retaining a pixel only if its response is the strict maximum within its neighbourhood, setting all others to zero. It is **step 8** (the last step) of the Harris pipeline and appears whenever a dense response map must be converted into a sparse set of feature locations.

---

## A2. High-level overview

**Key definitions**

| Term | Definition |
|---|---|
| $I$ | Input corner-response image (output of thresholding $C$) |
| $I'$ | Output image after NMS — zeros everywhere except at local maxima |
| 3×3 window | The neighbourhood centred on each pixel used to test maximality |
| BorderBoundaryPadding | Border-handling strategy: replicate the boundary pixel values outward (do **not** zero-pad) |
| Local maximum | Pixel whose value is the **strict** maximum of all 9 values in its 3×3 neighbourhood |
| $(y, x)$ | Coordinate convention for reported corners (row first, then column) |

**Steps (no math)**

1. Pad the image borders using boundary-replication so every pixel has a valid 3×3 neighbourhood.
2. For each pixel $(y, x)$: extract its 3×3 neighbourhood.
3. If the pixel's value equals the maximum of that neighbourhood, write that value into $I'$; otherwise write 0.
4. Collect all positions where $I' \neq 0$ as the final corner list, in $(y, x)$ order.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Simple and fast — one pass over the image with a fixed-size window.
- Guarantees at most one detection per local region, eliminating duplicate corners.
- Produces sparse, unambiguous feature locations suitable for downstream matching.

**Shortcomings / limitations**
- Window size is fixed (typically 3×3); small windows may still yield clusters of maxima, while large windows may merge nearby distinct corners.
- Strict-maximum criterion means that if two adjacent pixels share the identical maximum value, *neither* is kept (tie-breaking is not defined in this formulation).
- Sensitive to the preceding threshold $T$: a too-low threshold floods the map with above-threshold values, and NMS must pick one per cluster; a too-high threshold discards real corners before NMS even runs.
- Does not consider the *relative* strength of the response across the image — a weak local maximum in a flat region passes NMS just as a strong corner does.

### Likely exam questions

**Q:** Describe the NMS algorithm as applied to a corner-response map. What does "BorderBoundaryPadding" mean and why is it used instead of zero-padding?
**A:** For each pixel, compare its value to all pixels in a 3×3 window centred on it. Keep the value if and only if it is the strict maximum; otherwise write 0. BorderBoundaryPadding extends the image by replicating the value of the nearest border pixel outward, so border pixels are tested against real (replicated) neighbour values rather than artificial zeros that could falsely make them appear as maxima.

**Q:** Given the 5×5 response image below, apply NMS and list the surviving corners in $(y, x)$ order.

```
 8  10   7   6   0
 5   7   9  15   7
 2   9   7   6  14
 3   4   1   1   9
 7   8  17  20   9
```

**A:** Corners are `[(4, 1), (3, 3)]`. Pixel $(4,1)=10$ is the maximum of its 3×3 neighbourhood (max = 10); pixel $(3,3)=15$ is the maximum of its neighbourhood (max = 15). All other above-threshold pixels are dominated by a neighbour.

> [!note]
> Coordinate system: origin $O$ at bottom-left, $x$ rightward, $y$ upward. Row $y=4$ is the top row of the stored array.

**Q:** Why does pixel $(4,0)$ (value 8) not survive NMS even though its value is relatively large?
**A:** Its 3×3 neighbourhood (using border replication on the left edge) contains pixel $(4,1)=10$, which is larger. 8 is not the strict maximum, so it is suppressed.

**Q:** A student zero-pads the border instead of replicating boundary values. How could this change the output?
**A:** Zero-padding inserts artificial zeros as neighbours. A border pixel with any positive value would then always be a local maximum (since its zero-padded neighbours are all 0), producing spurious corner detections along the image boundary.

**Q:** In the Harris pipeline, at what stage is NMS applied and what is its input?
**A:** NMS is step 8 — the final step. Its input is the thresholded corner-response map (pixels where $C > T$ retain their $C$ value, others are set to 0). NMS then removes all but the local maxima of that map.

**Q:** If the NMS window is increased from 3×3 to 7×7, what is the expected effect on the output?
**A:** Fewer corners are retained. Each surviving pixel must now be the maximum over a larger region, so nearby peaks that would have survived with a 3×3 window are suppressed. This reduces clutter but risks merging genuinely distinct corners that are close together.

---

## A4. Directions of reasoning

### Forward / standard (input map → corner list)

**Given:** Corner-response image $I$ (values after thresholding).
**Asked:** Which pixels survive? What is $I'$?
**Method:**
1. Apply BorderBoundaryPadding.
2. For each $(y, x)$: compute $\max$ of 3×3 neighbourhood. If $I(y,x) = \max$, set $I'(y,x) = I(y,x)$; else $I'(y,x) = 0$.
3. Corner list = $\{(y,x) : I'(y,x) \neq 0\}$.

### Reverse / inferential (output map → which inputs survived and why)

**Given:** Output map $I'$ (shows zeros and a few non-zero values) plus (optionally) the original $I$.
**Asked:** Identify which pixels survived and justify each.
**Method:**
1. Every non-zero pixel in $I'$ is a local maximum — verify by reconstructing its 3×3 neighbourhood from $I$ and confirming no neighbour is strictly larger.
2. Every zero in $I'$ that was non-zero in $I$ was suppressed — identify the specific neighbour that dominates it.

> [!example]
> In the 5×5 worked example, $I'(3,4)=0$ even though $I(3,4)=7$. The neighbourhood of $(3,4)$ contains $(3,3)=15$, which is the dominant value. Conversely, $I'(3,3)=15$ because no neighbour in its 3×3 window (top row $y=4$: values 6,0; surrounding pixels) exceeds 15.

---

## A5. Standard implementation

### a. Setup

| Item | Value |
|---|---|
| **Input** | Image $I$ of size $H \times W$ (corner-response values; non-corner pixels already zeroed by thresholding) |
| **Output** | Image $I'$ of same size; sparse corner list in $(y,x)$ order |
| **Window** | 3×3 centred on each pixel |
| **Border handling** | BorderBoundaryPadding: replicate boundary pixel values (not zero-pad) |
| **Maximality** | Strict — pixel survives iff $I(y,x) \geq I(y',x')$ for all $(y',x')$ in the 3×3 window and $I(y,x) > 0$ |

### b. Steps

1. **Pad** $I$ by 1 pixel on all four sides using boundary replication:
$$I_{\text{pad}}(y,x) = I\!\left(\mathrm{clip}(y,0,H-1),\; \mathrm{clip}(x,0,W-1)\right)$$

2. **Initialise** output $I' = \mathbf{0}_{H \times W}$.

3. **For each pixel $(y, x)$, $0 \le y < H$, $0 \le x < W$:**

$$m(y,x) = \max_{\substack{dy \in \{-1,0,1\}\\ dx \in \{-1,0,1\}}} I_{\text{pad}}(y+dy+1,\; x+dx+1)$$

4. **Assign to output:**

$$I'(y,x) = \begin{cases} I(y,x) & \text{if } I(y,x) = m(y,x) \\ 0 & \text{otherwise} \end{cases}$$

5. **Collect corners:**

$$\text{corners} = \bigl[(y,x) : I'(y,x) \neq 0\bigr] \quad \text{in } (y,x) \text{ order}$$

> [!example]
> **Full 5×5 worked walkthrough (W4T):**
>
> Input $I$ (rows $y=4$ top to $y=0$ bottom, columns $x=0$ to $x=4$):
> ```
>  8  10   7   6   0     ← y = 4
>  5   7   9  15   7     ← y = 3
>  2   9   7   6  14     ← y = 2
>  3   4   1   1   9     ← y = 1
>  7   8  17  20   9     ← y = 0
> ```
>
> Scanning top-to-bottom, left-to-right with BorderBoundaryPadding:
>
> | Pixel $(y,x)$ | Value | 3×3 max | Survives? | $I'$ value |
> |---|---|---|---|---|
> | $(4,0)$ | 8 | 10 | No | 0 |
> | $(4,1)$ | 10 | **10** | **Yes** | **10** |
> | $(4,2)$ | 7 | 10 | No | 0 |
> | $(4,3)$ | 6 | 15 | No | 0 |
> | $(4,4)$ | 0 | 15 | No | 0 |
> | $(3,0)$ | 5 | 10 | No | 0 |
> | $(3,1)$ | 7 | 10 | No | 0 |
> | $(3,2)$ | 9 | 15 | No | 0 |
> | $(3,3)$ | 15 | **15** | **Yes** | **15** |
> | $(3,4)$ | 7 | 15 | No | 0 |
> | … | … | … | (all suppressed) | 0 |
>
> **Final corner list:** `[(4, 1), (3, 3)]`
>
> **Partial output $I'$** (top two rows after scanning):
> ```
>  0  10   0   0   0
>  0   0   0  15   0
> ```

---

## A6. Variations

### Variation 1 — 3×3 strict-maximum (standard / default)

The approach described above and used in the W4T tutorial: a pixel survives iff its value equals the maximum of its 3×3 neighbourhood. Window size = 3×3. Used in the Harris pipeline as described in W2L (step 8) and W3T (assignment description).

### Variation 2 — Larger local-window maximum check (assignment approach, W3T)

**Change to setup:** Replace the 3×3 window with a larger window (e.g. 5×5, 7×7, or a user-specified size). The pixel survives iff it is the maximum within that window.

**Effect:** Enforces a larger minimum spatial separation between retained corners. Reduces the number of detections, suppressing clusters of nearby peaks that would each survive a 3×3 NMS. Useful when feature density needs to be controlled or downstream algorithms require well-separated keypoints.

**When used:** The W3T tutorial explicitly identifies this as the **assignment approach**: "Place a window around each pixel; if the pixel's $C$ is not the maximum in the window, discard it." The window size becomes a tunable parameter — larger windows yield sparser, better-separated corners.

> [!note]
> W3T (slide 29) describes two NMS approaches without naming them: the 3×3 standard approach and the local-window-maximum approach used in the assignment. Both share the same principle — keep only the local maximum — but differ in window size and the resulting density of detections.

### Variation 3 — Thresholding before vs. after NMS

**Change:** Apply NMS to the raw (unthresholded) corner-response map $C$, then threshold $I'$, rather than thresholding first and then applying NMS.

**Effect:** When thresholding first, only above-threshold values participate in the maximum comparison (zeros from suppressed pixels act as non-competitors). When NMS is applied first to the raw $C$, then even weak local maxima are retained before thresholding filters them. The two orderings can produce different results near the threshold boundary.

**Note:** The Harris pipeline as stated in W2L and W3T thresholds first (step 7), then applies NMS (step 8).

---

## Links

Related topics: [[harris-corner-detector]] (NMS is step 8 of the Harris pipeline), [[feature-points-and-autocorrelation]] (NMS converts the dense response map into sparse feature points).
