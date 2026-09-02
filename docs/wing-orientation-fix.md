# The fly wing-orientation fix: what was wrong, why, and the mathematics

*Written 2026-09-02. Sources, all in this repo:
`docs/specs/2026-09-01-wing-orientation-from-masks-design.md` (the spec),
`docs/benchmark/2026-09-01-wing-mask-fit/notes.md` (the measurement record),
`.superpowers/sdd/2026-09-01-wing-orientation-from-masks/task-{5,7,8,9-redesign,10-morebouts}-report.md`,
and the code in `third_party/jarvis_jax/jarvis_jax/tracking/{mask_sdf,mask_containment,wing_coverage,appendage_dof,wing_mask_refine,mask_quality}.py`
plus `wing_mask_fit_bout` in `scripts/run_bout.py`. Every number below is from
one of those. Where two of them disagree, both are given.*

> **Status in one line.** The stage is built, tested, wired into the pipeline
> with provenance, and validated on 4 bouts / 8 bout-flies — and it ships
> `wing_mask_fit.enabled: false`. It fixes the defect it was written for; what
> keeps it off is a penetration gate it misses on 7 of 8 female wings, a 60 s
> per-bout-fly budget it misses by 50–100%, and a single recording's worth of
> evidence.

---

## 1. The defect

In the STAC/IK fit the wings should lie roughly flat along the abdomen. Instead
they are rotated inward and the rendered wing blade passes *through* the
abdomen. It is clearest on the female, present on the male's **non-singing**
wing, and the SAM mask visibly disagrees with the rendered wing orientation.

**Measured signature**, using MuJoCo's `mj_geomDistance(wing, abdomen)` —
negative means the collision geoms interpenetrate:

| quantity | value | source |
|---|---|---|
| control (marker-only STAC) wing/abdomen distance, female | **−0.042 to −0.046** model units | notes §12.6, four bouts |
| control wing pitch, female | **−6 to −12°** | notes §12.8, four bouts |
| the model's own **rest** pose, same metric | **−0.0013** (it merely grazes) | spec §1 |
| mask-optimal wing pitch for a folded wing | **−20 to −40°** | spec §4.1 |
| the model's springref (`qpos_spring`) wing pitch | −57.3° (−1.0 rad) | task-5 report §3 |

Median penetration is about 0.043 model units, roughly 20% of a wing length
(spec §1). **The geometry is fine; the fit is not** — the same model at rest
does not interpenetrate.

This is not a bout-28 artifact. Three further bouts (10, 30, 20) were chosen by
a mask-only tracking-quality ranking of all 30 Session0 bouts, calibrated
against bout 28's known good/bad mask split and *nothing else* — no reference to
wing pitch, no reference to the defect. On all three, the female's control
penetration is −0.042…−0.046 and her control wing pitch −6…−12°, i.e. the defect
reproduces exactly (notes §12.6, task-10 report §3).

For scale, bout 28's control pitch values are fly0 (female) −8.4 / −7.8° and
fly1 (male) −5.7 / −11.1°, against penetrations of −0.04442 / −0.04515 (fly0)
and −0.02301 / −0.03324 (fly1) (notes §3, frames 0–1499).

---

## 2. Why the marker solve cannot fix it — the observability argument

### 2.1 The setup

Each wing body carries only **two** usable markers. `KEYPOINT_MODEL_PAIRS` maps
`WingX_base` to the **thorax**, not to the wing body, so only `WingX_V12` and
`WingX_V13` are attached to `wing_left` / `wing_right` (spec §2).

Two points on a rigid body define an **axis, not an orientation**. Rotation
about the line through them is unobservable from those two points alone. V12 and
V13 lie along the blade's long axis, so the unobservable rotation is rotation of
the blade about its own long axis — which is, physically, wing *pitch*.

### 2.2 The Jacobian and its SVD

Stack the two marker positions and differentiate with respect to the three wing
hinge angles:

```
J  =  ∂ [ p_V12 ; p_V13 ] / ∂ [ yaw , roll , pitch ]      ∈  ℝ^{6×3}
```

evaluated at the rest pose with the run's own fitted marker offsets (spec §2):

| quantity | yaw | roll | pitch |
|---|---|---|---|
| per-column sensitivity ‖J e_k‖ | 0.271 | **0.344** | **0.042** |
| right singular vector for σ_min | 0.035 | 0.088 | **−0.996** |

* Condition number `σ_max/σ_min = 13`.
* Roll's column is **8.2×** pitch's (0.344 / 0.042).
* One degree along the null direction moves the two markers **0.46 units**,
  against **6.04 units** for the best-conditioned direction — a ratio of 13.1,
  consistent with the condition number.

### 2.3 What "pitch is 99.6% of the null direction" means

Write the SVD `J = U Σ Vᵀ` with `σ_1 ≥ σ_2 ≥ σ_3`. The third right singular
vector `v_3 = (0.035, 0.088, −0.996)` is the joint-space direction the markers
see *least*. Its pitch component is 0.996, so `v_3` is 99.2% pitch by squared
weight (the spec quotes 99.6%; both readings say the same thing — `v_3` is
pitch, to within a few percent of yaw and roll).

Geometrically: pitch rotates the blade about an axis that both markers nearly
lie on, so their moment arm `r_⊥` about that axis is nearly zero and
`‖Δp‖ ≈ r_⊥ Δθ ≈ 0`. The marker cost

```
C_marker(q) = Σ_k w_k ‖ p_k(q) − p̂_k ‖²
```

therefore has a Hessian `Jᵀ W J` whose smallest eigenvalue lies almost exactly
along pitch: the cost is nearly **flat** in pitch. What sets pitch in the solve
is then not the data but everything else in the objective — the temporal
smoothness term, the joint-limit barrier, the initialisation, whatever
regularisation is present. The fit lands wherever the prior puts it, and that
place happens to be inside the abdomen.

Note the corollary that is easy to get backwards: **roll is the
strongest-observed wing DOF** (0.344, larger than yaw's 0.271). A "wing roll is
weak, regularise it" prior is aimed at the one wing DOF the markers constrain
best. That mistake was made in this codebase before (spec §4.2b records the roll
prior's docstring calling roll the weak DOF), and it is why roll is explicitly
out of scope here.

### 2.4 Marker offsets are not the culprit

All four wing marker offsets sit **on** the wing mesh, within 4–10% of the
blade's thickness of a mesh vertex, checked against the 20,184-vertex wing mesh
rather than the coarser collision ellipsoids (spec §2). The problem is
conditioning, not placement.

---

## 3. The mask objective

### 3.1 Where it sits

```
A  ViTPose 2-D on SAM3-masked crops ........... kp2d.npz
B  DLT triangulation .......................... kp3d.npz
C  STAC ik_only (marker solve) ................ stac_ik.h5
D  model->mm bridge (per-frame s, R, t) ....... qpos_refined.npz     <-- CONTROL
D2 wing-pitch vs the SAM masks (OPT-IN) ....... qpos_wingfit.npz     <-- this doc
E  FK outputs.h5 + qc + overlays
```

Stage D2 is a **post-STAC, wings-only, mask-driven refinement of wing pitch** —
not a term inside the marker solve and not a prior on it. It runs **after** the
bridge, not between C and D as spec §4.4 first said, because the refinement
projects wing mesh vertices to pixels *through* the per-frame bridge
`v_mm = s·(v R^T) + t`, so the bridge has to exist first
(`scripts/run_bout.py`, Stage D2 comment). The same bridges are reused rather
than refitted afterwards: the bridge is a Umeyama similarity fitted to all
keypoints, and this stage moves only the two wing markers whose Jacobian columns
are the direction the bridge is least sensitive to.

Only `wing_pitch_left` and `wing_pitch_right` are optimised, addressed **by
joint name** (`wing_mask_refine.WING_PITCH_JOINTS`); everything else — root,
thorax, abdomen, legs, wing yaw, wing roll — is frozen by `opt_mask`. Measured
consequence: `max |Δ|` over every non-pitch qpos address is **exactly 0.0** in
every arm and every mode (notes §3, §10.5, §12.3).

### 3.2 The two terms, and the diagram

```
                      SAM mask for (frame t, camera c)
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
     signed distance field                 mask pixels the projected
     (crop → EDT → orig px,                BODY does not explain
      negative INSIDE)                     (rasterise body, dilate, subtract)
              │                                       │
              ▼                                       ▼
   ┌──────────────────────┐                ┌──────────────────────────┐
   │ CONTAINMENT          │                │ COVERAGE                 │
   │ one residual per     │                │ one residual per sampled │
   │ WING VERTEX (~100)   │                │ TARGET PIXEL (128)       │
   │ r = relu(SDF + m)    │                │ r = softmin_v ‖p − π(v)‖ │
   │ pulls strays back IN │                │ pulls the blade OUT to   │
   │ one-sided: inside    │                │ cover unexplained mask   │
   │ costs nothing        │                │ area                     │
   └──────────┬───────────┘                └────────────┬─────────────┘
              │                                         │
              └────────────► Σ over 7 cameras ◄─────────┘
                                  │
             + smoothness( Δq )   + soft joint-limit barrier
                                  │
                          Adam, 300 steps, lr 0.01
                                  │
                    Δ pitch(t)  ──┴──►  free | spline | lowpass   (§6)
```

### 3.3 Containment — and why it alone is a trap

`mask_containment.containment_residual`:

```
grid_xy   = (orig_xy − grid_offset) · grid_scale          # into the SDF crop
d(v)      = SDF_bilinear(grid_xy) + overflow(grid_xy)     # ORIGINAL pixels
r_cont(v) = conf · relu( d(v) + margin )                  # gated by `present`
```

The SDF is **signed negative inside, positive outside**, in original image
pixels (verified −25.3 inside / +27.8 outside on a test box, spec §4.2b).
`overflow` is the distance a projected vertex lies beyond the cropped SDF box,
converted back to original pixels, so a vertex far outside the crop still gets a
finite inward gradient instead of a clamped zero-gradient plateau.

The term is **one-sided by construction**: `relu` means a vertex already inside
the mask costs exactly nothing. That is deliberate — the SAM shadow halo only
inflates the mask, so a two-sided term would let the halo drag the fit outward.
But it has a consequence that is fatal if unaddressed:

> **Minimising containment alone is solved perfectly by hiding the wing inside
> the body silhouette** — which is precisely the bug being fixed, not a variant
> of it.

Measured on the synthetic fixture at the springref (rest) attitude, 48–80% of
the wing is already hidden inside the body silhouette per camera, so the
degeneracy is real in the regime the defect lives in (task-5 report §3).

Against that, on real bout-28 data the degeneracy did **not** bite: 21% of the
wing vertices start *outside* the mask and supply the gradient, and
containment-only actually raises `explained%` (38.6 → 45.7 on fly0), which is
the opposite of a wing collapsing into the body blob (task-5 report §2, notes
§4). One bout-fly is not enough to retire a known degeneracy, which is why the
opposing term exists — but the honest reading is that on this data containment
alone was not observed to fail.

### 3.4 Coverage — a one-directional softmin Chamfer

`wing_coverage.wing_target_points` rasterises the projected **body** vertices,
dilates by `dilate_px` (8, shipped), subtracts that from the SAM mask, and
samples `n_target_points` = 128 of the surviving pixels uniformly at random with
a fixed seed. Those are the mask pixels *nothing but a wing* can explain.

For each such target `p`:

```
r_cov(p) = −(1/β) · log Σ_v exp( −β · ‖ p − π(v) ‖ )        β = 8.0
```

This is a **softmin** over the M projected wing vertices: it is a smooth lower
bound on `min_v ‖p − π(v)‖`, and `β → ∞` recovers the hard min exactly. β
controls how sharply the nearest vertex monopolises the gradient — large β gives
a near-hard nearest-neighbour assignment with an almost-discontinuous gradient
as the nearest vertex changes; small β spreads the pull over many vertices and
biases the value below the true minimum. It is computed inside a
`jax.lax.map(..., batch_size=chunk_size)` over target chunks so the peak
intermediate is `O(chunk_size · M)` and the full `(N, M)` distance matrix is
never materialised.

`huber_delta` (8.0 shipped) optionally robustifies: `_huber_sqrt` returns the
square root of the Huber loss so that the sum of squared residuals behaves as a
Huber loss on the raw distance. It matters more than a nicety here — at
`huber_delta = 0` the term is an unrobustified L2 whose gradient *grows* with
distance, so the farthest unexplained pixels dominate: SAM halo, crescents where
the mesh body does not register on the imaged body, an occluding second fly. On
the real bout, `huber_delta = 8` improves the fit at every coverage weight tried
(inside% 85.3→86.2, 81.5→84.6, 74.9→81.0 at three weights; task-5 report §F3).

Two gradient hazards in this file are load-bearing and documented in place: the
`huber_delta` branch must be a **static Python `if`** (a `jnp.where` traces the
unused branch, whose `sqrt(clip(·,0))` has a NaN gradient at 0), and
`_huber_sqrt` itself needs a double-`where` feeding the unselected branch a
constant. Before the second fix, *every* `huber_delta > 0` gradient was NaN
while the values were correct — measured grads `[nan, nan, 1.0, 0.5]` on probes
`d = [1,4,8,20]` at δ=8 (task-5 report §F1-adjacent).

### 3.5 The honest part: the two terms are not commensurate

Containment contributes **one residual per wing vertex** — a fixed geometric
quantity, ~100 of them in the `fps_300` subset, of which about **79% are exactly
zero** because the vertex is already inside the mask. Coverage contributes **one
residual per sampled target** — 128 per camera × 7 cameras — of squared *raw
pixel* distances, and that count is a free knob.

Summing both without normalisation left the two terms roughly **two orders of
magnitude apart**. At the nominal `containment_weight == coverage_weight == 0.3`
the objective was effectively **coverage-only**, and the break-even against
containment sat near `coverage_weight = 0.03` on Session0 bout 28 fly0
(`wing_mask_refine` module docstring; config comment at
`configs/pipeline.yaml:338`).

What that produced, before the fix (task-5 report §2, real bout, first 512
frames):

| config | inside% | explained% | median Δpitch L / R |
|---|---|---|---|
| STAC, no refinement | 79.3 | 33.2 | — |
| **coverage 0.30 as specified** | **74.7** | 36.0 | **+73.8° / −25.8°** |
| coverage 0.03 | 85.2 | 37.6 | −2.0° / −26.4° |
| coverage 0.00 | **86.2** | **38.8** | −29.8° / −29.1° |

At the specified weight the refinement drove the wing *out of* the silhouette
and swung the left wing +74°, most of the joint range — and lost on
`explained%`, the metric coverage exists to maximise.

**The fix is per-target normalisation.** `_frame_cost` takes the **mean** over
the finite targets rather than the sum:

```
cost(frame, camera) = Σ_v ( w_cont · r_cont(v) )²
                    + (1 / N_finite) · Σ_p ( w_cov · r_cov(p) )²
```

This makes `coverage_weight` invariant to `n_target_points` and puts the two
terms on one scale. The rescale is exact and was verified numerically:
normalised `coverage_weight = 3.0` reproduces the old unnormalised `0.3`
(74.9/36.0 vs 74.7/36.0) and normalised `0.3` reproduces the old `0.03`
(85.3/37.6 vs 85.2/37.6) — a 10× weight ratio, i.e. **100× in the squared cost**,
confirming the ~100× argued above (task-5 report §F3).

**And then, with normalisation and Huber in place, the coverage term is roughly
inert on real data — and its status is recorded UNDECIDED, not settled.**

* On bout 28, `coverage_weight` 0.0 and 0.3 agree to three significant figures
  on every headline metric on both flies, and the stage's own `dpitch` medians
  differ by ≤ 0.4° (notes §4).
* It is *not* strictly inert, and every real-data difference it makes on the
  hard fly is adverse: on fly0, `hp_rms(wing_pitch_left)` 2.5188 → 2.5785
  (+2.4%) and right-wing penetration −0.026717 → −0.027461 (2.8% **worse**) going
  0.0 → 0.3. At 1.0 the female's added jitter nearly doubles (2.58 → 5.09) and
  left/right symmetry breaks (notes §9.4).
* In the synthetic ground truth (25° perturbation, error after refinement) its
  only win is at the **rest** attitude — containment-only 2.49° → 1.14° with
  coverage at `huber_delta` 8 — and it is offset by an almost equal loss at the
  **extended** attitude (containment-only 0.53° → 1.63°), which is the attitude
  the song criterion exists to protect. Coverage **alone** at rest is
  catastrophic: 42.75–43.49°, i.e. 18° *further from truth than doing nothing*
  (notes §4, task-7 report §4).

Verdict of record: **UNDECIDED, leaning 0.0**. The 0.3 on disk is *inherited*,
not chosen; read the config comment before treating it as a decision.
`huber_delta` must stay > 0 whatever the weight.

### 3.6 The rest of the objective

Per chunk, over frames `f` and the two pitch DOFs (`_refine_chunk`):

```
J(θ) =  Σ_f  maskcost(q_f)                                    # §3.3 + §3.4
      +  smooth_weight² · [ Σ_f (q_{f+1} − q_f)² + (q_0 − q_prev)² ]
      +  limit_weight²  · Σ_f [ relu(q_f − ub)² + relu(lb − q_f)² ]
```

with `q = q_init + Δq(θ)`, minimised by **Adam** (300 steps, lr 0.01). Adam, not
the jaxls Gauss–Newton/LM used elsewhere in the pipeline, because jaxls damps
with a non-scale-invariant `λI` and the pixel-scale silhouette Jacobian has
`|diag(JᵀJ)| ~ 1e6`, so every GN step overshoots and is rejected — measured, a
plain gradient step cut the cost ~45% while LM rejected all steps. Adam's
per-coordinate normalisation also makes the step ≈ `lr` radians regardless of
cost scale (spec §4.2, `wing_mask_refine` docstring).

Note the magnitude of the smoothness term: `Δq` is in **radians** against a
raw-pixel mask cost, so the shipped `smooth_weight = 0.005` is a coefficient of
`2.5e−5` — effectively **no temporal coupling at all**, and the fit is
independent per frame. That fact is the whole of §5.

---

## 4. The anisotropy fix in the SDF

The recovered `silhouette_sdf.py` cropped the mask to the fly's bbox, resized
that crop **anisotropically** to a square grid, ran an isotropic distance
transform on the grid, and then rescaled by a single averaged factor:

```python
px_scale = ((x1 - x0) / W + (y1 - y0) / H) / 2.0     # one factor, two axes
sdf      = sdf_resized * px_scale
```

No single scalar can be correct in both axes. On a 40×60 test box — crop 108×72
resized to a 64×64 grid, `grid_scale = (0.5926, 0.8889)`, a **1.50× anisotropy**
— the SDF read **−25.3 px** at the box centre against a **true 20.0 px**
inradius. That is 25.3/20.0 = 1.27, an error the spec and the module docstring
both round to **27%** (25.31/20.00 is 26.6%; the artifacts state 27%, so both
figures are recorded here). For a containment penalty this mis-weights the
inward pull by up to the anisotropy factor, depending on *which way* the vertex
lies outside the mask.

**The fix** (`mask_sdf.mask_to_sdf_crop`): `scipy.ndimage.distance_transform_edt`
takes per-axis pixel spacing, so the transform itself works in original pixels
and no post-hoc rescale is needed.

```python
dy  = (y1i - y0i) / float(H)          # original px per resized px, per axis
dx  = (x1i - x0i) / float(W)
din  = distance_transform_edt( rc, sampling=(dy, dx))    # >0 inside
dout = distance_transform_edt(~rc, sampling=(dy, dx))    # >0 outside
sdf  = dout - din                                        # ORIGINAL px, signed
```

The sampling convention `grid_xy = (orig_xy − grid_offset) · grid_scale` is
unchanged, so the containment code needed no edit.

**Two sources disagree on the post-fix value.** `mask_sdf.py`'s docstring says
the same test box "reads exactly −20.0". The task-2 report's own test transcript
shows the fixture returning **20.250001907348633**, and `progress.md` (line 290)
cross-checks that value against "the fixture's true 20.0 px inradius". Taking
the measured transcript as authoritative, the residual error is **1.2%**,
attributable to the nearest-neighbour resize quantising the boundary — down from
26.6%. The docstring's "exactly" is an overstatement.

---

## 5. The song, and why the first working fit was wrong

**This is the most important section.** The version of the stage that got
placement right destroyed something real, and the acceptance criterion in force
at the time scored it a clean pass.

### 5.1 The marker-only control genuinely carries courtship song in wing pitch

Session0 `2025_10_20_13_20_04`, bout 28, fly1 (male). During song both wings are
driven by the same bilateral motor program; the non-extended wing is pinned
against extension at its yaw joint stop (`wing_yaw_right` sits exactly on the
+85.94° stop) but **still oscillates** — its pitch carries a peak at the
*extended* wing's f0 with peak/floor 34.6 on epoch 1 and 84.2 on epoch 2 (notes
§9.2a).

The decisive statistic is **magnitude-squared coherence** between the two wings'
high-passed pitch at the extended wing's song frequency `f0`:

```
MSC_xy(f) = |S_xy(f)|² / ( S_xx(f) · S_yy(f) )
x = hp(wing_pitch_left),  y = hp(wing_pitch_right),  f0 from the extended wing's YAW spectrum
```

Whole-bout scoring finds **three** extended-wing epochs, at `nperseg = 128`
(notes §10.3, §12.3):

| epoch (frames) | control MSC at f0 | 95% significance floor | 0.8 × control (the gate) |
|---|---|---|---|
| E1 0–869 (left extended) | **0.418** @ −155° | 0.24 | 0.334 |
| E2 869–1515 (right extended) | **0.861** | 0.35 (measured at 869–1500) | 0.689 |
| E3 1515–1943 (left extended) | **0.712** | not recorded | 0.570 |

Two caveats on that table, both from the artifacts:

* **E2's value depends on the window.** Over frames 869–1500 the notes measure
  **0.852** (§9.2b, spec §5.1′b); over the whole-bout epoch 869–1515 they measure
  **0.861** (§10.3, §12.3). Same statistic, different window; neither is wrong.
* The 95% significance floors recorded are **0.24 (E1)** and **0.35 (E2)** at
  `nperseg` 128, plus **0.13** for fly0. A **0.53** floor also appears in the
  record but it is E1 at `nperseg` **256** (K = 5 segments), not E3's floor — no
  E3 significance floor is recorded anywhere in these artifacts.

Supporting evidence that this is song and not shared pose noise:

* the two wings' `hp(pitch)` cross-correlation is a clean **±0.6 oscillation at
  the 6.4-frame song period**;
* the phase is ≈ ±160°, **not** 0 — the two `wing_pitch_*` joints hinge about
  their own body's local +y and the wing bodies are mirror-placed (quats
  `[0,−0.4031,0,−0.9152]` and `[0,0.9152,0,−0.4031]`), so bilaterally symmetric
  motion appears near 180° in these joint coordinates. Read phase against the
  model's axes or in-phase song looks like anti-phase;
* the **negative control behaves**: fly0 (female, no wing extension,
  `|yawL−yawR|` = 3.52°) also reaches MSC ≈ 0.8 at high frequencies — two wings
  on one body share pose noise — but her cross-correlation stays within ±0.2 and
  is aperiodic, and the f0 the test picks on her is 43.7 Hz (an 18.3-frame
  period), not the song's.

`fs = 800 Hz` is **assumed**, not read from config (`cfg.recording` carries no
fps key). The fps-independent form of the claim is the **6.4-frame period**;
`f0 = 125 Hz` holds only at the assumed fs. All ratios are unaffected because
control and treatment share the constant.

### 5.2 A free per-frame fit replaces it

The mask minimum in pitch is **broad** (±10–15°, spec risk 1) and the objective
is independent per frame (§3.6). The optimiser therefore fills the pitch null
direction with per-frame mask noise. Measured, bout 28 fly1, `param_mode: free`:

| quantity | E1 | E2 |
|---|---|---|
| MSC(pitch L, pitch R) at f0 — control | **0.418** | **0.852** |
| — shipped `smooth_weight` 0.005 | 0.038 | 0.070 |
| — `smooth_weight` 30 | 0.008 | 0.139 |
| pitch PSD peak/floor: ctrl → 0.005 → sw30 | 59.4 → 23.4 → 98.1 | 156.6 → 9.2 → 26.3 |
| `corr(hp_treat, hp_control)`: 0.005 / sw30 | 0.358 / 0.328 | **0.017** / 0.012 |
| song pulses at a common threshold: ctrl → 0.005 → sw30 | 43 → **168** → 16 | 29 → 24 → **4** |

Both arms fall **5×–42× below** the coherence gate on both epochs. Pulses are
never *delayed* (median matched offset +0.0 frames in every arm) — at
`smooth_weight` 0.005 they are **buried** under ~4× as many threshold crossings
(168 detected where 43 are real, 97% chance match); at 30 they are **removed**
(0 of E2's 29 kept). On the female, who does not sing, the free fit *adds*
oscillation the video does not have: `hp_rms` ×9.3 at a peak/floor of **1.0**,
i.e. flat broadband noise.

On E1 the song-band power at f0 actually *rises* ×6.8 under the fit — but the
broadband floor rises ×17 with it, so peak/floor falls, coherence with the
wing's own yaw falls 0.835 → 0.511, and bilateral coherence collapses
0.418 → 0.038. More song-band power, worse song.

### 5.3 Why an amplitude test cannot see this

The acceptance criterion in force was: *`hp_rms` of `wing_pitch_*` on the singing
wing must stay within 20% of control.* On **E2 the shipped configuration reads
`hp_rms` = 1.00× — a clean PASS — on exactly the epoch where the coherence,
peak/floor, content-retention and pulse tests all show the song destroyed**
(notes §9.2, spec §5.1′g).

An RMS cannot distinguish preserved song from substituted noise of equal RMS.
The correlation with the content it replaced is **0.017**: the high-frequency
content was not *modified*, it was *substituted*. An amplitude test also cannot
separate "jitter added" from "song energy moved into this DOF by a changed wing
orientation"; that second ambiguity is what made the first reading of this
measurement wrong.

The criterion failed three ways at once, and each failure is worth carrying
forward (spec §5.0):

1. **An absolute control value the pipeline does not produce.** The quoted
   "control 0.3615" came from `wing_pitch_rest_prior_ab.py`'s weight-0 arm,
   which re-solves STAC at `smooth_weight = 0.1` — 20× the pipeline's 0.005. The
   shipped `qpos_refined.npz` over the same frames gives **0.726**. Gating
   absolutely on 0.3615 would have scored the **unmodified control** a 100%
   failure. All gates are now ratios to the run's own control.
2. **One "singing wing" label per bout.** fly1 **swaps extended wing at frame
   869** (left-extended fraction 1.00 over 0–799, 0.00 over 900–1499; 101-frame
   majority filter). Scoring with one whole-bout label **inverted the verdict on
   which configuration passes**. Whole-bout scoring later found a *third* epoch
   at 1515–1943 that the 0–1500 window could not see.
3. **Amplitude blindness**, above.

### 5.4 Why coherence between the two wings is the right discriminator

Bilateral motor drive is **phase-locked across wings**: the two pitch traces
share a frequency *and* a stable phase. Independent per-frame mask noise cannot
manufacture that — the two wings' residuals are fitted separately from different
projected vertices — and smoothing cannot manufacture it either. The test is
also **pairwise**, so it needs the epoch split only to pick `f0`; it never has
to name "the singing wing", which is exactly where the old criterion broke.

The replacement criterion (spec §5.1′) is layered deliberately:

| id | test | threshold | why it is not the primary |
|---|---|---|---|
| 1′b | `MSC(hp pitch_L, hp pitch_R)` at f0 | `≥ max(0.8 × control, own 95% sig)`, verdict must hold at `nperseg` 64/128/256 | **PRIMARY** |
| 1′c | song-band `peak/floor` | ≥ 0.5 × control | it is a ratio, so smoothing *inflates* it — sw30 scores 1.65× control on E1 with a coherence of 0.008 |
| 1′d | `corr(hp_treat, hp_ctrl)` | ≥ 0.5 | a preservation-only gate would fail a fit that *recovered* song; an escape hatch exists for failing 1′d while raising 1′b |
| 1′e | pulse count within 0.5–2× control, `|median offset| ≤ 1` frame, cross-correlation **periodic** at 1/f0 | — | the periodicity requirement is what makes 1′b specific rather than firing on shared pose noise |
| 1′f | non-singing fly: no new spectral peak, `hp_rms ≤ 2×` | see §7 | negative control |
| 1′g | `hp_rms`, `lp_std` | **reported, never gated** | the statistic that was blind |

`smooth_weight` was swept 0.005 / 0.5 / 5 / 10 / 20 / 30 / 50 on both flies and
is **the wrong axis**: it controls the *amplitude* of high-frequency content,
and the defect is its *content*. Both ends destroy the phase lock, and every
mask metric is flat across the whole range (`inside%` 92.5, `explained%` 47.7,
penetration and residual unchanged to four decimals). Weight 30 buys E1's
amplitude (0.96×) by taking **37% of E2's real dynamics** and all 29 of its
pulses — the rejected rest prior's own failure mode. No single weight satisfies
both epochs.

---

## 6. The fix: band-limiting the correction

Placement and fine motion are separable here — the fit puts the wing in the
right *place* and gives it the wrong *texture*, and `lp_std` shows the slow
structure survives in every arm (fly1 E1: control 5.14 → 11.40 shipped → 11.56
at sw30). So the stage should not carry high-frequency content **at all**.

### 6.1 The reparameterisation

```
Δpitch(t) = [ B θ ](t),      B = knot_basis(T, K) ∈ ℝ^{T × (n_seg+1)},   n_seg = round(T/K)
```

`B` is a piecewise-linear interpolation basis with knots every `K` frames. The
optimisation variable is **θ**, the knot vector, not the per-frame offset. Rows
of `B` sum to 1 (partition of unity), so a constant θ is a constant offset — the
DC component the placement correction actually needs. Parameter count drops by a
factor of ~`K` (28-fold at the measured configuration).

**Chunk boundaries are part of the basis, not an afterthought.** Three chunks
splined independently and stitched have a *step* at every boundary, and a step
is broadband. Each chunk's first knot is clamped to the previous chunk's last,
and `knot_basis` puts its last knot at local position `F` (== the next chunk's
local 0), so the whole trajectory is **one globally continuous piecewise-linear
function**. Hence the requirement `frame_chunk % knot_spacing == 0`, enforced at
`wing_mask_refine.py:829` (256 % 64 == 0). Removing the anchor is
mutation-checked: the correction then leaves the global span by 2.81e−03 of its
own amplitude against a 1e−04 gate.

The frame gate has to move with the parameterisation. In `free` mode "a frame
with no mask evidence stays at its STAC pose" is enforced on the parameter
**update**. One knot spans many frames, so in `spline` that is impossible; the
gate moves onto the **cost** (`w_f`) instead, and with no evidence anywhere the
objective is identically zero and the knots never move — the contract still
holds exactly. A frame inside a *short* gap is interpolated rather than frozen,
because freezing it would put a step back in.

### 6.2 The control: post-hoc low-pass

`param_mode: lowpass` solves exactly as `free`, then keeps only the slow half:

```
q_out = q_init + LP( q_fit − q_init )
LP = zero-phase (filtfilt) 4th-order Butterworth at 1/(2K) cycles/frame
```

Zero phase matters: a causal filter would delay the placement correction by tens
of frames, and placement is the half that works. 4th order rather than 2nd
because at K = 32 the cutoff is only 3.2× below the scorer's own 40 Hz
high-pass, where a 2nd-order Butterworth still passes ~1% of the amplitude;
4th order takes that to ~1e−4.

`lowpass` exists as the **honest control** for `spline`: it preserves the song
equally well by construction but spends the whole solve on content it then
discards. **If the two land in the same place, `spline` is not "a better fit",
only a cheaper parameterisation** — and they do land in essentially the same
place. On the male, `free`, `spline` and `lowpass` agree to 0.1 pt of
`inside%`/`explained%` and ~3 pt of gap closure. On the female's clean stretch
`spline32` beats both on `inside%` (92.8 vs 92.3 / 92.5), `explained%`
(46.5 / 45.8 / 46.0) and both penetrations — **but not on the residual**: at
0–1500 the ranking is lowpass32 −23.33% < spline64 −23.25% < spline32 −22.85% <
free −22.54%, i.e. spline32 is the *worst* of the three band-limited arms there
(best over the whole bout at −24.11%; the metric flips with the window). Real,
marginal, axis-dependent, and only on the hard fly.

### 6.3 Why the song survives — stated as a factor, not an impossibility

The intuitive claim is that the basis cannot represent a 6.4-frame period when
K = 64. **That claim, as written, is false and was refuted by measurement.** A
C0 (piecewise-linear) basis is only *piecewise* smooth, and its kinks carry
`1/f²` energy at **every** frequency. The unit test that appeared to prove
otherwise only ever asserted < 1e−3 of variance.

The correct, measured statement is a **suppression factor**:

```
suppression = hp_rms( free correction ) / hp_rms( this correction )
```

minimum across both flies and both wings (notes §11.2):

| mode | suppression |
|---|---|
| `spline32` | **≥ 15×** (15.6 / 17.8 fly0; 43.7 / 42.6 fly1) |
| `spline64` | **≥ 37×** (37.6 / 37.3 fly0) |
| `lowpass32` | **≥ 129×** (129 / 173 fly0; 673 / 583 fly1) |

**Each conditional on the joint clamp not firing.** The clamp
`clip(q, lb, ub)` is a per-frame nonlinearity applied *after* the basis, so a
clamped frame is no longer in the knot span. Measured: `spline64` hits the
−72.77° stop on fly0's left wing for **17 consecutive frames (1720–1736)**, and
that alone takes its correction **11.45° — 0.146 of its own amplitude — out of
the span** it otherwise lies in to 4.5e−08. Drop those frames and the span
property is restored to 4.5e−08. `n_clamp_hits` is now reported by the stage and
printed in its log; a run with a non-zero count **has not been shown
band-limited**.

Against that, the span property itself holds better on production data than the
fixture showed: **4.5e−08 to 9.6e−08** of the correction's own amplitude across
all 8 chunk boundaries at T = 2007.

The residual leak is visible as a **comb at the knot rate and its harmonics** —
see §7 for its effect on the non-singing negative control, and why
`knot_spacing` 64 was chosen over 32.

### 6.4 What the band-limited arms actually score

Criterion 1′b, fly1, whole bout, `nperseg` 128 (notes §10.3):

| epoch | control | gate | `free` | `spline32` | `lowpass32` |
|---|---|---|---|---|---|
| E1 0–869 | 0.418 | 0.334 | 0.039 | **0.417** | **0.418** |
| E2 869–1515 | 0.861 | 0.689 | 0.111 | **0.861** | **0.861** |
| E3 1515–1943 | 0.712 | 0.570 | 0.043 | **0.712** | **0.712** |

1′d `corr(hp_treat, hp_ctrl)` is **0.996–1.000** for the band-limited arms
against 0.017–0.370 for `free`; pulse counts 30/30, 45/43, 10/10 for `spline32`
(`lowpass32` scores 43/43 on E2); cross-correlation periodicity 0.89–1.00 against
`free`'s 0.19–0.74.

**Read that as the trivial pass it is.** `corr = 0.999` says the band-limited
arms' high-frequency pitch *is* the control's. Criterion 1′ is therefore evidence
only that the stage is **not destructive** — never that the fit is good. What
carries that burden is criteria 2–6 (placement, penetration, residual, figures).

And placement does survive. Bout medians, whole bout: every arm lands in the
measured −20…−40° band and nowhere near the −57.3° springref
(fly0 −33.2/−35.4 spline32 vs −32.4/−35.2 free; fly1 −19.3/−31.0 vs −19.1/−32.0);
criterion 4 (wing-marker residual) is +9.2% on fly1 and −24.1% (improved) on
fly0; `max |Δ|` over every non-pitch qpos address is exactly 0.0 in every mode.

**Two defects in the criterion itself were found by scoring the unmodified
control as its own arm.** Do that from now on; it is the only way to know a gate
is falsifiable rather than unpassable. (i) 1′a applies **per window**, not per
epoch: on fly1 E1, `nperseg` 256 over 869 frames leaves K = 5 segments and a 0.53
significance floor, and the *control's own* MSC there is 0.317 — below its own
floor. Requiring that window fails the unmodified control. Such a window is now
reported and excluded. (ii) A window containing non-finite qpos makes the test
undefined: `hp_filt` is a `filtfilt`, one NaN returns an all-NaN trace, every
`>=` against NaN is False, and the criterion printed a confident FAIL for every
arm *including a copy of the control*. fly0 has 252 such rows and exactly one
long finite run (frames 0–1710); `criterion1prime()` now prints UNDEFINED.

---

## 7. The quality gate

### 7.1 Why area alone is not enough

Bout 28 fly0's masks collapse over frames 1500–2006: median SAM area **15 862 px
→ 4 882 px** (1500–1710) and **2 834 px** (1710–2007), with **40.2%** of the
still-`valid` masks under a quarter of their own camera's median. fly1 is
untouched across the same frames (20 838–22 014 px, 0% slivers, 7/7 cameras).
This is the female pressed against a wall, seen as a truncated fragment.
`min_present_cameras: 3` is blind to it — the masks *are* present, 90.7% of them,
they are just tiny (notes §10.6, §12.1).

But an area gate is still insufficient, for a reason that generalises beyond
this bout: **a wrong-fly or merged-fly mask has perfectly normal area** and
gives itself away only by disagreeing with the pose.

### 7.2 The two signals

`mask_quality.wing_fit_validity_gate` computes, per (frame, camera):

```
camera_ok = valid
          AND  area ≥ gate_min_area_frac · ref_c            # ref_c = that camera's own p75, over its valid frames
          AND  body_inside_fraction ≥ gate_min_body_inside  # POSE-AWARE
frame_keep = ( Σ_c camera_ok ) ≥ min_cameras
```

Shipped: `gate_min_area_frac 0.25`, `gate_min_body_inside 0.5`,
`gate_area_ref_pct 75.0`, `min_present_cameras 3`.

* The reference is **per camera** because cameras see the fly at very different
  scales (bout 28 fly0: 14 238 px on `Cam2012861` vs 20 156 px on `Cam2012853`),
  and a **high quantile, not a median**, because a median is dragged down by a
  long enough bad stretch — at which point the slivers define "normal" and the
  gate silently stops firing. p75 survives a bout up to a quarter bad, which the
  worst measured case (25% of bout 28) exactly is.
* `body_inside_fraction` is the generalisation of the number that made the
  failure visible in the first place: the control's own wing `inside%` drops
  **86.4 → 40.2** on that stretch, and the projected-body version of it is
  already computable.

### 7.3 The subtlety: zeroing `present` does not freeze a frame

The result is used **twice**, and this is the part that is easy to get wrong:

* `camera_ok` gates the **SDF stack**, so sliver evidence never enters the
  objective at all;
* `frame_keep` is passed to `refine_wing_pitch`, so the skipped frames are
  **held at the STAC pose**.

The first alone is not enough. In `free` mode, zeroing a frame's `present` row
does freeze it, because the frame gate lives on the parameter update. **In
`spline`/`lowpass` it does not** — one knot spans many frames, so the knots
interpolate across a gated frame and the low-pass smears across it. An
area-only, `present`-only gate would merely decay the correction toward STAC over
roughly one knot spacing while the frames at the edge of the gap stayed fitted to
slivers.

Hence `apply_gate_and_bound`, which acts on the **correction** rather than the
pose:

```
d(t)   = q_fit(t) − q_init(t)                      # optimised columns only
d(t)  ← clip( d(t), ±max_dpitch_deg )
d(t)  ← gate_envelope(frame_keep, ramp)(t) · d(t)
q_out  = q_init + d
```

The envelope is `e(t) = min(1, dist(t)/ramp)` where `dist` is the frame distance
to the nearest skipped frame; `ramp = knot_spacing` in the band-limited modes and
0 in `free`. It is a ramp rather than a hard 0/1 step because **a step is
broadband** and would put content straight back into the song band; the ramp's
fastest slope is `1/ramp`, no faster than the basis itself.

Both operations can only **shrink** the correction, so neither can push a frame
further outside the joint range than its own STAC pose already was — which is
why there is no re-clip afterwards. With `frame_keep is None` and
`max_dpitch_deg is None`, `apply_gate_and_bound` returns the *same object*, so
`gate_enabled: false max_dpitch_deg: null` is an exact escape hatch to the
pre-gate code path (pinned by a pure-numpy test).

`max_dpitch_deg: 50` is bracketed on both sides: across all four `param_mode`
arms, the largest correction on any **well-tracked** frame of bout 28 is 42.2°,
and the collapsed stretch reaches 71–98°. 50° is also about the distance from the
fitted pose (~−8°) to the springref (−57.3°). The model's own limits
(−72.8…+167.3°) are no constraint at all.

### 7.4 What the gate measured

Bout 28, `spline` / K = 64, whole bout:

| | fly0 OFF | fly0 ON | fly1 OFF | fly1 ON |
|---|---|---|---|---|
| refined / T | 1755/2007 | **1431/2007** | 2007/2007 | 2007/2007 |
| `n_gated_frames` | 0 | **324** (16.1%) | 0 | **0** |
| sliver / pose-reject camera-frames | 0 / 0 | 681 / 711 | 0 / 0 | 0 / 0 |
| `n_at_dpitch_bound` | **34** | **0** | 0 | 0 |
| max \|Δpitch\| 0–1500 L/R | 33.5 / 39.1° | **33.5 / 39.1°** | — | — |
| max \|Δpitch\| 1500–1750 L/R | 50.0 / 50.0° | **0.00 / 0.00°** | — | — |
| wall (s) | 92.3 | 104.1 | 108.0 | 118.5 |

The gated frames are three runs — 1430–1436, 1438–1709, 1711–1750 — i.e. exactly
the collapsed stretch, starting **70 frames before** the hand-drawn 1500
boundary, consistent with the timeline figure showing her mask area sliding
1.0 → 0.5 over frames 1050–1500. `n_at_dpitch_bound` going 34 → 0 is the
cleanest one-line statement of what happened: with the gate on, **no frame needs
the bound, because the evidence driving the correction into it is gone.** On the
well-tracked male the gate is a **measured no-op**: 0 gated frames, median Δpitch
−12.91 vs −12.92°, every metric identical to four significant figures.

Its costs, stated: `inside%` 87.7 → 86.1 and `explained%` 45.7 → 44.1 on fly0
(the skipped frames revert to a worse pose — that is the point and the price);
criterion 3′ gap closure falls on the female; `lp_std` rises; +11 s per bout-fly.

---

## 8. What was tried and rejected, with measurements

Each of these is a path someone would otherwise retry.

**1. Re-map `WingX_base` onto the wing body.** Provably useless. Adding a marker
at the hinge (local `(0,0,0)`) left the singular values **bit-identical**,
because a marker at the rotation centre has an identically-zero rotational
Jacobian. Even at its actual fitted offset (a 5–14% lever arm) the condition
number moves only 13.4 → 12.8. It would also *remove* a thorax constraint.

**2. Rigid-length projection** (`rigid_lengths.enforce_bone_lengths`). Fails
leave-one-camera-out on 4/6 wing keypoints. The error is **angular**, so setting
the correct length slides the point along a wrong direction.

**3. Cross-view L/R swap search** (`wing_lr_assign.resolve_wing_lr`). Fires on
2/1500 male frames. A *collapse* is not a *swap*: permuting two labels that
occupy the same point is a no-op.

**4. A wing-pitch rest prior** (`JAXLS_Q_REG_TO_REST` on `wing_pitch_*`). It
fixes the geometry — left-wing penetration −0.030 → +0.000, wings visibly flat,
`|yawL−yawR|` unchanged at 38 → 39° — and it **removes 75% of the extended wing's
pitch dynamics** (`hp_rms` 0.3615 → 0.0916 at w = 0.01, 0.0153 at w = 0.1) and
makes the folded wing's yaw spiky (kurtosis 4.6 → 145). A yaw-only song guard is
blind to that, which is how it nearly passed. It also aims at the **wrong
value**: the mask wants the folded wing 20–37° away from rest, and rest is
−57.3°.

**5. Tuning `smooth_weight`.** Swept 0.005 / 0.5 / 5 / 10 / 20 / 30 / 50 on both
flies. Both ends destroy the bilateral phase lock (coherence 0.038/0.070 at
0.005; 0.008/0.139 at 30, against a control of 0.418/0.852), and **30
additionally removes 37% of epoch 2's real dynamics** and all 29 of its pulses.
Weight 30 does buy the retired amplitude test (E1 `hp_rms` 3.77× → 0.96×) at zero
cost to any mask metric, and it manufactures a spectral peak on the non-singing
female (peak/floor 6.4 → 46.6). Amplitude is not the axis; frequency content is.
The value is also normalisation-specific: `smooth_weight² · Σ(Δq)²` with Δq in
radians against a raw-pixel cost. Re-derive, never copy, if the mask cost is
rescaled.

**6. Blade-normal angle as an acceptance metric.** It **is** the null direction,
so it is a hypersensitive function of exactly the quantity under test — a 12%
marker-residual change swung it **59°**. It cannot adjudicate a change to that
direction. Report it, never gate on it. (Reported, bout 28: fly1 L 12.2 → 7.7°,
R 15.8 → 2.2°; fly0 L 34.4 → 10.4°, R 30.7 → 5.9°.)

**7. "Pitch alone cannot close the female's penetration gap" — REFUTED.** This
was asserted twice as a structural ceiling and it was an *inference*: nobody had
ever swept pitch against `mj_geomDistance` (spec §4.1 swept it against mask
spill). Swept properly — 50 frames per fly, 241 pitch values across the model's
joint range, everything else at the control pose:

| fly | wing | control median | best achievable | best pitch | frames reaching ≥ −0.005 |
|---|---|---|---|---|---|
| fly0 | left | −0.04440 | **+0.00813** | +78.8° | **100%** |
| fly0 | right | −0.04526 | **+0.00642** | +78.8° | **100%** |
| fly1 | left | −0.02563 | **+0.03289** | +96.8° | **100%** |
| fly1 | right | −0.03677 | **+0.01954** | +97.8° | **100%** |

Pitch lifts the blade entirely clear of the abdomen on **every wing of both
flies, on 100% of sampled frames**. The 31%/41% shortfall is therefore **not**
evidence for adding roll and must not be used to reopen that scope decision.

What it *is* evidence for needs care: the best-achieving pitch (+79° / +97°) is
nowhere near the measured mask optimum of −20…−40° — it is the blade swung out
and up, which is not what a folded wing does. So criterion 3′ is measuring a
**disagreement between the mask objective and the collision metric**, not a DOF
limit. Two explanations are live: either the mask optimum genuinely leaves the
wing resting on the abdomen (real fly wings do), making the −0.0013 target wrong;
or the mask cost picks a wrong optimum. Two caveats the sweep cannot remove:
`mj_geomDistance` uses MuJoCo **collision geoms**, coarser than the
20,184-vertex mesh; and fly0's control penetration is ~2× fly1's while her best
achievable clearance is 4× worse (+0.008 vs +0.033), which points at an
**abdomen-pose or body-scale error on her** rather than a wing error.

**8. Two candidate tracking-quality signals, measured and rejected.** *Centroid
jump* does not separate bout 28's known good/bad split — fly0's p95 is 1.19 px on
her clean stretch and 2.04 px on the collapsed one, against fly1's 3.73 and 5.66
px while he is fine throughout; it ranks motion, not quality. *Mask-area p90/p50*,
the obvious wing-extension (song) proxy, reads 1.119 on bout 28's **singing male**
and 1.215 on its **non-singing female** — it measures occlusion, not song. There
is therefore **no mask-only way to pre-select for song**.

**9. "Gate on mask area" as the mask-evidence remedy.** Could not have worked, for
the reason in §7.3, and it was replaced before implementation.

**10. A C2 (cubic) basis as the fix for the knot comb.** Not implemented, and it
should not be the first thing tried: **raising `knot_spacing` is already measured
and already better.** On the hard fly, `spline64` dominates `spline32` on every
axis — `inside%` 92.89 vs 92.80, gap closure 31.9/41.9 vs 31.1/41.1, residual
−23.25% vs −22.85%, comb 1.06–1.11× vs 1.13–1.56×, suppression 37.6× vs 15.6×,
and the mask-failure excursion +47.4° vs +76.6°. A C2 stitcher would need value +
1st + 2nd derivative continuity across chunk boundaries rather than the single
value clamp the linear basis needs.

**11. `hp_rms` as a primary gate, and `peak/floor` as a primary gate.** Both
retired to "supporting" or "reported": `hp_rms` is blind to substitution (§5.3);
`peak/floor` is a *ratio*, so smoothing inflates it while deleting the song.

**12. Criterion 1′f as literally written.** "peak/floor must not rise above
control's" had no tolerance band, so **only the identity could pass it** —
`lowpass32` scored 3.9402 against 3.9392 and was recorded FAIL by 0.025%. It was
calibrated by measurement rather than judgement: `CRIT1F_PEAK_TOL = 2.0`, set
from a null that re-computes the *control's own* peak/floor on random
half-windows with no change of any kind, which moves it up to **2.07×** (400
draws per wing). A second, per-arm null permutes the arm's own knot vector and
adds it back; `spline32`'s measured 1.13× / 0.34× falls **below the median** of
that null (1.51 / 4.20), so its comb is *less* than a random correction in the
same basis would produce. With the band, every band-limited arm passes 1′f on all
three windows and the arms that should fail still do (`smooth_weight` 30 at
7.28×, `free` on amplitude at 9.25×).

---

## 9. Current status, stated honestly

**What ships.** `wing_mask_fit.enabled: false`, with
`param_mode: spline`, `knot_spacing: 64`. The default changed on 2026-09-02 from
`free` / 32 for a **safety** reason, not a new measurement: `free` is the one
mode measured to destroy the song, and shipping it as the mode you get by
flipping `enabled: true` was unsafe. `spline` preserves the song by construction,
keeps every placement gain, and is what all four validated bouts actually ran.

**What is validated.** 4 bouts (28, 10, 20, 30) × 2 flies = 8 bout-flies, all from
**one recording**, Session0 `2025_10_20_13_20_04`. Bouts 10/20/30 were chosen by a
mask-only quality ranking calibrated on bout 28's split and nothing else, and the
result generalises (notes §12.6, §12.8):

* same defect everywhere — female control penetration −0.042…−0.046, control wing
  pitch −6…−12°;
* same fix — into the −20…−40° band on every wing; penetration closes 35–50% of
  the gap on the female and 50–63% on the male's folded wing; wing-marker residual
  **improves** 11–26% on the female; `inside%` +4…+7 pt; `explained%` +6…+7 pt;
* **criterion 1′ passes on every epoch of every fly**, gate on and gate off;
* the self-targeting property (correct the wrong wing, leave the singer alone) is
  confirmed cleanly on two strongly-extending males: median Δpitch **+0.46° on the
  extended wing against −24.66° on the folded one** (bout 20) and +4.55 / −22.56°
  (bout 30). It is weaker where extension is weaker (bout 10, 14.7° extension:
  −8.28 extended / −15.61 folded), and on bout 30's first epoch the extended wing
  moves ~+15° — a strong tendency, **not a guarantee**;
* the gate is inert where the ranking says it should be (0 frames skipped on all
  six new bout-flies) and removes exactly the stretch it was built for on bout 28.

**What is still open.**

1. **Criterion 3′ falls short on 7 of 8 female wings.** The gate is ≥ 50% gap
   closure, `(control − treatment)/(control − (−0.0013))`, on both wings of both
   flies. Bout 30's right wing clears it at **50.3%** — the first female wing ever
   to do so; the range over the eight female wings is 28–50%, mean ≈ 41%. This is
   now understood as a **disagreement between the mask objective and the collision
   metric, not a DOF limit** (§8 item 7), so the thing to settle first is whether
   −0.0013 is the right target at the mask optimum, and whether fly0's ~2× control
   penetration is an abdomen-pose or body-scale error rather than a wing error.
   The 50% line itself is a judgement and is **not bracketed by a passing
   measurement** on the female.
2. **The 60 s budget is missed.** 92–119 s per bout-fly with the gate (85–97 s
   without). The Adam loop is not the cost — `prefetch=True` overlaps it with
   host-side coverage-target sampling and the serial `sdf_stack_from_masks`, and
   cutting `n_steps` 300 → 100 on top of a 28-fold parameter reduction buys
   **2.6%**. The measured lever is threading `sdf_stack_from_masks`
   (34.4 s → 4.8 s at 16 threads, bit-identical by construction — the output
   arrays are preallocated and each job writes only its own `(t, c)` slot). The
   benchmark record lists that as **deferred**; a threaded `workers` argument
   exists in the working tree at the time of writing but is **not committed**,
   and **no re-measured end-to-end stage time exists** in the record.
3. **The gate puts its own low-frequency step into the female's trajectory.** On
   bout 28 fly0, `lp_std` rises **2.29× → 4.76×** (left wing) and 5.87× → 6.28×
   (right) relative to control. The *rise* is explained — a corrected stretch and
   an uncorrected one joined over one 64-frame knot spacing — but the gate-off
   baseline of 2.29× is not explained anywhere in the record, and `lp_std` is
   **reported, never gated** (1′g), so nothing in the acceptance suite bounds it.
   `hp_rms` is unaffected (1.00×), so nothing is added in the song band.
4. **The gate's thresholds are calibrated on a catastrophe and miss mild
   degradation.** The correction exceeds 45° on 32 frames of bout 10 fly0 and 43
   of bout 20 fly0, each in one contiguous run (1010–1041 and 1086–1128) whose
   median mask-area ratio is 0.833 / 0.715 of that fly's own p75 — far above the
   0.25 sliver line, so only the 50° bound contains the result. On bout 20 the fit
   swings to +42° with the control at −5°; on bout 10 it dives to the −57.3°
   springref (`n_clamp_hits` = 0, so it is the fit's own excursion, not a joint
   stop). **Whether those are real behaviour or fit error is not settled** — the
   control pose moves in the same stretches too. That is the next threshold to
   calibrate.
5. **Criterion 3′ is undefined when the control already has zero penetration.** On
   bout 30 fly1's extended wing, two arms agreeing to 0.03° on the pose get
   opposite verdicts from a 0.005-unit difference in the median of a quantity that
   is zero. The criterion needs an "already clear" branch.
6. **One recording, no human-reviewed `sex.json`.** `male = fly1` rests on the
   pipeline's canonicalisation plus a behavioural check: mean `|yawL−yawR|` is
   5–205× larger on fly1 than fly0 on all four bouts (25.35 / 14.74 / 45.80 /
   41.07° vs 4.78 / 1.42 / 8.14 / 0.20°), which holds even where SAM3's own
   mask-area vote is weak (camera agreement 0.143 on bout 20). Bout 28 fly0's
   `kp3d.npz` is also pre-gate and required `pipeline.allow_stale_kp3d=true` —
   correct for an A/B, but it means her numbers ride on a triangulation that never
   saw the current Stage-B gates.
7. **`coverage_weight` is UNDECIDED, leaning 0.0**, and sits at 0.3 on disk by
   inheritance (§3.5).
8. **The folded wing can overshoot the measured band.** Bout 20 fly1's folded wing
   lands at −44.1° and bout 30 fly1's at −40.6°, past the −40° edge of the measured
   optimum (still 13° clear of the springref).

**Reproducing the older numbers.** The stage's default behaviour changed twice.
Sections of the spec and notes numbered 1–9 reproduce only with

```
wing_mask_fit.param_mode=free wing_mask_fit.knot_spacing=32 \
wing_mask_fit.gate_enabled=false wing_mask_fit.max_dpitch_deg=null
```

`gate_enabled=false` alone is not enough — the |Δpitch| bound fires on 34 frames
of bout 28 fly0. With both off, the `spline64` arm reproduces the saved one to
**0.028°** (0 frames over 0.1°) against a **0.014°** run-to-run solver noise
floor: bit-for-bit is not available from a GPU Adam solve and claiming it would
be false.

---

## Appendix: places where two sources disagree

| topic | source A | source B | resolution |
|---|---|---|---|
| SDF fixture value after the anisotropy fix | `mask_sdf.py` docstring: "reads exactly −20.0" | task-2 report transcript + `progress.md:290`: **20.25** against a true 20.0 | the measured transcript; the residual 1.2% is resize quantisation and the docstring's "exactly" is an overstatement |
| pre-fix SDF error | spec §4.2b and `mask_sdf.py`: "27%" | arithmetic on the same numbers: 25.3/20.0 = **26.6%** | both stated above; the difference is rounding |
| E2 control MSC | notes §9.2b / spec §5.1′b: **0.852** (frames 869–1500) | notes §10.3 / §12.3: **0.861** (frames 869–1515) | different windows, both correct; the whole-bout epoch is the later one |
| bout 28 fly0 gate-on gap closure | notes §12.3: 28.2% / 39.8% | notes §12.6 + task-10 report: **28.1%** / 39.8% | a 0.1 pt transcription difference; nothing turns on it |
| where the stage sits | spec §4.4: "after STAC and before the bridge" | `run_bout.py` Stage D2 comment: **after** `compute_bridges` | the code; the refinement needs the bridge to project at all, and the comment records the correction |
| shipped parameterisation | `wing_mask_refine.py` module docstring and function defaults still describe `free` / K=32 as shipped | `configs/pipeline.yaml`: `param_mode: spline`, `knot_spacing: 64` | the YAML. `wing_mask_fit_refine_kwargs` requires every key explicitly and raises on a missing one, so the module defaults never reach the pipeline — but the docstring is stale |
| V12–V13 rigidity | spec §2: "CV 30% male, 80–120% female" | notes §3, bout 28 frames 0–1499, `kp3d` indexed by name: **5.6–9.1% CV** on both flies, against `EyeL–EyeR` at 2.7–10.2% | the notes say plainly that the spec's figure is not what this bout-fly pair measures; on this bout the wing vein is about as rigid as head width |
| spec cross-reference | spec line 630 points at "section 11" for the 2026-09-02 default change | the spec has no §11; the change is recorded in notes **§13** | dangling reference in the spec |
