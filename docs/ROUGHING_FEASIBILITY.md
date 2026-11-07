# Roughing Feasibility - field-based roughing as a replacement for offset roughing

Status: feasibility assessment, 2026-09-28. No code yet.
Scope: 2.5D and 3-axis roughing only. Finishing and 5-axis are covered in
`ARCHITECTURE_AND_ROADMAP.md` (the "arch doc").

The arch doc asks "what does a field-native CAM kernel look like, up to simultaneous 5-axis?"
This doc asks a narrower, practical question: **can a field-based toolpath generator replace
brittle offset-based roughing on real production parts, and what would it take?**

**Stance:** the goal is to beat the incumbents, not to match them. FreeCAD Adaptive,
ModuleWorks, and the other commercial adaptive-clearing engines are benchmarks to measure
against, not designs to copy. Section 4.4 names what they leave on the table. Sections
4.5-4.6 describe the design that is meant to take it. Section 10 lists where this changes
arch-doc decisions.

---

## 1. Verdict

- **Feasible for 2.5D and 3-axis roughing.** Engagement-controlled clearing against a
  simulated stock is known to work. It ships in open source and in several commercial
  engines, so the basic control loop is not a research gamble.
- **The field representation removes whole failure classes**: no Z-slicing of B-rep/mesh
  into loops, no offset-loop self-intersection repair, no island/topology bookkeeping, and
  imperfect input geometry degrades gracefully instead of failing.
- **Where we can do better than the incumbents:** they are either open-loop (offset CAM
  plans from geometry and never measures) or greedy (adaptive engines decide one step at a
  time from local state). The design here does both: a **global clearing plan expressed as
  a field**, tracked by a **local controller that measures tool load**. The plan makes the
  order of removal, the link sites, and the final stock designed quantities instead of
  accidents. It also makes the path depend continuously on the geometry, which greedy
  walkers cannot offer.
- **Brittleness does not disappear; it moves.** It moves to controller stability,
  termination, residue slivers, linking, output point density, and the design of the plan
  field itself (section 3). All but the last are engineering problems with known
  mitigations. The plan field is the research core.
- **Effort (estimates):** a single-region 2.5D spike with a harmonic plan field in 4-6
  weeks; an optimized plan field another 1-3 months of research; a 3-axis rougher a shop
  would trust in production in 12-24 person-months total, most of it in linking, lead-ins,
  edge cases, output quality, and validation.
- **Top risks:** (1) generation time against a budget of seconds, (2) path quality as the
  machine controller actually executes it, (3) proving the result beats the incumbents on
  real failure cases, not on synthetic parts.
- **Recommendation:** start with a failure census (section 9, step 0). It is the test set
  every later step is judged against.

---

## 2. Where offset roughing actually breaks

"Offset roughing is brittle" covers several distinct failure classes. They have different
causes, and a replacement has to be judged against each one separately.

| # | Stage | Failure | Root cause |
|---|---|---|---|
| 1 | Section extraction | Slice loops do not close, open chains, slivers, wrong inside/outside | Slicing B-rep or mesh at a Z level: faces coincident with or tangent to the slice plane, gaps between surface patches, tolerance mismatch between kernel and CAM |
| 2 | Offsetting | Self-intersecting or invalid loops, dropped loops, crashes | Offset of arc/spline chains with global self-intersection trimming; near-degenerate loops |
| 3 | Topology events | Uncut cusps between rings, missing island passes, extra cleanup passes | Rings split, merge, and vanish between successive offsets; islands appear and disappear with depth |
| 4 | Engagement | Load spikes at concave corners and where rings collapse; tool breakage | Offset spacing is an open-loop proxy for engagement, and it is wrong at corners |
| 5 | Linking | Plunges into material, excess retracts, poor ordering | Connecting rings and regions is combinatorial and done after the fact |
| 6 | Sensitivity | A small change in tool size or stepover changes the result a lot or makes the op fail | Topology of the offset family is discontinuous in its parameters |

Note on class 2: integer polygon clipping (Clipper / Clipper2) made pure polygon offsetting
robust years ago. In practice much of the remaining brittleness is upstream (class 1: getting
clean loops out of the model) and downstream (classes 3-5). A replacement that only fixes
offsetting fixes the least of it.

---

## 3. What a field-based generator changes, class by class

| # | Field-based replacement | Does the failure go away? | Residual risk |
|---|---|---|---|
| 1 | The slab is a query of the part field at height z. There are no loops to extract or close. | **Yes.** | Inside/outside of bad input. Mitigated by conservative rasterization and winding-number sign (4.2). |
| 2 | Tool-center clearance is a distance query, `clear(c) >= r + delta`. No offset curve is built. | **Yes**, inside the generator. | Output still needs line/arc fitting of a point stream, which is well-conditioned. |
| 3 | The plan field fixes removal order; any residue left over is re-seeded from connected components. The generator never tracks loops. | **Yes, by design** (4.5: no stranded material). | Residue from controller deviation; a sliver threshold is still needed (4.7). |
| 4 | Tool load is measured every step and servoed, with feed co-scheduled. | **Yes**, by construction. | Controller oscillation; stuck states in narrow necks (4.6). |
| 5 | Link sites are the plan field's saddle points, known before any walking (4.5). Link moves are checked against the stock field. | **Reduced.** Link count and placement are planned; link ordering between regions stays combinatorial. | Ordering across many regions. |
| 6 | The plan is the solution of an elliptic boundary-value problem, which moves continuously with its boundary. | **Mostly.** A 0.1 mm wall edit moves the path by about that much instead of rerouting it. | Jumps remain where the plan's saddle structure changes. Greedy walkers are worse here than offset CAM; a planned field is better than both. |

---

## 4. Design for the roughing path

### 4.1 "Implicit" is the interface, not the storage

The robustness comes from representing regions as **functions you query** instead of
**boundary curves you construct**. Whether a given function is analytic, exact-to-mesh, or
sampled on a grid is a precision and speed setting, chosen per consumer:

- Roughing tolerance (stock-to-leave around 0.25-1 mm) is far above tessellation chord error
  (around 0.01 mm) and above a 0.1-0.25 mm grid. Sampled fields are sufficient here, and they
  are what makes a budget of seconds reachable.
- Analytic evaluation matters at finishing tolerance and for 5-axis grazing contact (the
  arch doc's grazing-contact argument in Part V). That is a Phase 2+ concern.

### 4.2 The part: a z-buffer and a 2D distance transform

**For a vertical tool, 3-axis roughing never needs 3D distance.** The tool only sees the
part's upper envelope `Ptop(x,y)` (highest part Z over each XY point). Two operations cover
everything:

1. **Conservative max-Z rasterization of the tessellated part** into a height grid (a
   z-buffer taken from above). Every triangle writes its maximum Z over every cell it
   touches. This has no degenerate cases (no ray-edge hits), and cracks between face
   tessellations that are smaller than a cell are covered by their neighbors. Cost: well
   under a second for 100k-500k triangles.
2. **Per slab at height z:** the silhouette `S_z = { Ptop > z - delta_axial }` is a binary
   image. An exact Euclidean distance transform (Felzenszwalb-Huttenlocher, linear time)
   gives `clear_z(x,y)`, the XY distance to the silhouette. The tool center is gouge-free iff
   `clear_z(c) >= r + delta_radial + h`, where `h` (one cell diagonal) makes the raster error
   conservative.

Tool shape: for the gouge check, treat any end mill as its enclosing flat cylinder. A bull
or ball nose occupies less volume than a flat end mill of the same radius, so this is
conservative. It leaves extra stock on slopes, which the shallow-area pass (arch doc
Phase 1) cleans up. Use the true profile for engagement, not for clearance.

The distance transform is exact Euclidean distance to the raster set, so the arch doc's
Euclidean-distance crux (its section 5) does not arise on this path.

**Input that is broken versus input that is merely imperfect.** Small cracks and T-junctions
are absorbed. A missing face is not: the z-buffer would see the surface below it, and the
tool would gouge. Detect open-edge length before running and refuse loudly. Rule: **degrade
gracefully on imperfect input; fail loudly on broken input.**

Where true 3D part distance is needed (tilted tools later, or checks), use exact distance to
the mesh through a BVH, with the sign taken from the generalized winding number
(Jacobson et al. 2013). That sign is robust to cracks and non-manifold patches, and the
distance is exact to the tessellation, so there is again no crux, because nothing is
composed with min/max.

### 4.3 The stock: a sampled cleared-region distance field

A binary "cut / not cut" raster is **too coarse to measure engagement**. With a 12 mm tool
at 10% radial engagement and a 0.25 mm step, one step removes about 0.3 mm^2, which is about
5 cells at 0.25 mm. Per-step area measured that way is noise.

Instead, store the **cleared region's distance field** per slab on the grid:

```
C(x) = min over swept moves of ( dist(x, move path) - r )      cleared where C < 0
```

- The swept region of each linear or arc move of a flat tool in a slab is a capsule or an
  annular sector, and its SDF is exact and closed-form.
- The union (min) of exact SDFs is **exact outside the union**, which is the material side,
  the only side the controller reads. So the stock boundary is resolved below cell size by
  bilinear interpolation of `C` (error is about h^2 / (8 R) on smooth boundaries and about h
  at creases where two sweeps meet).
- Updating after a step writes only the cells within `r + band` of the new segment, about
  100 cells per step. Query cost does not grow with path history. A polygon cleared-area
  model does grow with path length, and so does an exact min-over-all-sweeps evaluator
  without spatial culling.
- For 3-axis, the per-slab fields come from one shared stock height field `H(x,y)`: in slab
  k, material exists where `H > z_k`. Rest roughing with a second, smaller tool is free: the
  stock left after tool 1 is the starting stock for tool 2.

This is the same closed-form swept-tool math as the implicit toolpath viewer's exact stock
(`H(x,y) = min over moves`), sampled into a grid so the inner loop stays constant-time.
Section 4.9 explains why the two must remain separate code.

### 4.4 What the incumbents leave on the table

These are the targets. Each one is a property of a design family, not a claim about any one
vendor's internals.

1. **No global plan.** A greedy walker decides each step from local state. Removal order,
   where passes end, how many residue fragments are left, and where links go are all
   discovered along the way, not designed. That shows up as extra re-entries, links, and
   air time.
2. **A proxy objective.** "Constant engagement" stands in for "constant tool load". The
   actual goal is **minimum achieved machining time under a tool-load limit**. Holding
   engagement constant while the machine slows on tight curves gives that goal away.
   Machine dynamics usually enter only at the post-processor.
3. **Chaotic sensitivity.** Step-by-step integration can reroute the whole path after a
   tiny geometry edit. That hurts repeatability and shop trust.
4. **Uneven hand-off to the finisher.** The last material near the walls is whatever the
   walk happened to leave, plus an optional cleanup pass. Nothing designs it.

### 4.5 Plan globally: the clearing-schedule field

The plan is one scalar field per slab over the removable region:

```
T(x) in [0, 1]   = the schedule time at which point x is cleared
cleared at time t = { T < t }        front at time t = the level set { T = t }
```

**Requirements on T** (each one checkable on the grid):

1. **Starts at the entries:** `T = 0` on the entry helix disks. Seeds go at maxima of the
   region's distance transform, so the widest helix fits.
2. **Finishes at the walls together:** `T = 1` on the tool-center-reachable boundary and on
   every island wall. The last front follows the part, so the final loop is a
   wall-following pass that leaves **uniform stock for the finisher**, by construction.
3. **Smooth fronts:** front curvature stays below `1 / r_turn` wherever the tool must follow
   a front, so the path can be driven at feed.
4. **Spacing matches load:** the normal spacing between successive fronts tracks the
   load-optimal stepover field `s*(x)` (section 4.6).
5. **No stranded material:** no region is cleared last while surrounded by cleared area.
   An uncut island floating in cleared space is an interior local maximum of `T`.

**The first formulation: a harmonic plan.** Solve Laplace's equation with `T = 0` on the
seeds and `T = 1` on the walls, by multigrid on a coarse (0.5-1 mm) grid, in well under a
second per slab. This satisfies requirements 1, 2, and 5 exactly, and requirement 3 mostly:

- **Requirement 5 holds automatically.** By the maximum principle, a harmonic function has
  no interior local maxima or minima, so the plan cannot strand material.
- **Its only interior critical points are saddles.** A saddle is where one front pinches
  into two, for example behind an island. Saddles are few, isolated, and found before any
  walking. **They are exactly the planned split points, which is where links go.** Linking
  becomes a quantity the plan designs rather than something the walk discovers.
- **Stability:** the solution of an elliptic boundary-value problem depends continuously on
  its boundary. A small wall edit moves the fronts slightly instead of rerouting the path.
  Jumps remain only where the saddle structure changes.
- **Islands and multiple seeds need no special cases.** They are just more boundary. This
  makes the arch doc's topology-robustness claim concrete.

**What the harmonic plan does not give: requirement 4.** No single field makes the fronts
both conform to the walls and stay evenly spaced. Fronts crowd entering corners and narrow
passages and spread out in wide areas. The plan therefore fixes the **order and shape** of
removal, and the controller (4.6) fills each band between planned fronts at the local
target stepover. It adds partial loops where a band is wide and merges passes where it is
narrow. The plan's band-width field says where those partial loops go before the walk
starts.

**Stronger formulations (the research core):**

- **Designed conductivity:** solve `div(k grad T) = 0` and tune `k(x) > 0` iteratively to
  even out spacing. This is still elliptic, so the maximum principle, and with it
  requirement 5 and the saddle-only structure, still hold.
- **Optimized plan:** minimize spacing error against `s*(x)`, plus front curvature, plus
  estimated machining time, subject to requirements 1 and 2. Solve by Gauss-Newton with
  multigrid. The maximum principle is no longer automatic here, so enforce requirement 5 as
  a constraint.

The spike (section 9) builds the harmonic plan first and measures how far requirement 4 is
missed on the census parts. The stronger formulations are built only if those gaps justify
them. The arch doc's single spiral around a medial spine (its section 8) is the special case
of one seed and no islands.

Literature check: level curves of an elliptic PDE as pocket toolpaths go back to Bieterman
and Sandstrom (2003). What is added here is the saddle-as-link-site reading, the use of the
field as a plan tracked under load feedback (not as the path itself), and the
load-driven spacing target. Include the PDE-pocketing lineage in the patent claims read
(arch doc section 10).

### 4.6 Execute locally: a tracking controller on measured tool load

*Superseded in part (Derek, 2026-10-23): step 2 below, steering the lateral offset step by
step until the measured engagement hits a target, is an incumbent method, ruled out by
decision 8 (`CONTRIBUTING.md`; `ARCHITECTURE_AND_ROADMAP.md` section 2) and by section 13.2.
The objective, `s*(x)`, stays and belongs in the plan (section 13.4). Measuring engagement and
co-scheduling feed from it (step 3) stay.*

**The objective is minimum achieved time under a tool-load limit.** Constant engagement is
only a means to that. With radial engagement `ae` on a tool of diameter `D` (for
`ae < D/2`), maximum chip thickness is

```
h_max = fz * 2 * sqrt( (ae/D) * (1 - ae/D) )        (radial chip thinning)
```

so feed per tooth can be raised to hold `h_max` at its target as `ae` drops. The
**load-optimal stepover** `s*(x)` is then a local optimization. It maximizes material
removal rate `ae * ap * vf` subject to the chip-thickness limit, spindle power, and **the
feed the machine can actually reach along the local path curvature**. On tight curves, where
the machine slows anyway, a larger `ae` at a lower feed wins. On open sweeps, a thinner cut
at full feed wins. `s*(x)` is a field, and it is what the plan's spacing requirement (4.5
item 4) targets.

Per step:

1. **Predict** the heading from the plan: the tangent of the current front of `T`, offset
   toward the band's target position.
2. **Correct** from measurement: sample the cleared-region field (4.3) around the cutter to
   get the true engaged arc and area. Adjust the lateral offset (a 1D bracketed root-find)
   so the measured `ae` hits `s*(x)`.
3. **Co-schedule feed** from the measured `ae` through the chip-thinning relation, so that
   where `ae` still deviates, chip thickness and tool load stay on target. Engagement
   error then costs time, not tool life.
4. **Hard constraints** on every candidate: tool-center clearance (4.2), a fixed cutting
   side (climb or conventional), a minimum turn radius, and no cutting through material on
   link moves.
5. **End of pass** at a planned saddle or wall; otherwise the walk continues. If no
   candidate within +/-90 deg of the prediction reaches the minimum engagement, the pass
   also ends, and the location is logged as a plan deviation.

The plan supplies direction and order, where fields are reliable. Measurement supplies how
far in to step and how fast to go, where only ground truth is reliable. A noisy plan
gradient costs root-find iterations. It does not cause a wrong cut.

**Termination invariant:** every committed step removes at least `A_min * step` of material,
or the pass ends. Residue area strictly decreases, and seeds are placed only in residue
components above the sliver threshold, so the generator terminates. Put this invariant in
the code as an assertion, not just in this doc.

### 4.7 Coverage and slivers

With the plan satisfying requirement 5, residue comes only from controller deviation. After
the walk, the residue is `removable AND not cleared`. Compute its connected components on
the grid. For each component, the maximum of its interior distance transform is its
thickness:

- thickness >= `t_seed`: re-seed there (a local plan, a local walk);
- thickness < `t_seed` and lying within the finish allowance: accept (the finisher removes
  it);
- otherwise: a cleanup pass along the component's ridge, or report it.

`t_seed` and the finish allowance are explicit parameters, so "no slivers left" becomes a
number that can be checked. Residue count per part is itself a quality metric for the plan.

### 4.8 Output and machine dynamics

- The walk emits a dense point stream (0.25-1 mm steps). Fit lines and arcs (biarcs) to a
  chord tolerance (for example 0.005-0.01 mm) before posting. Without this, block count
  starves the controller's lookahead.
- **Score paths by achieved cycle time, not programmed feed.** Use a jerk-limited kinematic
  model of the target machine (or logs from it). The same model supplies the
  curvature-to-feed limit inside `s*(x)` (4.6), so machine dynamics shape the path from the
  start instead of being discovered at the post.
- A curvature limit in the controller and a smoothing pass before fitting are the rougher's
  job, not the post-processor's.

### 4.9 Verification must be independent

The generator's in-loop stock drove the controller, so checking the result against that same
stock only shows the code agrees with itself. Verify the **posted G-code** with an
independent exact swept-stock evaluator (the implicit toolpath viewer effort's evaluator is
exactly this), which also covers post-processor errors. Keep that evaluator as separate code
from the generator's in-loop stock, even though the math is the same.

---

## 5. Runtime budget (estimates, to be confirmed by the spike)

Assumptions: 12 mm tool, 0.25 mm stock grid, 0.5-1 mm plan grid, 128 samples around the
cutter per engagement check, 8-12 candidates per step, 0.5 mm steady-state step, single
thread.

| Item | Cost |
|---|---|
| Z-buffer of the part (100k-500k triangles) | < 1 s, once |
| Distance transform per slab (750 x 750 mm region = 9M cells at 0.25 mm) | 0.2-0.5 s per slab |
| Harmonic plan per slab (0.5-2M cells, multigrid) | 0.2-0.5 s per slab |
| One engagement check (128 bilinear samples plus crossing refinement) | about 2 us |
| One step (predict, correct, co-schedule feed) | about 25 us |
| Stock update per step (about 100 cells) | < 1 us |
| **Walk, per meter of toolpath** | **about 50 ms** |

Rule of thumb: **compute scales with toolpath length.** At a 10 m/min cutting feed, that is
about 0.5 s of compute per minute of machining, single-threaded. Separate regions run in
parallel. For 3-axis, slabs within one region are sequential (upper slabs clear first), so a
deep 3-axis part with 30-50 slabs is the case most likely to exceed a budget of seconds.
Measure it early. The optimized plan (4.5) is the other unknown cost; it must stay inside
the budget or run only as an offline preset.

---

## 6. Benchmarks and prior art

These are benchmarks to beat and a record of what is known. They are not templates.

- **FreeCAD CAM Adaptive** (`src/Mod/CAM/libarea/Adaptive.cpp`, Kresimir Tusek, 2018,
  LGPL-2.1+). A greedy design: cleared area is kept as Clipper polygons, and each step
  searches heading to match the cut area to a target. **It is the open baseline**, runnable
  on the same parts, and useful as a measured reference for everything in section 4.4.
- **ModuleWorks** and the other commercial adaptive / HEM engines (Freesteel-derived
  adaptive clearing in the HSMWorks / Fusion lineage, VoluMill, iMachining, and others).
  Internals are proprietary. Benchmark on output (cycle time, load, links, residue), never
  on assumed internals. See arch doc section 10 for the patent posture.
- **Field and PDE pocketing.** Bieterman and Sandstrom (2003) take toolpaths as level curves
  of an elliptic PDE solution. Held and Spielberger (2009) build smooth spirals from the
  medial axis. Both use the field or skeleton as the path itself, and both find pockets with
  islands hard. Section 4.5 uses the field as a plan under load feedback instead, with
  islands handled through the saddle structure.
- **Z-map / height-field 3-axis machining.** A decades-old, robust alternative to offset
  methods. OpenCAMLib's drop-cutter, and its waterline built from sampled "fibers", both
  sample and then reconstruct instead of offsetting curves. Section 4.2 is in this tradition.

---

## 7. Host integration options (a commercial CAM host as the case at hand)

A rougher only replaces offset roughing if its output reaches the machine through the shop's
existing workflow.

| Option | How | Keeps host sim / post / setup sheets | Risk |
|---|---|---|---|
| A. Standalone G-code | Read the exported part, stock, and tool; emit G-code through our own post | No | Needs a per-machine post; splicing into host programs by hand |
| B. CL-record injection | Create a placeholder rough op in the host (tool, feeds, setup), then replace its toolpath through the host's API (delete the op's CL records, create new ones, append them; **not yet verified live**) | Yes | A host regenerate or rebuild probably overwrites injected records; out-of-process COM costs about 5-15 ms per call, so a 20k-record path must be written in-process (add-in) |
| C. Steer the host rougher | Use fields only to compute clean contain/avoid boundaries, or to flag regions where the host rougher will fail | Yes | Lowest value; does not fix engagement spikes (class 4) and cannot beat the incumbent |

Recommendation: probe Option B early (about 1 week, alongside the spike). If injected
records survive posting and simulation, it is the only option that replaces the rougher
without replacing the rest of the host workflow.

---

## 8. Risks

Top five, in order:

| Risk | Why it matters | Mitigation / gate |
|---|---|---|
| Generation time | A generator slower than the host's own tools will not be used | Section 5 budget; measure a deep 3-axis part in the spike |
| Achieved feed on the real controller | Load-optimal paths lose if the machine cannot drive them | Machine model inside `s*(x)` (4.6), arc fitting, curvature limit; score on achieved cycle time |
| Harmonic plan misses spacing badly | Too many partial loops erode the plan's advantage | Measure on census parts; escalate to designed conductivity or the optimized plan only if needed |
| Beating the incumbents on real failures | Synthetic wins do not justify switching | Failure census as the test set (section 9) |
| Controller instability / stuck walks | Chattering headings, loops, zero progress in necks | Plan-guided prediction, termination invariant, per-pass iteration caps |

Later: patent claim scope before any production or public use (arch doc section 10; not legal
advice). Workholding interactions on sheet parts (tabs, onion skin), which roughing order
affects. Input geometry that is broken rather than imperfect (4.2).

---

## 9. Recommended plan, with gates

| Step | What | Estimate | Gate to continue |
|---|---|---|---|
| 0 | **Failure census**: collect 10-20 real cases where offset roughing failed or misbehaved; classify each by the section 2 table | 1 week (depends on collecting cases) | The census is the test set for every later step |
| 1 | **Engagement and load analyzer**: measure engagement, chip thickness, and air time along existing host G-code with the viewer's exact stock (stock "before move i" = min over sweeps with move index < i) | 1-2 weeks | Quantifies classes 3-5 on census parts and on FreeCAD Adaptive output; becomes the measuring instrument for everything after |
| 2 | **2.5D single-region spike**: z-buffer and distance-transform part (4.2), sampled stock (4.3), harmonic plan (4.5), load-tracking controller (4.6), residue re-seed (4.7), biarc output (4.8) | 4-6 weeks | Metrics below, against the host and FreeCAD Adaptive on the census parts |
| 3 | **Host injection probe** (Option B, section 7) | 1 week, in parallel with step 2 | Injected records survive post and sim |
| 4 | **Stronger plan** (designed conductivity, then optimized) | 1-3 months, research | Only if step 2 shows the harmonic plan's spacing gaps cost real cycle time |
| 5 | 3-axis slab stack, shared `H(x,y)`, rest roughing, shallow-area pass | 2-4 months | Same metrics on freeform census parts |
| 6 | Production hardening: linking between regions, lead-ins/outs, edge cases, UI, docs | 6-12 months | Shop sign-off on real jobs |

**Spike acceptance metrics.** Each is scored against both the host's offset rough and
FreeCAD Adaptive on the same part and tool:

1. Zero gouges against the part at shop tolerance, verified on the posted G-code by the
   independent evaluator (4.9).
2. Chip thickness within +/-10% of target at the 95th percentile, outside entry and exit
   transients; no full-slot events except planned slotting.
3. Every residue component is thinner than the finish allowance, or is reported. Fewer
   residue components and re-entries than both benchmarks.
4. Finish-stock uniformity along the walls: a lower spread than both benchmarks.
5. Stability: move one wall by 0.1 mm and measure how far the path moves. The target is
   about 0.1 mm away from saddle changes.
6. Estimated achieved cycle time, including links, below both benchmarks, with generation
   time within the section 5 budget.
7. Produces a valid path on every census part where the host failed.

Matching the benchmarks on metrics 1-3 is the floor. Winning on metrics 3-6 is the reason
the project exists.

---

## 10. Deltas this suggests to ARCHITECTURE_AND_ROADMAP.md (not yet applied)

1. **Section 2 ("Not a voxel/mesh CAM")**: add a roughing exception. At roughing tolerance,
   sampled fields derived exactly and conservatively from the part (z-buffer, distance
   transform, mesh SDF) are the right tool. The ban should target meshing the *stock* and
   caching the part for *finishing*.
2. **Sections 6 and 9 (stock)**: for 3-axis, a single-direction dexel (height field) is exact
   and sufficient. Tri-dexel or VDB is needed only once the tool tilts or undercuts exist.
3. **Section 8 (the walk)**: replace the wall-tangent plus inward blend with a clearing-schedule
   plan field (4.5) tracked by a load-feedback controller (4.6). Replace "constant
   engagement" with "minimum achieved time under a load limit" as the objective. Add the
   termination invariant. The spiral around a medial spine stays as the one-seed,
   no-island special case.
4. **Appendix A (B-rep/STEP ingestion parked)**: for the roughing-replacement goal,
   ingestion is on the critical path, not a side track. Tessellate, then z-buffer and
   winding-number SDF (4.2).
5. **Phase 0 gate**: add the runtime budget (section 5) and the failure census (section 9)
   as entry criteria, and name the benchmarks (host, FreeCAD Adaptive) with the section 9
   metrics.
6. **Section 9 (stack)**: the roughing path does not need Fidget. Fidget's value (JIT
   interval evaluation of analytic trees) arrives with constructive parts and finishing.

---

## 11. Open decisions (for the project owner)

- **Spike language/runtime.** The arch doc picks Rust. The roughing path does not depend on
  Fidget, and the existing viewer and stock evaluator are in C#. Recommendation: write the
  spike in the language that gets to the acceptance metrics fastest, and keep the generator's
  stock separate code from the verifier's (4.9) whichever language is chosen.
- **Whether the plan field is exposed to users** (for example, choosing seeds, or painting
  regions where the plan should finish first) or kept internal. Recommendation: internal
  for the spike; revisit once the metrics show it working.

---

## 12. First spike results (2026-09-28)

A first spike ran on a production nest: 3/4 in HDPE sheet on a 3-axis router,
a 1/2 in 2-flute end mill, single-level pockets up to 16 mm deep. The host
CAM program is an auto-generated baseline (an inside-out offset rough: a
full-depth slot on the medial line, then 40% stepover rings), not tooled for
production. It is used for scale and
for its feature recognition (which pockets, at which cut level), never as a
quality target: the checker gate and the allowance are absolute. The spike
re-roughs the same pockets, posts into the same program, and checks both
programs with an independent exact swept-stock evaluator.

**What was built (5-7 s to generate five pockets; about 32 s including posting,
checking and the report):**

- Part: the exact XY distance to the part's shadow above the cut level
  (triangles clipped and projected, bucketed, evaluated exactly; no raster, no
  slicing), with faceted cylinders refined back onto their exact surfaces.
  This supersedes the raster recommendation in sections 4.1-4.3: direct exact
  evaluation was fast enough.
- Stock: the union of swept tool capsules, exact and incremental.
- Plan: T on a 0.5 mm grid (the only discretized object), walls and entry at
  sub-cell accuracy from the exact fields.
- Seeds: one per significant maximum of the exact wall-distance field (by
  persistence), each with its own ridge segment, helix and trochoid.
- Path: loops at each level are keyed by what they enclose (seeds: outer
  fronts; islands: holes). Each key across levels is a branch, cut as one
  morphing spiral whose revolutions are carried between planned fronts along
  the gradient flow. Seed branches first, merged branches after their parents,
  then island rings, each ending on its wall; branches are joined by retract,
  rapid and a plunge on a front already cut.
- Gate: an op replaces the baseline's only if the checker finds no gouge and at
  most 5 mm2 of the pocket's reachable stock left uncut.

**Results on the five pockets generated:** all passed the gate on the first try, with no
gouges, at most 0.2 mm2 of reachable stock left uncut, peak load 1.04-1.09 times the
reference cut and engagement 112-130 deg at the 95th percentile.

The reference cut is a unit, not a target: the program's own feed at the
baseline's median engagement.

**Findings that change the design:**

1. **Seed on the ridge of the exact wall-distance field, not at a point.** A
   point seed in an elongated pocket makes the far end set the revolution count
   while the near walls get crowded fronts. Seeding on the distance field's top
   level set (its ridge) makes the fronts implicit offsets of the wall: 13
   revolutions dropped to 6 on the deepest pocket. The ridge is opened by a
   trochoid, not a slot.
2. **A harmonic plan lags into every convex corner** (section 4.5). Measured
   gradient at the corners about 0.0006 against 0.025 elsewhere; the last 8 of
   13 revolutions only advanced the corners.
3. **The distance-ratio plan strands material around islands.** It has no
   maximum principle; around three bosses it produced interior maxima of T,
   rings of material no front reaches (861 mm2 left uncut).
4. **Designed conductivity restores the guarantee.** div(k grad T) = 0 with
   k = d_seed + d_wall reproduces the ratio field's even spacing along straight
   flow lines and is elliptic: 0 interior maxima on every pocket. With a single
   seed it still plateaus near T = 1 far from the seed; on the island pocket
   the flow-back then left the free area and the path gouged.
5. **Islands: seed every room of the distance field.** One seed per
   significant maximum of the wall distance (persistence at least 1/8 of the
   tool radius) keeps every point near a seed, so the plan never plateaus.
   Fronts from neighboring seeds merge at the ridge minima (the saddles of the
   distance field) and holes are born behind islands, so links go exactly
   where the field's topology changes. On the three-boss pocket this gave 6
   seeds and 11 branches, and it passed: 0 gouges, 0 mm2 uncut. At a persistence of half the tool radius, two small
   "ears" of that pocket got no seed; each became a flat plateau of T that
   one front jumped, leaving 26 mm2 uncut. The threshold matters.
6. **Better still: drive the path by the cut, not the profile ("max-cut").**
   Every plan above bends its fronts so they reach every wall together, and
   that one constraint is what crowds fronts in some places, stretches them in
   others, and forces a seed per room. Dropping it: T = the geodesic distance
   from one entry per connected region (fast marching), fronts at exactly one
   stepover everywhere, and a front that reaches a wall simply becomes an open
   arc. While fronts are closed they form one morphing spiral; after that the
   arcs are scheduled lobe by lobe (a piece is ready once the pieces of the
   level before that touch it are cut; the nearest ready piece goes next) and
   cut zigzag, with a climb-only option. The profile only shapes the last pass,
   one loop per wall. Islands need no special handling: the fronts go around
   them. On the five pockets it was faster than the morphing plan, and the
   three-boss pocket needed one entry instead of six. All five passed the gate
   first time; engagement p95 127-146 deg, peak load 1.04-1.15. Stepover sweep:
   time fell from 7.5 mm to 9.5 mm (the 120 deg cap) and stayed flat to 11 mm.
   The reference load binds past the cap:
   a wider cut only means a slower feed at the same removal rate, so the next
   lever for "maximize cut" is the load reference itself, not the path shape.
7. **Plans need the unsaturated distance.** A band-limited clearance field
   (exact only near the part) silently distorts any plan that reads distance
   magnitude in the pocket's interior.
8. **Under a load limit tied to a fixed reference cut, time is roughly area /
   reference removal rate for any compliant path.** Offset rings already run at
   that rate, so no path can win much on time here (the morphing plan lost its
   time mostly to helix entries, trochoids and links); a faster posted time comes
   only from running above the reference load, as full-depth slots do.
   The lever is load quality, unless the reference cut is set below what the
   machine and tool really allow.
9. **The checker gate is not optional.** It caught every broken intermediate
   version (the island gouges, the plateau leftovers); the posted program only
   ever contains verified ops.
10. **Open pockets need a scope for "uncut".** Where part pieces leave gaps to
   the open sheet, stock in the gap mouths is equally reachable from outside;
   counting it against the pocket made a clean path look like 72 mm2 of
   leftovers. Stock also reachable from free space outside the pocket belongs to
   whatever cuts the outside.

**Also found:** on three pockets the cut levels taken from the program did not
match the design part, which is solid there; there is nothing there to rough.

**Next:** cut levels from the design part instead of the program's feature
recognition; a time model calibrated on the real machine; fewer links (join
neighboring branches at the tool's level where the stock between is already
cut); the reference load itself (the lever left for max-cut); and the failure census
(section 9, step 0), which this spike skipped.


**Follow-up 2026-09-29: physical limits instead of the reference cut.** The
reference cut was a unit, not physics, so finding 8 could only say "time is
area / reference rate". The spike now also runs under a physical limit set
for aluminum 6061-T6, used as a proving material because its cut data is the
most published there is (router vendor chip loads, 1,492 HSC rows with ae and
ap, handbook unit power): chip ceiling with radial thinning, a rubbing floor,
spindle power (Kienzle), tool deflection from the force integrated along the
helical flutes on a two-section cantilever, and the machine feed cap. Both
paths are fed at the fastest feed inside every limit. 1/2 in 2-flute, 2.0 in
hangout, 18,000 rpm, 13.6 hp gantry router:

1. **Deflection binds, not chip load or power.** Past about 12% stepover the
   0.001 in deflection limit sets the feed; steady removal is nearly flat from
   10% to 50% stepover and highest in a full slot, because with 2 flutes the
   peak force stops growing past about 120 deg of engagement. A fixed "narrow
   stepover" rule is not automatically right; the stepover has to come from
   the binding limit.
2. **The path concept holds on cutting time under the same limits**, but its
   links (rapids and entries) lose the gain back. Under physical limits,
   linking is the next lever, ahead of path shape.
3. **Acceleration is a load-quality problem, not only a time problem.** Light
   engagement earns high feeds the machine cannot hold through turns: 26-55%
   of the new path's cut length runs below its ceiling, and
   100-160 mm per pocket falls under the chip floor there. Harmless in
   aluminum; in a polymer with a melt floor that is exactly where it melts.
   The planner has to budget acceleration, not only engagement.
4. **Tool setup outweighs path design.** 1.25 in hangout (or a 0.002 in
   limit) cuts the time by about 38% and moves the binding limit to
   spindle power.

The dominant inputs, in order, are the deflection limit, the machine's
acceleration and its rapid rate; none needs cutting time to pin.


**Follow-up 2026-09-29 (night): the stepover as a physics field.** Fronts from
fast marching with a speed field, so they are s*(x) apart; s*(x) = the widest
stepover whose engagement stays stable against chatter (Altintas-Budak
zeroth-order, lobe-independent) at the local axial depth left by earlier ops,
on a front of the local curvature; then measure the generated path's
engagement and shrink s* where it ran past the limit (three rounds).

1. **Chatter is the wide-cut penalty the time objective was missing.** Without
   it every limit rewards wider cuts and the objective collapses to "as wide as
   allowed". With it the optimum is interior and depth-dependent, and full
   depth with a thin stepover beats splitting the depth (two to four levels
   raise the time bound 2.3-2.5x): high-efficiency milling falls out of the
   physics rather than being assumed.
2. **Stability costs about 2x on these pockets** against a constant stepover
   that is stable at the median depth but chatters on 9-67% of its length in
   the deep, complex pockets. Local corrections propagate through the fronts
   that follow.
3. **At stable stepovers, execution dominates.** Chip-thinned feeds of
   900-1000 ipm are unreachable on short arcs under gantry acceleration (45-100%
   of the cut on the acceleration limit), and thin passes cross oblique walls
   too far apart to step over, so links multiply. Pass topology and linking,
   judged under the machine's real acceleration, are the next design problem.


**Follow-up 2026-09-29: continuity, tested.** Section 4.4 item 3 claims fields
avoid the incumbents' chaotic sensitivity. That is now a gate: rerun with a
5 um wall change (and a 0.1% physics-constant change) and require that at most
5% of the posted path moves more than 1 mm (1 mm sampling, each point matched
within +/-10% of cutting order, so a pass that appears at a threshold is local
but a reorganized order is not). The first physics-stepover version failed it
(52-54% of one pocket's path moved). Two design errors caused it, and either
alone was survivable: feeding the plan's own front curvature (a second
derivative of the field being solved) back into its speed, iterated; and a
greedy "nearest ready piece" order over hundreds of thin arcs, where near-ties
are everywhere. The fixes, now defaults: the stepover field comes from fixed
geometry only (the curvature of fronts from a round entry, 1 / (T_0 + rho)),
smoothed over half a tool diameter and capped at the stable value; the level
unit is fixed; the open-phase order is read from the field (depth first over
the level-dependency graph with positional keys, direction by level parity).
All five pockets then pass with 0-0.3% moved, and the plan got faster (48% of
the time bound instead of 33%). Lesson for the design: every decision layered
on a field must be a continuous function of it, or the field's stability
buys nothing.


---

## 13. Field-native priorities (2026-09-29)

After the spikes, most of what was built computed what curve CAM computes
(offsets from an entry, a feed schedule on a finished path) by way of fields.
That buys robustness, not results. From here, generator work is ranked by how
much it uses what only fields can do, and it has to be measurably different
from the incumbents, not a re-implementation of them in another representation.

### 13.1 What fields give that curve-based CAM cannot

1. **Queries anywhere, before cutting:** distance to the part, stock height,
   channel width, how enclosed a cut is, wall thickness. An objective can be
   scored at every point.
2. **Topology for free:** fronts merge and split without curve surgery. The
   plan's saddles are where passes split, so they are designed link sites.
3. **The plan is solved, not constructed:** an eikonal with a speed field, or
   an elliptic plan with designed conductivity.
4. **Gradients:** the plan and the stock are differentiable fields, so the
   time a plan costs can be descended, not searched.
5. **State over time:** stock, enclosure, heat and hold-down change with the
   cutting order T, and can be computed before any path exists.
6. **Continuity:** a well-posed field problem depends continuously on its data,
   but only if every decision built on the field does too.

Curve-based CAM builds the geometry first and assigns feeds afterward; feed
optimizers tune a finished path. Neither lets the physics reshape the path.

### 13.2 The field-native test (every generator change)

1. The physics shapes the path, not only its feed.
2. Every decision is a continuous function of fields: no argmax on a flat
   curve, no threshold that switches structure, no greedy "nearest" choice, no
   feedback of a field's own derivatives without regularization.
3. Engagement, links and order are read from the plan (spacing, curvature,
   saddles, T), not discovered by walking.
4. Gates stay: gouges (exact stock on the posted program) and continuity
   (5 um wall / 0.1% physics perturbation: at most 5% of the path moves more
   than 1 mm). Reported beside them: share of the cut past the stable depth,
   retracts, and time as a share of the physics bound. Scores are absolute,
   never relative to the host's own program.
5. Engineering polish that is not a field advantage (lead-ins, arc fitting,
   link smoothing) is copied from known-good practice, not reinvented.

### 13.3 Anti-patterns hit so far

| Pattern | Where | Result | Fix |
|---|---|---|---|
| Feed schedule on a fixed path | material model on max-cut | looked like feed control, path unchanged | physics enters the plan |
| Argmax on a flat curve | stepover knee, first objective | 12% vs 72% jumps; stepover flat at the cap | the missing limit (chatter); smooth optimum |
| A field's own derivatives fed back, iterated | stepover from the plan's curvature | 52% of one path moved by 5 um | geometry-only input, one solve |
| Greedy extraction | nearest-ready open arcs | near-ties amplified; 150-340 retracts | order from the field |
| Thresholded correction loop | shrink where measured unstable | oscillating fronts, not continuous | stability as a plan constraint |
| Pointwise guess instead of reading the field | front curvature 1 / (T_0 + rho) | 28% of one pocket past the stable depth | curvature of the smoothed plan |

### 13.4 Ranked field-native work

1. **Read engagement from the plan.** Spacing to the previous level and the
   curvature of the smoothed plan give the engaged angle everywhere before any
   path exists; stability becomes a constraint on the plan.
2. **The walls-together elliptic plan with physics in its conductivity.** Its
   fronts close on the walls, so passes never break against a wall; they split
   only at saddles, which are the designed links. Needs the spiral's gouge at
   sub-mm spacing fixed.
3. **One variational objective** (machine time under chip, power, deflection,
   chatter and acceleration, plus smoothness), solved by gradient descent;
   adjoints later. Continuous by construction, and it optimizes the real goal.
4. **A curvature bound in the plan** at the scale the machine can follow at
   feed (requirement 3 of 4.5), instead of discovering slowdowns afterward.
5. **Order-dependent physics as fields over T:** enclosure and heat (HDPE,
   with its chip-evacuation limit), vacuum hold-down on nested sheets.
6. **Per-point fields already built:** tool assignment (no gain on this nest,
   which has no narrow necks) and the finish-allowance field (39% faster
   finish at the same tolerance; next, a corner rest pass).

Later: conformal 3D layers for carving, and differentiable plan optimization
with the stepover or conductivity field as the design variable.

**Status 2026-09-29.** Item 1 done: curvature read from the smoothed plan;
the path's engagement then tracks the plan within about 10% (median 0.84-0.98
of the prediction), so the plan targets 88% of the stable angle, and the share
of the cut past the stable depth fell from 5-35% to 2-9%. Item 2 working: the
walls-together plan with the physics stepover passes the gouge check on all
five pockets (every point held on the exact clearance, every feed move checked),
has no retracts in four of five pockets and 14 links in the island pocket
(against 58-165), 1-4% past the stable depth, and zero measurable movement in
the continuity test. It is slower (29% of the time bound against 41% for the
fast-marching plan): every level steps by its tightest point, so passes crowd
where the walls are near, and the island pocket takes six entries. That is the
case for item 3: design the conductivity against the time objective.

---

## References

- Tusek, K. FreeCAD CAM Adaptive clearing, `src/Mod/CAM/libarea/Adaptive.cpp` (2018-), LGPL-2.1+.
- Bieterman, M. B., Sandstrom, D. R. "A Curvilinear Tool-Path Method for Pocket Machining."
  ASME J. Manufacturing Science and Engineering, 2003.
- Held, M., Spielberger, C. "A smooth spiral tool path for high speed machining of 2D
  pockets." Computer-Aided Design, 2009.
- Jacobson, A., Kavan, L., Sorkine-Hornung, O. "Robust Inside-Outside Segmentation using
  Generalized Winding Numbers." SIGGRAPH 2013.
- Felzenszwalb, P., Huttenlocher, D. "Distance Transforms of Sampled Functions." Theory of
  Computing, 2012.
- OpenCAMLib (drop-cutter, waterline): github.com/aewallin/opencamlib
- Clipper2 polygon clipping and offsetting: github.com/AngusJohnson/Clipper2
- See also the arch doc's Appendix C (Fidget, libfive, OpenVoronoi, OpenVDB, patents).
- The full reference set, with verified citations and one line on why each paper matters
  here, is `whitepaper-references/MANIFEST.md` (entries 42-60 and 67-68 are the pocketing
  and engagement papers this document leans on).
