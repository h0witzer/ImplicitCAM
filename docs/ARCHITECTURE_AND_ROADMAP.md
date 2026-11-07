# ImplicitCAM — Architecture & Roadmap

**An implicit-field CAM kernel for subtractive machining, built as one planner for simultaneous
5-axis.** Toolpaths are read off solved fields over exact geometry, not built by combinatorial
B-rep or mesh geometry. The planner plans in the tool's whole configuration space, a position
and a tool axis; 3-axis, 2.5D and 3+2 machining are that planner with the axis held.

Rewritten 2026-11-07 (Derek): "We want one planner that targets 5 axis and everything else should
fall out as subsets of our real tool." The previous version's target, "high-efficiency 2.5D
roughing first", produced a part model that only prismatic parts fit and a roughing pipeline built
on it; the record and the replacement are in the next section and in Part III.

See also: `EXACT_FIELD.md` (the part's field and the four guarantees), `ROUGHING_FEASIBILITY.md`
(the field-native test and its ranking, section 13), `SIM_VISUALIZATION_ENGINE.md` (the viewer),
`SPAN_IMPLICIT.md` (trim loops and the exact evaluator), `PROOFS.md` (the kernel's claims).
License: AGPL-3.0-only (see `LICENSE`); commercial licenses for closed products and services from
the copyright holder.

---

## One planner, built for simultaneous 5-axis

The planner's unknown is a path in the tool's configuration space,
`q(s) = (c(s), â(s)) ∈ ℝ³ × S²`: the tool tip's position and the tool's axis. A milling tool is a
solid of revolution, so roll about the axis removes nothing and is never a planning variable; a
full frame may be stored where something needs it (a feed reference, asymmetric tooling). Every
mode a shop knows is this one problem with the axis constrained:

| Mode | The axis | What the planner is given |
|---|---|---|
| Simultaneous 5-axis | free on `S²` along the path | nothing held |
| 3+2 (indexed) | constant over each region, one value per setup | the axis held per region |
| 3-axis | constant `(0,0,1)` | the axis held everywhere |
| 2.5D | constant `(0,0,1)` | the axis held; the part happens to have flat floors and vertical walls |

2.5D is not a mode. It is a 3-axis job on a part that happens to be prismatic, and the planner
must not know the difference.

**The rules** (Derek, 2026-11-07):

1. **Every query takes a pose.** No query assumes a fixed axis, a vertical tool, a slab, a flat
   floor or a vertical wall. The 3-axis caller passes the axis `(0,0,1)`.
2. **Every part is the exact field.** There is no part model for a class of parts (prismatic,
   2.5D, turned, sheet). Whatever the exact evaluator reads, the planner plans.
3. **A restriction is a value, never a code path.** Holding the axis means passing a constant
   axis field to the general code. A branch with its own algorithm for a restricted mode is not
   built, however simple the mode looks.
4. **The test for any change:** say how simultaneous 5-axis calls it. If it doesn't, the change
   is out.
5. **Restricted results come from the general problem.** Where a constant axis is best, the
   planner should choose one on its own; nobody picks a mode.

**What went wrong, for the record.** From its first version (2026-09-29) this roadmap said "2.5D
roughing first", and
its own risk register warned against contracts "written in a secretly-3-axis shape". The target
line won. The result was the prismatic part model: `icam-core`'s `frep.rs`, `prismatic.rs`,
`part.rs` and the reference queries in `query.rs`, a part made of horizontal faces whose section
boundaries are the STEP edge curves (which sit off the solid's surfaces by the file's tolerance,
so it fails guarantee 4) with fillets carried as solid over their footprint, and anything else
refused by name. The roughing pipeline was built on it: `icam-query`'s slab clearance,
`icam-rough`'s pocket finding and levels, `icam-strategy`'s plans solved on flat slab grids,
`icam-stock`'s one-slab in-process stock, `icam-gpu`'s contact kernel, and `icam-sim`'s column
views. Each was right for prismatic parts and none says how simultaneous 5-axis would call it.
Part III, milestone 0, replaces all of it. The design guard has reported `frep.rs` since
2026-10-25 and stays red until the replacement lands; nothing is grandfathered.

---

## Part I — Foundational Thesis & Decisions

This section is the explicit record of the reasoning behind the project. It exists so that
future decisions can be checked against *why* the architecture is shaped this way.

### 1. The core thesis

Conventional CAM generates toolpaths as **combinatorial geometry**: intersect trimmed NURBS
surfaces, offset boundary loops, trim and stitch curves, project points onto tessellations. That
is what makes B-rep CAM fragile (offset, blend and boolean failures), and it is the language every
commercial high-efficiency-machining patent is written in ("parallel offsets", "transition paths
bridging offsets", "morphing spirals matching boundaries").

ImplicitCAM generates toolpaths as **field processing**: the part, the tool, the stock and the
plan are fields over ℝ³ (and over the tool's configuration space), and paths *emerge* as the
solution of a PDE, the level sets of a solved field, or a trajectory through one. This is a
different *mechanism*, the source of both its robustness and its distance from the incumbents'
patents.

### 2. What we are explicitly NOT doing

- **Not building a planner, query or part model for a restricted mode** (Derek, 2026-11-07). See
  the rules above.
- **Not a voxel or mesh CAM.** The part is never cached to a uniform grid and meshed; the stock is
  never drawn from a sampled grid (voxels, dexels, height maps) (Derek, 2026-10-24). Sampling the
  exact fields into the sparse brick cache for the global field and for viewing is allowed;
  decisions are made on the exact values.
- **Not tessellating exact input, and not lowering splines** (decisions 5 and 6 in
  `CONTRIBUTING.md`).
- **Not using the incumbents' toolpath constructions as our template (Derek, 2026-10-23,
  revised 2026-10-26; decision 8 in `CONTRIBUTING.md`).** Engagement-controlled roughing as
  commercial CAM does it is a set of patented methods for a physics objective: keep tool load,
  chip thickness and engagement inside what the tool and machine can take. The patents protect
  the methods, not the objective; the objective is ours too, pursued through the plan field with
  chip thickness as the load measure and feed free in the plan (engagement is measured, never the
  target). Ruled out as methods, in any form including fallbacks, in-repo baselines, flags, tests
  and plans: offset or contour-parallel passes; engagement held by offsetting, trimming or
  searching boundary curves, offset rings or cleared-area polygons; engagement held by steering a
  walker step by step until the measured cut matches a target; spirals morphed or interpolated
  between boundary curves; transition or bridge moves built between offset passes; recovering a
  boundary curve from a field and then doing any of these. Results may converge; intent may not
  (an incumbent construction is never the stated target). Trochoids and field spirals were taken
  off the list on 2026-10-26 because prior art clears them; they stay clear of the live claims
  in section 10.
- **Not fixing a path after the fact.** Smoothing and corner blending on a finished cutting
  path are the incumbents' fixes. A path that is wrong is fixed in the plan field that produced
  it (`ROUGHING_FEASIBILITY.md` section 13). The links between cuts, lead-ins and the arc
  fitting a controller needs follow known practice (Derek, 2026-11-07). Feed is scheduled
  along the path from the cut model; the path never moves to hold a load.

### 3. Source material assessed

- **A Gemini deep-research report** ("Functional Implicit and SDF Kernels for Subtractive CAM",
  not kept in the repository): a literature map of F-rep, SDF, B-rep and voxel kernels, offsetting
  and medial-axis math, open CAM tools, the constant-engagement lineage, hybrid
  additive-subtractive work and implicit neural fields for multi-axis planning. It oversells
  "trivial, mathematically-guaranteed offsetting" (section 5).
- **YouTube: "I'm making a game engine based on dynamic SDFs"** (Mike Turitzin; the first version
  of this roadmap credited it to holger-kibur and concluded the engine meshes for rendering). Its
  transcript shows it renders from its brick cache (`SIM_VISUALIZATION_ENGINE.md`), which became
  the model for our viewer. See section 4.

### 4. Case study: the "dynamic SDF" game engine

The engine caches SDF evaluations to a sparse brick cache and carves by instancing CSG primitives.
Three warnings were drawn from it: carving by instancing makes evaluation cost grow with edit
history (the stock in ImplicitCAM is the box less every sweep, and a point reads only the sweeps
near it, found through a tree of their boxes); interpolated samples blend like clay (decisions read exact values, never the
cache's interpolation); and meshing brings back facets (nothing exact is tessellated). Its brick
cache and clipmaps are the model for `icam-sim`'s viewer (`SIM_VISUALIZATION_ENGINE.md`).

### 5. Exactness: what each field is, and where it is exact

- **The part** is its exact signed distance (decisions 2 and 3; `EXACT_FIELD.md`). Face interiors
  are exact; within the band beside an edge the file's declared tolerance is the accuracy line;
  answers the evaluator cannot decide are marked unsure, never guessed.
- **The tool** at any pose is an exact solid of revolution with an exact signed distance read in
  its meridian half-plane.
- **The stock** is the box less every sweep: `F = max(box, −min over sweeps)`. Each sweep is
  certified to a tolerance over its move; min and max of exact distances are a lower bound on the
  true distance, exact wherever one term is active. That is the remaining place the two-field
  discipline applies: the bound is the safe side for gouge and collision checks, and an exact
  value is computed where a precise boundary is needed.
- **Distances between solids** (the tool against the part, a move against the part) are the core
  queries of Part II. They are exact where the inputs are, and certified where they bound a whole
  move.

**The rule that survives from the earlier crux:** never read a bound as a distance where the
answer depends on its magnitude, and never decide structure by a threshold of our own (guarantee 3).

### 6. Stock vs part

- **Part:** the exact evaluator. Planning reads exact values through its GPU kernel; the sparse
  brick cache samples the evaluation for drawing and for coarse entries far from the surface
  (Derek, 2026-11-07), never for a decision near it.
- **Stock:** the box less the program's sweeps, for any machine (`icam-sweep`), applied at runtime
  as moves arrive, for planning and for checking alike. The check's independence is separate
  inputs, not separate code (Derek, 2026-11-07): it reads the posted program, never the planner's
  moves, against the exact part through the certified sweeps, and `icam-sweep` is itself tested
  against brute force. `icam-stock`'s `ImplicitStock` (a vertical-tool height function, so
  3-axis only) retires with the prismatic code. Neither is a voxel, dexel or height-map grid.

### 7. Why paths read off fields are robust

1. **Direction survives a bound.** The gradient of a 1-Lipschitz field points at the nearest wall
   even where its magnitude is a bound, so methods that read direction keep working on the stock's
   bound field.
2. **Arc length is measured, not read off a field.** Path length and time come from the path and
   the machine model, so field error never enters them.
3. **Load is a property of the plan.** Chip thickness, power, deflection and stability come from
   the cut model (`icam-motion`) at the tool's pose against the live stock, and they shape the plan
   field (its speed, conductivity or curvature bound). The path is never servoed step by step
   against a target (decision 8; `ROUGHING_FEASIBILITY.md` section 13.2); feed along the path is
   scheduled from the same model and may follow the load point by point (Derek, 2026-11-07).

### 8. The planner: one functional

Roughing is one constrained optimization over the tool's configuration path
`q(s) = (c(s), â(s)) ∈ ℝ³ × S²`:

```
minimize   T[q]         # machining time
  over     q(s) = (c(s), â(s)) ∈ ℝ³ × S²
subject to
  coverage    sweep{ cutting part of the tool along q } ⊇ {R < 0}     # everything removable is removed
  collision   non-cutting part of the tool along q clears P, S and fixtures
  load        chip thickness, power, deflection and stability within the cut model's limits
              (engagement is measured and feeds the chip model; it is never the target)
  kinematics  q reachable by the machine · joint limits · axis rate ≤ the rotaries' · singularity-clear
  finish      encoded by R = max(S, δ − P): nothing closer than δ to the part is removable
```

`P` is the part's field, `S` the stock's, `δ` the stock to leave. Every term is a query of Part II
with a gradient with respect to the pose. Finishing is the same planner with the objective changed
to surface contact and scallop (Part III, milestone 2): the same pose, the same queries, the same
axis assignment and machine model.

**The held-axis ladder.** The modes are the functional with the axis field constrained:

| Mode | Axis field over `{R<0}` | Leaves of `{R<0}` | Constraint |
|---|---|---|---|
| 3-axis | constant `(0,0,1)` | flat, horizontal | axis held |
| 3+2 | piecewise constant, one value per region | flat, one stack per region | axis held per region |
| Simultaneous 5-axis | continuous `â(s)` on `S²` | curved, following the axis field | none |

**Paths are read off plan fields on leaves.** Within `{R<0}` the planner solves plan fields (a
clearing schedule `T`, shaped by the cut model; `ROUGHING_FEASIBILITY.md` sections 12 and 13) on
the leaves of a foliation, and reads the path off them: fronts are level sets of `T`, order and
links come from the plan's own structure (spacing, curvature, saddles). Flat leaves are what a
held axis produces; they are the same solve on a surface that happens to be a plane. The solvers
are written for a leaf, not for a plane: a solver that only works on a flat grid is the 2.5D
mistake again.

**Mode selection disappears.** The planner emits whatever axis field the constraints and the
objective imply: constant over open regions, gently curving where a feature deepens, tilted only
where reach demands. The axis is as constant as it can be and as variable as it must be, and "3-axis
vs 3+2 vs simultaneous" becomes a description of what regions of one output look like.
Rigidity, overhang and predictability (why a machinist prefers a steady axis) enter the objective as
an axis-motion and stick-out penalty, or the optimizer will produce clever but fragile motion.

**Open: the research core** (milestone 4). The field framing makes this well posed and
differentiable; it does not make it free.
1. **Reachable smooth foliation with coverage.** Curved leaves must each be reachable by a
   collision-free axis (query 4), give an axis field smooth enough for the rotaries, and still tile
   `{R<0}`. The three pull against each other; the choice of foliation is unsolved in general.
2. **State-coupled sequencing.** Reach depends on how much stock is left, which depends on the
   order leaves are cut. The state is itself the stock field, updated by the same sweeps, so the
   search runs on a queryable field rather than a bespoke in-process model.

Verification of coverage and collision over the continuous motion rests on move clearance (query 3)
and the certified sweeps, not on checks at stations.

### 9. Kernel & stack decision

Current as of 2026-11-07; the dated entries below are the record of how it got here.

- **Rust workspace, GPU kernels in wgpu (WGSL).** The sweep kernels are straight-line WGSL
  generated for each tool and machine (`icam-sweep`); the part's kernel reads packed span records
  (`icam-core` `solid3/gpu`). Neither interprets an expression tape. WGSL has no double type, so kernels run single precision in local
  frames with rounding folded into the certification margin; a double-precision CPU reference of
  every query lives beside it, every kernel is tested against it, and borderline gouge verdicts are
  re-decided on the CPU.
- **GPU first** (Derek, 2026-10-24; decision 9 in `CONTRIBUTING.md`). Every loop over samples,
  cells, poses or pixels is considered for a GPU kernel first, and code that stays on the CPU says
  why. Formulations are chosen to parallelize (fast sweeping or the fast iterative method over fast
  marching, Jacobi or multigrid over Gauss-Seidel, per-point walk-on-spheres).
- **The part's field is the exact evaluator, and it has no holes** (Derek, 2026-10-27 and
  2026-11-07; decisions 2, 3 and 10). The goal was pure implicit CSG, with no trims and no
  intersection tests; it is blocked for fitted B-spline faces, and the evaluator over trim loops
  certified at import is the compromise. Every query answers: the import never refuses a face,
  cusp edges carry a sign rule, the search is bounded by construction, and no answer is "unsure"
  (milestone 0 closes the three cases that still are). Judged against the four guarantees before
  adoption (`EXACT_FIELD.md`).
- **Exact input is never tessellated; B-splines stay native** (decisions 5 and 6; Derek,
  2026-09-30: "Most CAM engines do the lazy thing and approximate as lines and arcs").
- **Machines are plug-ins.** A machine turns axis values into a tool pose (`icam-sweep`'s `Machine`
  and `Chain`: serial machines with rotary and linear axes in any order, head or table side). No
  kinematics is assumed anywhere (Derek, 2026-10-24).
- **Input hygiene is out of scope.** Leaky or inconsistent STEP shells go to a preprocessor ahead of
  the importer when they become a real problem.
- **Not used:** Fidget, libfive, OpenVDB and OpenVoronoi were assessed in the first version of this
  roadmap. No CSG tree is compiled (decision 2), the stock is exact from sweeps, and medial
  information comes from solved plan fields, so none of them is a dependency.

How the part's field got here:
- 2026-09-29 (Derek): F-rep is the only internal representation; B-rep is an import format.
- 2026-09-30: exact 3D solids (`curve3`, `surface3`, `solid3`) and native B-splines; the prismatic
  compiler `frep.rs` (now on the replacement list).
- 2026-10-02 (Derek): per-cell formulas reversed; the field must be one global function.
- 2026-10-25 (Derek): the glued field and the pure field rejected against the four guarantees.
- 2026-10-26 (Derek): the file's declared tolerance is the accuracy line at edges.
- 2026-10-27 (Derek): the exact evaluator over certified trim loops is the field. `frep3.rs` was
  deleted on 2026-10-30.
- 2026-11-07 (Derek): one planner for simultaneous 5-axis; every query takes a pose (decision
  11). The same day: holes are impossible (no refusals, no unsure answers; decision 3); the
  pure-CSG goal recorded as blocked, the evaluator as the compromise, decision 4 restated as sign
  from topology (decisions 4 and 10); planning reads exact values, the brick cache draws
  (decision 2); `icam-sweep` is the one stock, the check independent by its inputs; grids only
  solve plan fields; no Voronoi diagram in planning; feed follows the load, the path never does;
  links and lead-ins follow known practice; CONTRIBUTING.md is the record of the decisions and
  this section their history.

### 10. Patent posture (not legal advice)

The live patents near this work (high-efficiency roughing, toolpath linking, 5-axis slot
milling, feature recognition and tool assignment) were read claim by claim on 2026-10-26; the
maintainer keeps that reading. Infringement is read claim by claim against what the program
does; intent and how a path was derived do not enter, so a field method avoids a claim only by
missing an element of it. The rules that follow from the reading:

- **Toolpaths** (decision 8 in `CONTRIBUTING.md`): no maximum and minimum engagement angle is
  chosen for a path to vary between; chip thickness is the load measure, with feed free. No
  region is split into a spiral part and a trochoid-only remainder. Links run only through
  space already empty, and no region is cut, or ordered early, to host links. No Voronoi
  diagram anywhere in planning (Derek, 2026-11-07; wider than the claim, which recites a spiral
  traced around a wave over one); medial information and spirals come from solved fields. In 5-axis, flank positions
  never come from a slot's two walls, and until May 2032, no simultaneous 5-axis loops whose
  upper and lower patterns differ while stepping down a slot between two walls.
- **Trochoids and field spirals are allowed** (Derek, 2026-10-26). The engagement-limiting
  patents on circular passes have expired, and trochoidal milling predates them. Field spirals
  rest on Bieterman and Sandstrom (2003), PDE spirals whose stepover varies along each loop.
- **Feature recognition** (Derek, 2026-10-25): regions are found in the field after import,
  where the removal, reach A(x, d), fit and floor fields agree, with no B-rep feature
  templates; what a region means (a thread, a tolerance) comes from PMI. Regions come only from
  field values, and nothing groups faces or F-rep leaves by shared edges, in planning or in the
  UI. PMI is sampled into the field from the faces it names, never grouped by walking their
  edges. The import compiler reads edge convexity (decision 4) but never groups faces into
  features, finds a perimeter or highlights anything.
- **Tool assignment:** plan each tool's toolpaths from the stock the earlier tools' actual
  paths left, not in parallel from shares found by subtracting reach. Prior art: Yao, Gupta and
  Nau (2003, Computer-Aided Design 35(9)) compute the area each cutter can cut and search
  cutter sequences for the least time; Veeramani and Gau (2000, IIE Transactions 32(7)) choose
  multiple tool sizes for 2.5D pockets.
- **Open:** patent-database searches beyond the engines read so far. Run them before any
  public "patent-free" statement.
- **Strategy:** publishing openly *creates prior art*, pre-empting future fences around
  implicit CAM toolpaths. This is the highest-leverage move and aligns with the open ethos.

---

## Part II — Architecture

### The core queries

**Decided (Derek, 2026-11-07):** every strategy reads the part and the stock through queries that
take a pose (or a motion); the 3-axis and 3+2 callers pass a held axis. **The method below for
queries 1 and 2 is planned, not built or adopted.** Its verdict against the four guarantees comes
first, per decision 10:

- **Guarantee 1 (parallel, with a measured rate): open.** Each (pose, feature) pair is independent,
  the layout of the part's GPU kernel, but no rate has been measured. The known risk is cost:
  certifying the least tool distance over a B-spline face is new work, and a flat tool bottom over
  a near-flat face has a whole patch of near-equal minima, so the certificate bounds the least
  *value* by intervals instead of chasing one point.
- **Guarantee 2 (exact where it claims): met by construction, to be checked against brute
  force.** It is the distance between the tool's exact solid and the part's exact faces, with the
  file's tolerance at edges.
- **Guarantee 3 (no holes): met, with one inherited case.** Every pose gets an answer, and the
  part's sign settles a buried tool; the part's open sign case (`PROOFS.md`, piece 27829) carries
  over until it is fixed.
- **Guarantee 4 (no intersection or B-rep logic): met.** No two model surfaces are ever
  intersected; each face is measured against the tool's own analytic profile, the same class of
  solve as a face's nearest point, and edges and vertices answer for their own distance.

It is adopted only when guarantee 1 is measured. The per-feature work lives beside the exact
evaluator in `icam-core`, because faces, edges, vertices and loops are the evaluator's business
(decision 1); `icam-query` exposes the queries to the planner, which reads only their answers.
**Not yet built** except where noted.

1. **Tool–part distance** `d(q)`: the distance between the tool at pose `q` (cutter, shank and
   holder) and the part, negative when they overlap. Computed per part feature against the tool's
   own profile: each face reports the least tool distance over itself inside its certified loops,
   edges and vertices report their own, and the least wins (the same scatter the part's GPU kernel
   uses). A tool buried wholly inside the part is caught by the part's sign at one tool point. It
   returns a contact record: the part feature, the tool element, the two contact points, and the
   gradient of `d` with respect to the pose. Gouge, collision, reach and the pose optimizer all
   read this record.
2. **First contact along a direction:** how far the tool moves along a direction before `d`
   reaches the stock to leave, per feature, least wins. A 3-axis drop is this query along `−â` with
   `â = (0,0,1)`; approaches, retracts and plunges at any axis are the same query.
3. **Move clearance:** the least `d` over a move's whole duration, certified by branch and bound on
   the move's rate bounds, as `icam-sweep`'s `Sweep` already does for the stock. Gouge and
   collision between stations, for any machine. (Built for the stock side; not yet against the
   part.)
4. **Accessibility:** for a point of the part or the stock, the set of axes on `S²` from which the
   cutting part reaches it while the non-cutting part stays clear. Query 1 over axes on the GPU,
   boundaries refined exactly. This is the field over `ℝ³ × S²` of Part V, Advantage 3.
5. **Material at the tool:** the live stock inside the cutting part at a pose: chip thickness and
   engagement measured for the cut model, at any axis. It replaces the one-slab in-process stock.

**How the held-axis callers read them:**
- **3-axis levels and regions** come from drop heights (query 2 along `−z`). A level's region is
  where the drop height is at or below the level. Floors are where the drop height is flat: over a
  flat face it is the same closed form at every point, so the comparison is exact, with no
  threshold and no lookup of which faces are horizontal.
- **3+2** is the same calls with a tilted held axis, one per region, chosen from accessibility.
- **Simultaneous 5-axis** reads all five along the path, with the axis a variable.

### Layered design

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 6  OUTPUT                                                     │
│      Programs for any machine through its model; lines and arcs only │
│      where a controller forces them                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5  THE PLANNER                                                │
│      One functional over q(s) ∈ ℝ³ × S² · removal and surface        │
│      objectives · axis assignment (local or global) · sequencing     │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4  PLAN FIELDS AND MOTION                                     │
│      Plan solves on leaves · paths read off them · feed from the     │
│      cut model · the machine model consulted in the loop             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3  CORE QUERIES (pose-general)                                │
│      Tool–part distance · first contact · move clearance ·           │
│      accessibility · material at the tool                            │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2  STATE                                                      │
│      Part (exact evaluator, brick cache) · tools · poses · motions · │
│      machines · sweeps · stock · removable region R                  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1  KERNELS                                                    │
│      f64 CPU reference · straight-line WGSL per tool, machine, span  │
└─────────────────────────────────────────────────────────────────────┘
```

The machine model is not a Layer 6 service: Layers 4 and 5 consult it during pose selection and
feed scheduling (through-line 8); Layer 6 owns only the final program.

### Through-lines: built general from the first line of code

Each is written in its 5-axis form and is the only form built. A held axis is a value passed in.

1. **The tool is a solid assembly** (cutter, shank, holder) with the cutting part told apart from
   the non-cutting part. Never "just a radius". (`icam-core`'s `Tool` and `icam-sweep`'s
   `ToolSolid` model the solid today, and `ToolSolid` already treats the holder as colliding but
   not cutting; no query reads the split against the part yet.)
2. **Pose on `ℝ³ × S²`.** Store a frame where needed; optimize and interpolate the axis on `S²`, so
   a phantom roll never adds roughness to the axis path. Never hardcode positions with Z alone.
3. **Engagement and collision are pose queries** (queries 1 and 5), the same at every axis.
4. **Axis assignment sits behind its own interface:** "produce a feasible, smooth, kinematically
   valid `â(s)` over the path". It may be implemented locally or globally (the admissible axes per
   station are non-convex and position-dependent, so real axis planning is often global: the
   admissible set per station, a graph, a smooth least-cost axis path). The path builder never
   assumes locality. A held axis is this interface returning a constant.
5. **Gradients with respect to the pose** come from the contact record (query 1) and feed the pose
   optimizer and the plan.
6. **The removable region `R = max(S, δ − P)` is pose-independent.** What a tool can reach of it is
   accessibility (query 4).
7. **Exactness discipline:** decisions read exact values or certified bounds, never cache
   interpolation (section 5).
8. **The machine model is a service:** forward and inverse kinematics, joint limits, singularity
   proximity, axis rates, consulted while planning, not only when posting. A 3-axis machine is a
   `Chain` of three slides, consulted like any other.

### Module layout (Rust workspace, as of 2026-11-07)

| Crate | Role in the one planner | Prismatic-era code it carries |
|---|---|---|
| `icam-core` | exact geometry, STEP import, `solid3::Exact`, the part's GPU kernel, tool profiles | `frep.rs`, `prismatic.rs`'s part path, `part.rs`, `query.rs`'s references |
| `icam-sweep` | poses, motions, machines, tool solids, certified sweeps, the exact stock, their GPU fields | none |
| `icam-query` | the core queries | the slab clearance (`SliceClearance`) |
| `icam-motion` | the cut model (chip, power, deflection, stability, machine) and the instrument | the instrument measures on `SweptStock` (one slab) |
| `icam-strategy` | plan solves and path reading | plans solved on flat slab grids |
| `icam-rough` | the planner's driver for the removal objective | pocket finding and levels from the prismatic model |
| `icam-finish` | allowance so the finisher sees constant load | reads the prismatic model |
| `icam-assign` | tool assignment: which tool cuts each point (it plans no toolpaths) | reads the prismatic model; splits material among tools by reach on an XY raster |
| `icam-stock` | (retires; `icam-sweep` is the one stock) | `ImplicitStock` (vertical-tool checker) and `SweptStock` (one slab) |
| `icam-post` | programs in and out | 3-axis dialects only |
| `icam-gpu` | (to be folded into the core queries' kernels) | the contact kernel on the prismatic `Part` |
| `icam-sim` | the viewer | the column views |
| `icam-check` | checks of the exact evaluator, its trim loops and spans (the part's GPU kernel is checked in `icam-sim` today) | the prismatic compile and contact-kernel checks |

---

## Part III — Roadmap: milestones of one planner

Each milestone ends with a verifiable artifact: a posted program, checked on the exact stock, and a
measured property. The gates of `CONTRIBUTING.md` hold at every milestone: no gouges, measured with
the exact stock on the posted program; the continuity gate (a 5 µm wall change or a 0.1% physics
change moves at most 5% of the path by more than 1 mm); scores absolute, never relative to another
program. **No milestone adds a mode.** Each one either builds general machinery or lets a held
degree of freedom go.

### Milestone 0 — The core queries, and the prismatic part model retired

1. **No holes in the evaluator** (decision 3): the import certifies every edge to the deviation
   it achieves and reports it (no refusals), cusp edges get a sign rule, the search is bounded by
   construction with no cap; the `unsure` flag and the retired path-walking fallback inside
   `Exact` are deleted. The piece-27829 sign case (`PROOFS.md`) is settled.
2. **CPU reference for tool–part distance and first contact** (queries 1 and 2) for every tool
   profile, shank and holder, tested against brute force on the STEP Tools suite and THE PART.
3. **Levels and regions from drop heights**; `icam-rough` and `icam-strategy` read them in place of
   the slab clearance. The pipeline tests pass.
4. **Checker, bench, finishing allowance, tool assignment and `icam-sim`'s comparison views** read
   the exact field and the core queries. `bench` is re-based to physics: its reference load from
   the cut model, its gates absolute, any comparison to another program a separate report that
   never gates (`--align`, which fitted the part to a program, was removed on 2026-11-07).
5. **GPU kernels** for queries 1 and 2, with measured rates against the CPU reference.
6. **Delete** `frep.rs`, the prismatic part path, `part.rs`, `query.rs`'s references,
   `SliceClearance`, the prismatic contact kernel and the column views. The design guard goes
   green and `EXACT_FIELD.md`'s replacement table is empty. (`SweptStock` stays until milestone
   1 replaces it with query 5, since the instruments measure engagement against it.)

**Exit:** the existing roughing runs, axis held at `(0,0,1)`, on any part the exact evaluator reads,
prismatic parts included as ordinary parts; the gates hold; the guard is green.

### Milestone 1 — The planner, removal objective, axis held

- The functional of section 8 with the axis field held constant: coverage from the `{R<0}` residue,
  load from the cut model through the plan field, the machine model consulted (a three-slide
  `Chain`).
- **Plan solves on leaves.** The solvers in `icam-strategy` rewritten for a leaf surface; a held
  axis gives flat leaves, and nothing in a solver may depend on that.
- **Material at the tool** (query 5) in place of the one-slab stock; `SweptStock` and
  `ImplicitStock` are deleted, and `icam-sweep` checks every posted program.
- **Shallow regions** are part of coverage: a held axis leaves terraces on near-flat areas, and the
  planner's leaves must cover them, not a separate pass.
- **Multi-tool cascade:** each tool plans from the stock the earlier tools' actual paths left
  (`icam-assign`; section 10).
- **Materials without literature:** cut-model constants derived from datasheet properties through
  cutting mechanics, validated leave-one-out, with the validation error carried as margin (Derek,
  2026-10-25). Literature and shop tests are never gates.

**Exit:** a freeform multi-surface part roughed with the axis held, gates met, chip-thickness
scores reported absolute, a big-to-small tool cascade, and a real feed schedule.

### Milestone 2 — The surface objective (finishing), same planner

- The objective changes to surface contact and scallop; the pose, queries, axis interface and
  machine model are the same.
- **Contact patterns from first contact** (query 2) along any direction.
- **Constant scallop from a solved field:** solve `|∇_S g| = 1/s(x)` on the contact surface, with
  the stepover `s` from the exact curvature of the contact feature; paths are level sets of `g`.
- **Rest and pencil** regions from the plan fields and the contact record, not from a 3D medial
  axis.
- The finish allowance from `icam-finish` so the finisher sees constant load.

**Exit:** finished surfaces with measured scallop, gouge-free between stations by move clearance
(query 3), on the same part as milestone 1.

### Milestone 3 — The axis free per region

- The axis field held constant per region instead of everywhere, the values chosen from
  accessibility (query 4) by the planner, not by the user.
- **Full holder and fixture collision** at tilted axes, the first real use of the non-cutting part.
- Seams between regions measured as part of coverage.
- **Output through the machine model** for rotary machines (`Chain`).

**Exit:** an undercut part roughed and finished with the regions' axes chosen by the planner,
holder-checked, posted for a real rotary configuration.

### Milestone 4 — The axis free along the path (simultaneous 5-axis)

- The full functional of section 8: a continuous axis field, curved leaves, the axis-motion,
  stick-out, joint-limit and singularity terms in the objective.
- **The research core** of section 8: reachable smooth foliation with coverage, and state-coupled
  sequencing.
- **Envelope milling** (Part V, Advantage 5): fit any tool's swept envelope to a surface band, with
  a certified error bound.
- **Move clearance** (query 3) verifies every move against the part, between stations.

**Exit:** simultaneous 5-axis rough and finish of a curved, undercut part, collision- and gouge-free
by construction, posted for a real 5-axis configuration. THE PART (a 5/7-axis mill-turn part by
@emm0sh; `test-parts/README.md`) is the standing benchmark: tooling it automatically is the bar.

### Research (ongoing)

- Implicit neural-field process planning (arXiv 2511.17578) as a comparison for the functional.
- Hybrid additive-subtractive: the stock as a field updated by both (`F_new = max(F_old, −F_milled)`
  and its additive counterpart).

---

## Part IV — What is held, and what is free

| Abstraction | Held (milestones 0–2) | Free (milestone 4) | The same code |
|---|---|---|---|
| Pose | axis `(0,0,1)` | axis on `S²` along the path | one pose type on `ℝ³ × S²` |
| Queries | called with a held axis | called with any axis | written once, pose-general |
| Tool | cutter, shank and holder; holder rarely binds | holder and fixtures bind | one solid assembly |
| Axis assignment | returns a constant | local or global smooth `â(s)` | one interface |
| Leaves | flat | curved | one leaf solver |
| Removable region `R` | the same field | the same field | pose-independent |
| Stock | box less sweeps | box less sweeps | `icam-sweep`, any machine |
| Machine model | three slides | any `Chain` | consulted in the loop either way |
| Gradients | plan and path | plan, path and pose optimizer | from the contact record |

**The design invariant:** nothing written for a held axis may break when the axis is free. Later
milestones let degrees of freedom go and add terms; they never replace a subsystem.

---

## Part V — Categorical advantages in the 5-axis solution space

The discipline the four guarantees force (exact fields, certified bounds, no facets, no
intersection logic) is the same machinery that makes the hard 5-axis problems well posed. Each item
states the problem conventional CAM struggles with, why the field formulation removes it, and the
honest catch, so the claim survives an adversarial read.

### Advantage 1 — Grazing-contact accuracy is best exactly where 5-axis lives

Mesh CAM checks gouge and contact against a faceted part, and at near-tangent (grazing) incidence a
small facet-normal error produces a large contact-point error, because the contact slides along the
near-flat approach. Grazing contact *is* the 5-axis regime (high lead and lean, near-flank contact,
curvature-matched passes), so mesh CAM is least accurate exactly where 5-axis is most aggressive.
Exact fields carry no facets, so the gouge verdict is exact independent of incidence angle. The
advantage of exact over mesh is concentrated in the grazing regime that defines 5-axis machining.

**Caveat:** the verdict is only as tight as the certificates (the move clearance's bounds, the f32
margins), so their tightness is measured, not assumed.

### Advantage 2 — Simultaneous 5-axis roughing is a near-empty square in the commercial grid

Commercial CAM roughs with 3+2 (index, rough as 3-axis, re-index) because simultaneous roughing
needs load control against a volumetric in-process stock while the axis tilts, which offset methods
cannot represent. This architecture has the ingredients: an exact stock for any machine, a load
model read at any pose and shaping the plan field, and a differentiable pose. Reach forces
simultaneous motion (deep cavities, undercuts, ports, impeller hubs), and the general method holds
the axis on its own where that is best, so this is the superset of today's roughing modes, not a
niche bolted on (section 8; milestone 4).

**Why 3+2 exists at all:** offset roughing needs a fixed slicing frame to build offset curves, so
it must hold the axis constant over a region. That leaves seams (rest-material steps where two
indexed regions meet) and reach cliffs (an index reaches a feature or it doesn't, with no leaning
in as a cavity deepens). One continuous foliation has neither.

### Advantage 3 — Accessibility is a field over ℝ³ × S², not a combinatorial visibility problem

The classical 5-axis accessibility problem (which axes are collision-free at each contact point)
is attacked by sampling orientations and ray casting or mapping configuration space: expensive,
resolution-limited, and the sampling itself makes axis motion jerky. Here collision is the sign of
query 1, and its gradient with respect to the pose comes with the contact record, so the admissible
axes form a set with computable boundary and gradients (query 4), and accessibility becomes a
smooth constraint along the path.

**Caveat:** the set is non-convex and can be disconnected, so a locally best axis can force a
discontinuous flip downstream; real axis planning is global (through-line 4). The field still pays:
each node's feasibility and each edge's cost are exact queries with gradients.

### Advantage 4 — Exact differential geometry as a shared service

The contact record names the feature, so normals, principal curvatures and directions come from
that feature's own exact surface, not from finite differences on facets (noisy, and worst at high
curvature). They serve curvature-adaptive stepover and scallop (milestone 2), the pose optimizer's
tangency terms, and curvature-matched tooling where a shop uses it.

**Caveat:** the distance field's own Hessian jumps across the medial axis and at edges; curvature
is read from the contact feature, never differentiated out of the field there.

### Advantage 5 — Flank milling generalizes to envelope milling

Flank milling of ruled surfaces is among the hardest 5-axis problems; for non-developable surfaces
the error cannot reach zero and the state of the art is heavy motion optimization, almost always
for cylindrical tools. Here the error is the part's field over the tool's contact band, which is
differentiable in the motion, and its worst case is certified over the whole band. Because the tool
is any solid of revolution, this fits the swept envelope of *any* tool (barrel, conical, tapered)
to any surface strip.

**Caveat:** for non-developable targets the residual has a floor. The deliverable is the least-error
fit plus a certified bound, which also decides per band whether flank milling meets the tolerance.

### Advantage 6 — Gouge-freedom between stations

Conventional CAM checks gouge at discrete contact points and can miss a gouge between them as the
axis interpolates, a well-known 5-axis failure. Here a move's sweep is certified over its whole
duration from the move's rate bounds (`icam-sweep`, already built for the stock), and move clearance
(query 3) applies the same certificate against the part: a guarantee for the continuous motion, not
for samples of it.

### Advantages 7–9 — Sharper statements of advantages already in the design

- **Advantage 7 — Topology changes cost nothing.** As stock is removed, `{R<0}` merges, splits and
  grows islands. Coverage reads the residue of `{R<0}` and plans what is left, so the planner never
  detects or manages regions by hand, at any axis.
- **Advantage 8 — One representation from design to manufacture.** Because process targets are
  fields, scallop, load and stock to leave can vary in space: finer scallop where curvature is high,
  more stock near thin walls, a design-authored tolerance field driving the toolpath directly with
  no lossy handoff through B-rep or STL.
- **Advantage 9 — Machine space folded into generation.** Singularity proximity, joint-limit margin
  and rotary-rate cost are terms in the same objective that chooses the axis, optimized jointly with
  the geometry, not checked after it.

**Dependencies, so no claim reads as free:** Advantages 1 and 6 ride on the certificates' tightness;
Advantage 2 rides on the research core of section 8; Advantage 3 rides on the global axis solve of
through-line 4; Advantages 4 and 5 ride on the contact record of query 1.

---

## Appendix A — Parked and rejected (intentional)

- **Planners, queries or part models for a restricted mode:** rejected 2026-11-07 (Derek). The
  prismatic part model is being replaced (Part III, milestone 0).
- **Global CSG for B-rep solids:** not pursued (decision 2): fitted B-spline faces cannot supply
  separating primitives.
- **Per-cell formulas** (2026-10-02), **the glued field and the pure field** (2026-10-25): rejected;
  see `EXACT_FIELD.md` and `FREP_GLUING.md`.
- **An eikonal redistance from F as the distance:** replaced by the exact evaluator (2026-10-27).
- **Voxel, dexel or height-map stock:** rejected 2026-10-24 (Derek: "I don't machine with a
  minecraft pickaxe").
- **A 3D medial axis:** unstable; medial information comes from solved plan fields.
- **Fidget, libfive, OpenVDB, OpenVoronoi:** assessed, not used (section 9).

## Appendix B — Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| A query or model written for a held axis | Foundational | The rules at the top; the 5-axis caller is named in every proposal; review |
| Tool–part distance too slow, or its certificate unbounded on near-flat contact | Foundational | Certify the least value by intervals; GPU scatter per (pose, feature); rates measured before adoption (decision 10) |
| Plan solvers that only work on flat grids | Milestone 1 | Write the leaf solver first; the flat leaf is a test case, not a target |
| Foliation and sequencing unsolved in general | Milestone 4 | Treated as the research core; the held-axis milestones ship value meanwhile without mode-specific code |
| Pose optimization local minima, axis flips | Milestones 3–4 | Global axis assignment behind its interface; warm starts; smoothness terms |
| Leaving rest material | Every milestone | Coverage from the `{R<0}` residue, checked on the exact stock |
| The part's open sign case (piece 27829) | Every milestone | `PROOFS.md` claim 4; gouge verdicts re-decided on the CPU |
| Patent exposure | Legal | Field mechanisms, prior-art publication, claims read before public claims (section 10) |

## Appendix C — Key references

- **The reference set** (80 papers, verified citations): `whitepaper-references/MANIFEST.md`. Edit
  `whitepaper-references/references.json` and run `node tools/fetch-references.mjs` to add a paper;
  `--verify` re-hashes what is on disk.
- Implicit Neural Field Process Planning for Multi-Axis Manufacturing — arXiv 2511.17578.
- Bieterman and Sandstrom (2003): PDE spiral toolpaths; the prior art field spirals rest on.
- Constant-engagement lineage, for what decision 8 rules out: Kramer (NIST 1992), Stori and Wright
  (1998), Ibaraki (2005).
- Assessed and not used: Fidget (github.com/mkeeter/fidget), libfive (github.com/libfive/libfive),
  OpenVoronoi, OpenVDB.
