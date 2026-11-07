# The exact field

**Decision, 2026-10-25 (Derek): the pure field is rejected.** Earlier the same day it was
recorded here as adopted with conditions; it was built under those conditions, judged against
the project's guarantees, and found to fail them. The guarantees are the standard every field or
distance method is judged by before it is offered for approval.

**Rewritten 2026-10-27 (Derek): the distance is the exact evaluator's.** The route below this
page first planned (an eikonal redistance from F) is replaced by `solid3::Exact`, the signed
distance through trim loops certified at import, sampled into the brick cache for the global
field ("What aligns"). CONTRIBUTING.md decisions 2, 3 and 10 were rewritten to match, with the
evaluator's verdict against the four guarantees.

## The four guarantees

The project rests on these, and a method that fails any one of them is not offered for
approval, with or without conditions:

1. **Fast and parallel.** The method evaluates as a straight-line kernel over independent
   points, cells, poses or pixels (decision 9), and the proposal states the measured points per
   second.
2. **Exact where it claims to be.** A true field, not an approximation presented as one.
3. **No holes.** No unresolved cells, no refused parts hidden behind fallbacks, no threshold
   that decides structure (an epsilon step that says whether a point is on the boundary is a
   hole waiting to open).
4. **No brittle intersection or B-rep logic.** No solving where surfaces meet, no trim tests,
   no stored edge curves or vertices deciding anything, at compile time or at a query.
   **Read by Derek, 2026-10-26:** nothing in the architecture may open holes or failures in the
   exact inside/outside test everything downstream relies on, and no edge finding whose search
   time explodes. Under that reading (2026-10-27), trim loops built at import and certified
   within the file's tolerance, read by exact crossing parity, are allowed: they always answer,
   with no retry and no threshold. Edges and vertices are read at a query for their own
   distance and sign.

The roughing plan fields read the distance to build the path plan, so the distance away from
the surface has to meet the guarantees too, not only the sign and the zero set.

## The verdict on the pure field

The construction (FREP_GLUING.md, "The pure field and the cell certificate") has two parts.

**F, the sign and zero set,** is one CSG formula, F = min over the octree's cells of
max(support ball, cell formula), over exact primitives. It meets guarantees 1, 2 and 4. It does
not carry the distance: each term is capped by its cell's ball, so away from the surface F is a
shallow bowl per cell, which is why the glued field was rejected in slices on 2026-10-02. It
cannot drive roughing on its own.

**E, the distance,** is the distance from x to the nearest boundary point among the critical
points of distance on the surfaces, on the curves where two surfaces meet and at the points
where three meet, each kept only where F takes both signs just off it. Built and measured on
2026-10-25 (the code was discarded, not committed):

- Guarantee 4 fails. Finding a point on the curve where two surfaces meet is Newton onto both
  surfaces, a slide along their crossing, bisection to where the exposed part ends, and a
  failure branch where the surfaces are tangent. That is surface–surface intersection logic
  with F standing in for trim loops: the same class of failure under another name. The
  compiler behind it still called the trim test (`face_distance`), sampled stored edge curves to
  choose between max and min, and read edge frames off stored curves.
- Guarantee 3 fails. "On the boundary" was a sign change of F across an epsilon step.
- Guarantee 1 fails. Each query is a serial, branchy, data-dependent search: 34 to 277 points
  per second measured against the reference, about 3,000 per second alone, on the CPU. Not a
  kernel.
- Guarantee 2 held: on the three test parts it matched the B-rep reference within the file
  tolerance at 99.98 to 100% of 2,000 points per solid, with no sign errors and no Lipschitz
  breaks, where F's formulas were valid.

Exactness alone does not carry a method. The pure field is the exact answer computed the way
the project exists to replace.

## What aligns

**Rewritten 2026-10-27 (Derek).**

- **The field** is the exact signed distance, evaluated by `solid3::Exact`: a face's nearest
  critical point when its certified loops (`trim2`) put it in the face, otherwise its edges and
  vertices, signed by outward normals and pseudo-normals. The file's declared tolerance is the
  accuracy line at edges (2026-10-26); answers it cannot decide are marked unsure. Measured
  (`icam-check exact`, docs/SPAN_IMPLICIT.md section 15): every check
  clean on THE PART at 2,030 to 2,850 queries a second on all threads, no unflagged failure in
  393,200 suite queries.
- **The global field** for planning and viewing is that evaluation sampled into the sparse
  brick cache, read as a straight-line kernel. Near the surface, finishing reads the exact
  value.
- **Guarantee 1** is met by the cache reads now and by the step-5 GPU kernel for exact queries
  (fixed Newton steps per brick from per-span seeds, with a certificate; measured 99.997% exact
  in 0.5 mm bricks before the certificate). The CPU evaluator is the reference it is tested
  against. Until the kernel exists, exact queries are the CPU's per-point search.
- **F** and the eikonal redistance planned here on 2026-10-25 are no longer the route: no CSG
  formula is compiled (decision 2).

## What the guard now refuses (2026-10-25)

Recorded here so the same thing is not built a second time:

- `distance-by-intersection` (every crate outside import and icam-check): identifiers for
  intersecting, meeting, projecting onto surface pairs, bisecting, exposure, sliding.
- `brep-geometry-in-compiler` (the compiler): the trim test `face_distance`, `edge_frame`,
  `use_of`, stored curve geometry (`.curve.at`, `.curve.closest`, ...), vertex positions.
- The session-start text, the edit checks and the doc checks carry the four guarantees and the
  verdict above, so a new session cannot reach the same construction by a different name.
- **2026-10-27 (Derek):** runtime code may build and query the exact evaluator
  (`step::read`, `Solid3::from_step`, `solid3::Exact`); B-rep internals stay out of it, and the
  rules' reasons now name the certified loops as the only trim test and drop the eikonal route.
  `trim2.rs` is import code.
- **No allowances.** The guard's baseline grandfathers nothing (Derek, 2026-10-25): code that
  breaks a rule is reported until it is replaced, a commit that touches it is refused, and an
  edit may only lower its count. Grandfathered code reads as precedent to the next agent and
  means the project is not building toward its own plan.

## Slated for replacement

Code that breaks the rules as written, kept only until its replacement exists. It is not a
precedent for anything.

| Where | What | Count | Replacement |
| --- | --- | --- | --- |
| `frep.rs` | the prismatic compiler's section leaves are the STEP edge curves, trimmed to their vertices, picked by edge-to-face topology and a wall-depth test; exact to the curves, which sit off the solid's surfaces by the file tolerance (fails guarantee 4; Derek, 2026-10-25) | 1 | the exact evaluator and the brick cache (2026-10-27); the planner reads the prismatic field until the cache exists |

The tree reports these at every session start and stop until they are gone.

Done: `frep3.rs` (probe-chosen formulas, the B-rep `sdf` in the compiler, trim loops, the trim
test and stored curves, the sampled exposure test: 47 hits) was deleted on 2026-10-30 with the
`icam-check` commands built on it (`frep3`, `primtime`, `slice`, `surfcheck --field`). Git
history keeps it.

## How this happened, for the record

The proposal in FREP_GLUING.md was presented to Derek as a choice between approving,
approving with conditions and rejecting, with a list of decision violations to fix, instead of
a verdict against the guarantees. "Approved with conditions" then licensed building the
intersection logic under the guard's existing allowances (`face_distance` in the compiler was
allowed by decision 3's wording). The guarantees above are the test that should have been
applied first; from now on they are applied first.
