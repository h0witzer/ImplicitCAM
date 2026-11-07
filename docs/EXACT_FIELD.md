# The exact field

**Decision, 2026-10-25 (Derek): the pure field is rejected.** Earlier the same day it was
recorded here as adopted with conditions; it was built under those conditions, judged against
the project's guarantees, and found to fail them. The guarantees are the standard every field or
distance method is judged by before it is offered for approval.

**Rewritten 2026-10-27 (Derek): the distance is the exact evaluator's.** The route below this
page first planned (an eikonal redistance from F) is replaced by `solid3::Exact`, the signed
distance through trim loops certified at import ("What aligns"). CONTRIBUTING.md decisions 2, 3
and 10 were rewritten to match, with the evaluator's verdict against the four guarantees.

**Restated 2026-11-07 (Derek): the goal was pure implicit CSG; it is blocked; the evaluator is
the compromise; and holes are impossible.** The goal was never to rebadge B-rep logic as an
F-rep planner but to turn B-rep constructs into pure F-rep ones through an implicit CSG, with no
trims and no intersection tests, because trims and intersections are what make B-rep brittle and
an implicit CSG has no holes or seams by construction. That is blocked for fitted B-spline faces
(no exact implicit, no separators at tangent edges; sections 3 to 5 of `SPAN_IMPLICIT.md`), so
the certified loops are the compromise, and a rigorous global route would still be preferred.
Every "unsure" answer is a hole: the import never refuses a face, cusp edges get a sign rule, the
search is bounded by construction, and the flag goes (milestone 0 of the roadmap). CONTRIBUTING
decisions 3, 4, 7 and 10 carry the record; this page is the reasoning.

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
4. **No brittle intersection or B-rep logic.** Nothing in the architecture may open a hole, a
   seam or a failure in the exact inside/outside test everything downstream relies on, and no
   edge finding whose search time explodes (Derek, 2026-10-26). That rules out solving where
   two model surfaces meet, a trim decided by walking a surface, an on-the-boundary test by a
   threshold of our own, and any per-point search over intersection curves. It allows what
   cannot open a hole: trim loops built at import, certified to the larger of the file's
   tolerance and the measured deviation, read by exact crossing parity (always an answer, no
   retry, no threshold), and edges and vertices read at a query for their own distance and
   sign. The earlier wording of this guarantee ("no trim tests, no stored edge curves or
   vertices deciding anything") was the pure-CSG goal stated as a rule; the goal is blocked
   (above), and the rule is this one (Derek, 2026-11-07).

The plan fields read the distance to build the path plan, so the distance away from the
surface has to meet the guarantees too, not only the sign and the zero set. Planning reads it
exactly, through the evaluator's GPU kernel; the brick cache (values clamped to a cell's
diagonal) is for drawing and for coarse entries far from the surface (Derek, 2026-11-07).

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

**Rewritten 2026-10-27 (Derek); restated 2026-11-07.**

- **The field** is the exact signed distance, evaluated by `solid3::Exact`: a face's nearest
  critical point when its certified loops (`trim2`) put it in the face, otherwise its edges and
  vertices, signed by outward normals and pseudo-normals. The accuracy line at an edge is the
  larger of the file's declared tolerance and the measured deviation (2026-10-26, 2026-11-07).
  Measured (`icam-check exact`, docs/SPAN_IMPLICIT.md section 15): every check clean on THE
  PART at 2,030 to 2,850 queries a second on all threads; the suite has 3 Lipschitz breaks and
  1 sign failure in one inconsistent solid (`rear`), unflagged, on the retirement list.
- **Holes, closed in milestone 0** (Derek, 2026-11-07: holes are impossible). Today the
  evaluator marks three cases "unsure": a face whose loops were refused (26 suite faces at the
  1e-6 tolerance floor, one sphere face on THE PART whose loop end lies past a knot line), a
  cusp edge, and a search at its step cap; for a refused face it still runs the retired
  path-walking test. Each is a hole. The import will certify every edge to the deviation it
  achieves and report it, cusp edges get a sign rule, the search is bounded by construction, the
  flag and the old test are deleted. One wrong sign carried through the brick cache (piece
  27829, `PROOFS.md` claim 4) is open.
- **Planning reads the exact value** through the GPU kernel (`solid3/gpu.wgsl`, run by
  `icam-sim`'s `partgpu.rs`: fixed Newton steps per brick from per-span seeds, with a
  certificate; THE PART in 1 mm bricks in 2.3 s). The CPU evaluator is the reference it is
  tested against (`icam-sim part-bench --gpu`). The sparse brick cache samples the evaluation
  for drawing and for coarse entries far from the surface.
- **Guarantee 1** is met by that kernel. Guarantee 2 is met on THE PART. Guarantee 3 is not
  yet met (the holes above). Guarantee 4 is met as read on 2026-10-26.
- **F** and the eikonal redistance planned here on 2026-10-25 are no longer the route: no CSG
  formula is compiled (decision 2).

## What the guard now refuses (2026-10-25)

Recorded here so the same thing is not built a second time:

- `distance-by-intersection` (every crate outside import and icam-check): identifiers for
  intersecting, meeting, projecting onto surface pairs, bisecting, exposure, sliding.
- `brep-geometry-in-compiler` (the compiler): the trim test `face_distance`, `edge_frame`,
  `use_of`, stored curve geometry (`.curve.at`, `.curve.closest`, ...), vertex positions.
- The session-start text, the edit checks and the doc checks carry the decisions, copied
  verbatim from CONTRIBUTING.md by `tools/frep-guard/sync-decisions.mjs`, so a new session
  cannot reach the same construction by a different name.
- **2026-10-27 (Derek):** runtime code may build and query the exact evaluator
  (`step::read`, `Solid3::from_step`, `solid3::Exact`); B-rep internals stay out of it, and the
  rules' reasons now name the certified loops as the only trim test and drop the eikonal route.
  `trim2.rs` is import code.
- **No allowances.** The guard's baseline grandfathers nothing (Derek, 2026-10-25): code that
  breaks a rule is reported until it is replaced, a commit that touches it must lower every
  count it is over on, and an edit may only lower its count. Grandfathered code reads as
  precedent to the next agent and
  means the project is not building toward its own plan.

## Slated for replacement

Code that breaks the rules as written, kept only until its replacement exists. It is not a
precedent for anything.

| Where | What | Count | Replacement |
| --- | --- | --- | --- |
| `frep.rs` | the prismatic compiler's section leaves are the STEP edge curves, trimmed to their vertices, picked by edge-to-face topology and a wall-depth test; exact to the curves, which sit off the solid's surfaces by the file tolerance (fails guarantee 4; Derek, 2026-10-25) | 1 | the pose-general core queries against the exact evaluator (roadmap Part II and Part III, milestone 0; Derek, 2026-11-07), which retire the whole prismatic part model, not only this line; the planner reads the prismatic field until they land |
| `solid3.rs` (`Exact::face`) | the retired path-walking trim test (`nurbs_face_closest`, `classify`) still runs for a face whose loops were refused, and three cases answer "unsure" (no pattern sees this; a hole by decision 3 as restated 2026-11-07) | 0 | no refusals, a cusp sign rule, a cap-free search; the fallback and the flag deleted (milestone 0) |

The tree reports the first at every session start and stop until it is gone; the second is
listed here because the guard cannot see it.

Done: `frep3.rs` (probe-chosen formulas, the B-rep `sdf` in the compiler, trim loops, the trim
test and stored curves, the sampled exposure test: 47 hits) was deleted on 2026-10-30 with the
`icam-check` commands built on it (`frep3`, `primtime`, `slice`, `surfcheck --field`).

## How this happened, for the record

The proposal in FREP_GLUING.md was presented to Derek as a choice between approving,
approving with conditions and rejecting, with a list of decision violations to fix, instead of
a verdict against the guarantees. "Approved with conditions" then licensed building the
intersection logic under the guard's existing allowances (`face_distance` in the compiler was
allowed by decision 3's wording). The guarantees above are the test that should have been
applied first; from now on they are applied first.
