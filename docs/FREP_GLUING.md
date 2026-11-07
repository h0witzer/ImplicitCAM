# Gluing local formulas into one global F-rep

*`frep3.rs` and the `icam-check frep3` and `slice` harness this page measures were deleted on
2026-10-30; the measurements below are kept as the record.*

*Proposal, 2026-10-02 (Claude, for Derek's review). Not adopted. It answers the open item in
`ARCHITECTURE_AND_ROADMAP.md` section 9: compile every B-rep solid, B-spline faces included, to
one global F-rep.*

**Decision, 2026-10-02 (Derek):** the glued and one-cell fields are dropped as planning
fields. Seen in slices, they carry nothing like the medial-axis information roughing needs.
The exact field section below remains a proposal for Derek's review.

## The problem

The compiled field F must be, from section 5 and the 2026-09-29 amendment:

1. one function on ℝ³, continuous, with no seams between cells;
2. signed like the solid: negative inside, positive outside, zero exactly on the boundary;
3. a bound: |F| never exceeds the distance to the boundary, and F is 1-Lipschitz, so F(x) > 0
   certifies a clear ball of radius F(x) around x;
4. exact (|F| equal to the distance) near the faces wherever one face is nearest;
5. prunable per cell, the pruned tree being an exact restriction of the global function;
6. free of trim decisions at runtime: a cell evaluates min and max over exact primitives.

The canonical route compiles the B-rep to CSG with separating halfspaces. The literature in the
reference set solves it for faces on planes, cylinders, cones and spheres bounded by planar edges
(Shapiro and Vossler 1993, entry 70), gives tangent-plane and polar-plane constructions for
nonplanar edges between convex quadrics (the Vossler and Shapiro patent, entry 76), and builds the
tree once the separators are known (Buchele and Crawford 2003, entry 72). Nothing covers tori,
B-spline faces or curved tangent edges. The algebraic level-set route (Upreti and Subbarayan,
restated in the Liao thesis, entry 80) fails by its own account: the implicit of a patch can cross
itself inside the patch's hull, and knot insertion cannot remove it.

## Where frep3's seams came from

`frep3.rs` builds, for each leaf cell, a formula meant to hold over the cell's reach ball (twice
the cell's radius), clamped by `r_out - |x - c|`, which already falls to zero at that ball's edge.
Evaluation then reads only the leaf that holds x. Neighbors' formulas differ on their shared
face, so the field jumps there: 244 of 20,000 close pairs on the test nest, by up to 945 mm. The
formulas are not the problem. The rule that reads one of them is.

## Definitions

S is a closed regular solid, sd its signed distance (negative inside), σ(x) the sign of sd(x).

A **local formula** on an open convex set U is a Boolean formula Φ over halfspaces
{f_j ≤ 0}, each f_j the exact signed distance to a whole surface (any 1-Lipschitz function
whose magnitude is the distance to its zero set will do), such that

    S ∩ U = Φ ∩ U.

Its F-rep φ takes max for intersection, min for union and negation for complement. Then φ is
1-Lipschitz, φ has sign σ on U wherever it is nonzero, and |φ(x)| ≤ dist(x, ∂Φ) (min and max
of signed distances bound the signed distance of the result from below; Balint and Valasek 2019).
Call the formula **strict** if φ is also nonzero on U away from ∂S. A select node (frep3's `Sel`,
which takes one operand or the other by the sign of a separator s) is allowed when the two
operands agree on {s = 0} ∩ U.

Let ρ_U(x) = dist(x, ℝ³ \ U), the distance to the edge of U.

## Lemma (fading)

Let φ be the F-rep of a local formula on U and

    φ̂(x) = sign(φ(x)) · min(|φ(x)|, ρ_U(x))   for x in U,   φ̂(x) = 0 otherwise.

Then φ̂ is 1-Lipschitz on ℝ³, has sign σ wherever it is nonzero, and |φ̂| ≤ |sd|.

*Proof.* Sign: from φ. Bound: the boundary point of S nearest x is either in U, where ∂S agrees
with ∂Φ, at distance at least dist(x, ∂Φ) ≥ |φ(x)|; or outside U, at distance at least ρ_U(x).
Lipschitz, for x and y in U: if φ(x) and φ(y) have the same sign, |φ̂(x) − φ̂(y)| ≤
max(|φ(x) − φ(y)|, |ρ_U(x) − ρ_U(y)|) ≤ |x − y|, because min is 1-Lipschitz in each argument.
If they have opposite signs, the segment from x to y lies in U (U is convex) and holds a zero z
of φ, so |φ̂(x) − φ̂(y)| ≤ |φ(x)| + |φ(y)| ≤ |x − z| + |z − y| = |x − y|. For x in U and y
outside: |φ̂(x)| ≤ ρ_U(x) ≤ |x − y|. ∎

## Theorem (gluing)

Let (U_i, φ_i) be local formulas whose sets U_i are locally finite, and let

    F = max(0, max_i φ̂_i) + min(0, min_i φ̂_i).

Then:

1. F is 1-Lipschitz on ℝ³;
2. F has sign σ wherever it is nonzero; if the U_i cover ℝ³ and every formula is strict, F is zero
   exactly on ∂S;
3. |F| ≤ |sd|, with equality at x whenever some i has |φ_i(x)| = |sd(x)| ≤ ρ_{U_i}(x);
4. on any box B, F equals the same expression over only the i whose U_i meets B (an exact
   restriction).

*Proof.* By the lemma, every nonzero φ̂_i(x) has sign σ(x), so at each point one of the two terms
is zero: F = P = max(0, max_i φ̂_i) where σ ≥ 0, and F = N = min(0, min_i φ̂_i) where σ ≤ 0.
(1) If σ(x) and σ(y) are both ≥ 0, |F(x) − F(y)| = |P(x) − P(y)| ≤ |x − y|, because a max of
1-Lipschitz functions is 1-Lipschitz; likewise when both are ≤ 0. If σ(x) > 0 > σ(y),
|F(x) − F(y)| = P(x) + |N(y)| ≤ |sd(x)| + |sd(y)| = sd(x) − sd(y) ≤ |x − y|, since sd is
1-Lipschitz. (2) Sign: as above. For a point x in some U_i away from ∂S, a strict φ_i gives
φ̂_i(x) ≠ 0, so F(x) ≠ 0; on ∂S, |F| ≤ |sd| = 0. (3) |F(x)| = max_i |φ̂_i(x)|, and the lemma's
bound. (4) φ̂_i vanishes outside U_i, and zeros do not change max(0, ·) or min(0, ·). ∎

The fact that makes this work is that valid local formulas cannot disagree in sign anywhere,
because the sign belongs to the solid, not to the formula. So gluing them by largest magnitude
cannot cancel anything.

## What it buys

- **No global separation problem.** Only local describability is needed: S ∩ U = Φ ∩ U inside
  each small set. Shapiro and Vossler's global separators, the open case in their Table I, are
  never constructed.
- **B-spline faces.** The halfspace of a B-spline face is the signed distance to its surface, by
  certified closest point (`bspline.rs`, `surface3.rs`), with no implicitization. It is a valid
  f_j in any U inside the face's normal tube: every point of U has its nearest surface point
  inside the patch and closer than the focal distance there. The patch is extended past its
  border exactly, by de Casteljau subdivision at parameters outside [0, 1], so sets near the
  face's edges still see an interior nearest point. The global sign question for a bounded patch
  never arises.
- **Tori** are already exact primitives; nothing global is needed.
- **Curved tangent edges.** Take the separator s(x) = (x − c(x)) · m(c(x)), where c(x) is the
  edge point nearest x and m points across the edge into face 1, perpendicular to the edge and
  tangent to the face. Near the edge, s is zero exactly on the normal sheet {c + t·n(c)}. On that
  sheet both faces' signed distances equal t while |t| stays below both faces' focal distances,
  so a select node between them is continuous there. For a straight edge s is frep3's separator
  plane. A ball larger than the fillet radius breaks this agreement, which bounds the cell size at
  a tangent edge.
- **Selects at tangent edges.** Where two faces meet tangentially, max and min both fail: near
  the edge one face's surface lies wholly on one side of the other's, so their intersection
  drops one face and their union drops the other (a plane and a fillet: the solid cylinder lies
  under the plane). This is Shapiro and Vossler's non-describable case, and Buchele and
  Crawford's Fig. 11 adds a separator there. frep3 enters the separator as a select, taking each
  face's distance on its own side. That is exact, but continuous only where the two sides agree
  on the separator. The same set has a min/max form,
  min(max(−s, pos), max(s, neg), max(pos, neg)). The third term is the consensus of the first
  two: it adds no points and keeps the value off zero where both sides are inside. This form is
  always continuous and 1-Lipschitz, but beside the separator its magnitude is held to |s| on
  one side. Under gluing, an overlapping single-face cell supplies the exact value there.
- **Mixed vertices** (convex and concave edges at one vertex). Near the vertex the solid is close
  to a cone over a spherical polygon. Central projection turns that into a planar polygon, which
  has a CSG formula using each edge's half-plane once (Dobkin, Guibas, Hershberger and Snoeyink
  1993, entry 71, going by its abstract; no open copy yet). That would replace probe-chosen
  formulas at vertices, which are what fail below.
- **Pruning and intervals.** A cell's pruned tree is the max and min over the few supports that
  meet it, each pruned further by interval bounds. The interval of φ̂ follows from those of φ and
  ρ_U. The per-cell straight-line kernels of the 2026-09-29 amendment apply unchanged.
- **Booleans.** min and max of glued fields of two solids bound the result's distance, as for any
  bound fields, so parts, fixtures and stock compose as before.

## The remaining obligation: certify each local formula

The theorem asks for S ∩ U = Φ ∩ U on every support. frep3 picks rule formulas (one face, convex
edge, concave edge, tangent edge, chains, all-convex or all-concave vertices, joined parts) and
falls back to formulas chosen by probes, a 4 × 4 × 4 grid over the reach ball. Probes are
sampling, not proof.

A certificate that stays local:

1. *Faces exposed.* On each face Q_j ∩ U, φ = 0: the other operands do not dominate f_j there.
2. *Remainders hidden.* On each remainder (surface_j \ Q_j) ∩ U, φ ≠ 0.
3. *One sign per component.* At one point beside each face piece, on each side, sign(φ) = σ.

Conditions 1 and 2 make the zero set of φ in U equal ∂S ∩ U. Then φ's sign is constant on each
component of U \ ∂S, and condition 3 fixes it. Conditions 1 and 2 are statements about
two-dimensional surface pieces inside U, checked by interval bounds over subdivided parameter
boxes (Bezier patches for B-spline faces). On failure, the cell splits.

## Measurements

The test nest, the 38 solids frep3 compiles (3 refused for B-spline faces), 20,000 points (half
uniform over each solid's padded box, half near its edges), against the exact B-rep reference.
Continuity counts close pairs (0.01 to 1 mm apart) whose values differ by more than their
distance. The support is the ball each leaf contributes over, in cell radii: 2 is the whole reach
ball.

| Evaluation | Sign errors | Above the exact distance (worst) | Both signs at a point (worst) | Continuity breaks (worst) | Exact |
| --- | --- | --- | --- | --- | --- |
| frep3 today: one leaf per point | 0 | 0 | not applicable | 244 (945 mm) | 36.4% |
| glued, support 2, selects | 11 | 61 (1.57 mm) | 26 (8.59 mm) | 9 (0.30 mm) | 46.4% |
| glued, support 1.5, selects | 0 | 2 (0.041 mm) | 0 | 1 (0.002 mm) | 43.0% |
| glued, support 1.25, selects | 0 | 0 | 0 | 1 (0.002 mm) | 40.4% |
| glued, support 1.25, min/max selects | 0 | 0 | 0 | 1 (0.002 mm) | 39.8% |
| glued, support 2 for rule formulas and 1.25 for probed ones, selects | 0 | 1 (0.124 mm) | 1 (8.59 mm) | 2 (0.081 mm) | 45.4% |
| the same with min/max selects | 0 | 0 | 1 (8.59 mm) | 1 (0.002 mm) | 44.8% |

The glued evaluator lives in the `icam-check frep3` harness (`ICAM_GLUE`, `ICAM_GLUE_PROBED`,
`ICAM_SEL=minmax`); `frep3.rs` is unchanged. The harness also reports every contribution that is
wrong (wrong sign, or above the exact distance) by leaf kind and by its place in the support:

- At support 2, 100 of the 102 wrong contributions come from probe-chosen formulas, 82 of them in
  the outer fifth of the ball. No formula chosen by rule (one face, convex, concave, tangent,
  joined parts) is wrong anywhere in its whole reach ball, apart from one chain and one vertex.
- The chain is a select used beyond the region where its two sides agree. Its min/max form
  removes both the 0.124 mm overshoot and the 0.081 mm break.
- The vertex is an all-convex or all-concave vertex rule used about 200 mm from its vertex, where
  the untrimmed surfaces come back around. It is gone at support 1.25.

## Differentiability: what any field can have, and what min/max gives up

**No field matching the part is differentiable everywhere.**

- *At sharp edges and vertices.* If F is continuously differentiable near p with ∇F(p) ≠ 0,
  the implicit function theorem makes {F = 0} a smooth surface near p. The boundary has a
  crease at every sharp edge and vertex, so any F with that zero set is either not
  differentiable there or has ∇F = 0 there. R-functions take the second option; the normal
  ∇F/|∇F| is then undefined at the edge anyway.
- *On the medial axis.* Off the boundary, sd is differentiable exactly where the nearest
  boundary point is unique, and there |∇sd| = 1. sd has a minimum at the deepest interior point.
  A differentiable function has zero gradient at a minimum, so sd is not differentiable there:
  every bounded solid has a nonempty medial axis, and no field equal to sd around a medial point
  is differentiable at it.

**A min/max field points the wrong way at edges.** Wherever a min/max expression over face
surfaces is differentiable, its gradient is one primitive's gradient: the normal direction of
some face's surface. At a point whose nearest boundary point q lies on a sharp edge (in the
edge's normal wedge), the true gradient (x − q)/|x − q| lies strictly between the two faces'
normals at q. So the field's direction is wrong there. For planar faces the error is exactly the
angle to the nearer normal, up to half the turning angle. This holds for a global CSG formula
as much as for glued local ones: it comes from min and max, not from locality. It contradicts
the roadmap's section 7, point 1 ("direction survives the bound") inside every edge and vertex
region.

Measured on the test nest: 17,567 points pulled toward their nearest boundary point to a true
distance of 0.02 to 1 mm, gradient by central differences, against the true gradient (support
1.25, min/max selects). The share with a direction error above 1 degree:

| Nearest feature | Points | One leaf per point | Glued |
| --- | --- | --- | --- |
| Face | 7,177 | 0.1% | 0.9% |
| Edge | 8,849 | 93.8% | 86.5% |
| Vertex | 1,541 | 99.9% | 97.5% |

The fade caps account for at most 1% of the wrong directions. The 0.9% at faces is mostly the
min/max select form, whose magnitude beside a separator is |s|.

**What reaches the limit.** A glued field equals sd at every point where some support's formula
is exact and uncapped, because the max of lower bounds attains the exact one. Where that holds on
an open set, F is differentiable exactly where sd is: everywhere except the medial axis and the
edges, which no field can avoid. Gluing admits exact edge and vertex pieces: the distance to the
edge curve inside its normal wedge (or to the vertex inside its cone), picked by sign tests
against the edge's normal sheet. No min/max formula over face surfaces can express those. So in a
band around the part, glued formulas with exact edge and vertex pieces can be as differentiable
as any valid field, and min/max formulas, global or glued, cannot. Not built yet.

**What each planning method takes from the part field** (roadmap sections 5, 7, 8, Phase 2,
Phase 4, Advantage 4):

| Method | What it takes | Min/max field, global or glued | Glued, with exact edge and vertex pieces |
| --- | --- | --- | --- |
| Gouge, collision and interval-over-time certification | a 1-Lipschitz lower bound and its intervals; no derivative | yes (glued: proved above) | yes |
| Roughing walk on the wall of R = max(S, δ − P) | a continuous, piecewise-smooth field (R is itself a max) | yes; the wall is the sharp offset at convex edges, which leaves extra stock (safe) | yes; the wall becomes the true rounded offset |
| Nearest-wall direction (section 7, point 1) | ∇P parallel to ∇sd | no, in every edge and vertex region | yes, off the medial axis |
| Curvature and scallop (Phase 2, Advantage 4) | the Hessian | no (the roadmap already sends this to the exact field) | yes inside the face regions, as smooth as the surfaces |
| 5-axis pose gradients | gradients almost everywhere of a Lipschitz field | as for any min/max field | exact gradients off the medial axis |

## The exact field (built in the harness, 2026-10-02)

**Why edges, and why not loops.** In an edge's normal region the nearest boundary point lies on
the edge curve, so the true distance there is the distance to that curve. That is the failure
above: min and max of face values cannot produce it. For two planes the edge distance is a fixed
function of the two face values, √(f₁² − 2c·f₁f₂ + f₂²) / sin θ with c = n₁·n₂, and needs no
edge object at all; in general it is a distance-to-curve primitive. Either way it is geometry, a
point set, not topology. Loops are not needed. A first version decided whether a foot lies in its
face by walking the face's loops to the nearest edge and reading its orientation. That is B-rep
evaluation at runtime, and it failed on circle edges of solid #46244 whose orientation came out
reversed. It is replaced by the F-rep's own test.

**Construction.** Let F be the glued bound field (sign exact, zero set exactly ∂S). For a cell U
with reach ball B_U, let its candidates be the edge curves meeting B_U, and each face surface
meeting B_U at each critical point of the distance from x (the foot, plus the far points on
cylinders, spheres and tori), kept only where F = 0 there. F is evaluated with that surface's
own term set to zero, so the test is a sign test of the other terms, with no tolerance on the
face's own value. Let u_U(x) be the distance to the nearest kept candidate and
ρ_U(x) = radius(B_U) − |x − c_U|, with ρ = ∞ at the root. Take the cells from the leaf holding x
up to the root, stop at the first with u_U(x) < ρ_U(x), and set E(x) = sign(F(x)) · u_U(x).

**Lemma.** Every kept candidate is a boundary point (edges are; a surface point is exactly when
F = 0 there), so u_U ≥ |sd|. If u_U(x) < ρ_U(x), the nearest boundary point q is closer than
ρ_U, hence inside B_U. There it lies on a candidate edge or in the interior of a candidate face,
where it is a critical point of distance on that face's surface. So u_U(x) = |sd(x)|.

**Theorem.** E = sd everywhere, the root being uncapped. So E is continuous and 1-Lipschitz, and
differentiable exactly where sd is: everywhere except the medial axis and the boundary's own
edges, the best any field can do by the results above. Its gradient is the true nearest-boundary
direction. The only premise is the bound field's zero set, and so the validity of the local
formulas.

Measured on the test nest (support 1.25, min/max selects; gradient by central differences against
the true gradient):

| Points | Field | Exact | Direction off by more than 1 degree (face / edge / vertex) |
| --- | --- | --- | --- |
| 17,567 at 0.02 to 1 mm | one leaf per point (min/max) | 40.8% | 0.1% / 93.8% / 99.9% |
| | exact field | 100.0% | 0.00% / 0.00% / 0.00% |
| 12,469 at 1 to 10 mm | glued (min/max) | 23.3% | 19.2% / 91.9% / 98.5% |
| | exact field | 100.0% | 0.00% / 0.00% / 0.00% |
| all 20,000 | exact field | 99.99% | |

The two misses (0.014 mm high) and one continuity break of the same size sit where the
probe-chosen formula in solid #46246 is invalid: the exact field inherits that formula's wrong
zero set. Reading from the leaf upward costs 1.69 cells per point on average. The harness
evaluates 55,000 points per second on 12 threads, about 8 times slower than the B-rep reference,
because each face test walks the glued field again; nothing is optimized yet.

## The pure field and the cell certificate (2026-10-02, a proposal for Derek's review)

**Construction.** One CSG formula for the solid: F(x) = minᵢ max(|x − cᵢ| − rᵢ, φᵢ(x)), the
union over cells of (cell ball ∩ cell formula). The pure field is E(x) = sign F(x) times the
distance to the nearest boundary point among the critical points on the exact surfaces, on the
exact curves where two surfaces meet, and at points where three meet. Curves and points are
solved from the surfaces (Newton from the B-rep's edge and vertex as a start); a candidate counts
when F takes both signs just off it. No trim loop, seam, adjacency or B-rep curve geometry is
read at evaluation.

**Certificate.** Every cell formula is checked before the cell is accepted, and refused cells
split:

1. A curved surface may cross the ball in one sheet only: the ball is narrower than the spacing
   between separate sheets (a cylinder's or sphere's radius, a torus's tube or hole).
2. If a surface crosses the ball but its face does not reach it, the surface's point nearest the
   cell must lie near an edge of that face. Elsewhere it is another part of the surface.
3. Sampled: on each surface in the ball, the formula shows a boundary exactly where the B-rep
   reference finds the point itself on the part.

Condition 2 caught the shaft cover's failing tangent-edge rule. A ball inside a concave fillet's tube
reached only the tube's far wall, and the rule showed a sliver of it. The sliver was thinner than
the sample spacing, so condition 3 missed it. Condition 3 first used the file tolerance for "on
the part", which refused correct formulas next to tangent edges (191,490 refusals on the shaft cover)
and was tightened to the point itself.

**Measured** (20,000 points per part; support 1.25 cell radii):

| Part | Matches the reference | Breaks | Sign errors | Gradient off by more than 1° within 1 mm |
| --- | --- | --- | --- | --- |
| Pipe coupling | 100.00% | 0 | 0 | 0.00% |
| Shaft cover | 99.98% (worst 0.0007 mm) | 0 | 0 | 0.00% |
| Test nest (41 solids) | 99.03% (worst 0.080 mm) | 0 | 0 | 0.00% |

**Corrected 2026-10-03 (measurements; the proposal is still not adopted).** The verdicts first
given here used a test offset of 1e-5 × (1 + |p|), 0.04 mm at the test nest's coordinates, too
coarse to tell the fields apart. The harness pure field's own exposure test had the same flaw at
1e-8 × (1 + |p|): 0.000042 mm there, wide enough to accept a cylinder's continuation just past a
tangent edge (distance short by up to 0.0022 mm on solid #46276). Both offsets are now four times
the point's distance from its own surfaces, at least 1e-9 mm. After the fix, on the test nest's
20,000 points, the pure field differs from the reference at 156:

- 154: the reference's nearest point is on a stored B-spline edge curve. Those curves sit up to
  0.0056 mm off the two surfaces they join (file tolerance 0.005 mm); the pure field measures to
  where the surfaces meet.
- 1: the reference's trim test returned "unsure" beside a hole's edge and skipped the face.
- 1: the pure field is wrong. On solid #46246 a probe-chosen leaf formula, Sel{s, P(22), P(22)},
  leaves out a cylinder that meets cylinder 22 tangentially, so F is negative in a sliver up to
  0.003 mm thick outside the part. The sampled certificate missed the sliver.

**Defect of the proposal, for Derek's review:** in the harness, F is assembled from frep3.rs's
compiled leaves, 341,218 of which on the test nest are probe-chosen, a route decision 2 rules out
(reversed 2026-10-02). The sliver above comes from one of them. An exact F needs leaf formulas
that do not come from probes.

On 3D grids (icam-check volume) the
pure field equals the same formula read over every surface with no cells at every point of five
solids. It equals the exact field and the reference at every point of the shaft cover (504,000), the
pipe coupling (2,646,000) and the test nest's solid #46270, and of solid #46276 except 12 reference trim
failures. On solid #46255 it differs from both at 115,946 of 343,040 points, all at the
off-surface B-spline edges above, worst 0.00559 mm.

**Open:** cells left with no formula at the depth limit (141 on the test nest), which bypass the
certificate; condition 3 is sampled, not proved; the pure field evaluates at about 3,000 points
per second on the test nest in the harness.

## Costs and open points

- **Every remaining failure is a premise failure,** as the theorem says it must be: a
  probe-chosen formula, a vertex rule used far from its vertex, or a select used beyond its
  agreement region. Each one is a local question about one support.
- **The one break left at support 1.25.** Values 0.0384 and −0.0888 mm at points 0.1252 mm
  apart. F changes sign along the segment, so the boundary crosses it and |sd| at the two ends
  sums to at most 0.1252 mm. Some contribution therefore exceeds the distance by at least
  0.002 mm. The endpoint that is a sample point has no wrong contribution, so the overshoot is at
  the other end: a probe-chosen formula that leaves out a nearby face.
- **Evaluation cost.** A point reads every support that holds it: 5.1 on average at support
  1.25, 19.1 when rule formulas keep the whole reach ball. The harness walks the octree for each
  point (0.11 to 0.45 million evaluations per second on 12 threads, against 9.7 million for one
  leaf per point). In production each leaf would store the list of supports that meet it (item 4
  of the theorem), so a point costs one leaf lookup plus about five formula evaluations.
- **Exactness against support size.** 36.4% today; 39.8% to 46.4% glued. Larger supports carry
  exact values further from the surface, but only certified formulas may have them.
- **Not yet measured:** the other six test parts, and the B-spline and cone primitives (frep3
  refuses those faces today).
- **For Derek: does this meet the 2026-09-29 rule?** The field is one function, and a cell's
  pruned tree is its exact restriction, so neighboring cells agree on their shared boundary by
  the theorem rather than by luck. It is not one CSG formula over the whole part; it is local
  formulas joined by one global rule.

## Proposed next steps

1. Glue in `frep3.rs` itself: support 1.25 cell radii, selects in their min/max form, a per-leaf
   list of overlapping supports. About half a day.
2. Replace probes with the local certificate above, and give mixed vertices the polygon formula.
   Certified formulas then get the whole reach ball back. About two to three days.
3. Add cones and B-spline faces as primitives (signed distance to the extended patch). About two
   days, then the other six test parts.
