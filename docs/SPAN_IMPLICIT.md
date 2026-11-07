# Span implicits: the exact field of one B-spline span, measured

*Analysis and measurements, 2026-10-25 (Claude, for Derek's review). The construction in
section 5 is a proposal and is not adopted.*

**Withdrawn, 2026-10-25 (Derek: "I don't recall saying we were allowed to get loose with the
definitions in that way").** Earlier versions of this document offered the floor-based matrix
(near-syzygies at 1e-8 read as syzygies) as the usable reading of a fitted span. It is not the
span's surface: it is the analytic surface the span was fitted to, matched to about 1e-9, and
presenting it as exact is the loose definition the guarantees forbid. Every passage below that
used it as a field reading is marked withdrawn. The floor remains only as a measurement in
`icam-check`. The definition of exact for decimal input is settled in section 8 (Derek,
2026-10-25).

The literature research behind this, done the same day and not published, found that
an untrimmed rational span has an exact implicit field,
that field is necessarily the span's whole extended algebraic surface, and trimming has to come
from a non-analytic composition whose separators are an open problem for curved faces. This
document is the first measurement of that primitive on real CAD data: `icam_core::mrep`
(Busé's implicit matrix representation of a rational Bézier patch), `icam-check spans` on THE
PART (776 B-spline faces, 36,856 spans) and `icam-check spanlab` on synthetic spans with a
known truth.

## 1. Verdict against the four guarantees, for the span primitive alone

1. **Fast and parallel.** The field of one span is M(p) = M0 + x M1 + y M2 + z M3 over four
   stored 18 x 18 matrices (a bicubic; 6 x 6 for a (1,3) span) and one determinant by LU, a
   fixed arithmetic kernel per point. Measured on one CPU thread in f64, allocating per call:
   167,000 points/s (6.0 us). The singular-value form (Jacobi) is 4,700 points/s and is a
   measurement tool, not the kernel. No GPU version yet (decision 9 asks for one; the kernel is
   straight-line and small).
2. **Exact.** The determinant vanishes on the span's algebraic surface and changes sign across
   it: on a generic bicubic the value on the patch is under 1e-9 of the value 2% of the span
   away, and the moving-plane residual is 1e-15 (unit tests `mrep::tests`). That holds for a
   generic span, whose 18 moving planes are separated from the rest by a wide gap. It does not
   hold for a fitted span (section 3): its 18 directions computed in double precision are
   mixed with near-null ones at about 1e-16 over the 19th singular value, about 2e-5 on THE
   PART, so the computed square matrix is not the bicubic's exact surface either. The rank-drop
   figures on THE PART (1.5e-12 to 1.2e-9) were taken at the withdrawn 1e-8 floor.
3. **No holes.** The square matrix takes exactly 2 d1 d2 directions and decides nothing by a
   threshold. The 1e-8 floor this item used to describe decided which near-syzygies count as
   syzygies: a threshold that decides structure, and the loose reading withdrawn above.
4. **No B-rep logic.** The construction reads the span's control net and nothing else: no
   edges, no trims, no neighbors, no intersections.

The square primitive meets 1 and 4, and 2 and 3 for generic spans only. It is a leaf, not a
field for a face: the trimming and the fences are section 5 and are not built.

## 2. What was built

- `crates/icam-core/src/mrep.rs`: `MRep::of_patch` (floor 1e-8), `MRep::square` (exactly
  2 d1 d2 directions, no floor), `det`, `svd`, `sigma_min`, `delta`, `invert` (the Bernstein
  basis read off the left null vector), `basis`. One-sided Jacobi for the null space and the
  singular values; coordinates normalized to the unit ball about the span's box center. Tests:
  a bilinear saddle (a quadric, 2 x 2), a sphere octant with base points (more planes than
  rows), a generic bicubic (18 x 18, clean spectrum), the determinant's zero and sign change,
  the growth of the smallest singular value with distance.
- `crates/icam-check/src/spans.rs`: `icam-check spans <file.step> [n]` measures every span
  (or an even subsample of n): the build, the spectrum, the rank drop on the patch, second
  sheets through the patch (corank 2), zeros in the padded box away from the patch, and real
  sheets of the exact surface by sign changes of the square determinant along 75 lines through
  the padded box, each sorted into the patch, the span's own sheet continued past its edges
  (a real preimage within half a span, or within three spans), or another sheet.
  `icam-check spanlab` runs the same on synthetic spans with a known truth.

## 3. Fitted CAD spans have no exact degree

The moving-plane system of a generic bicubic has exactly 18 null directions and then a gap of
twelve orders of magnitude. The bicubic spans of THE PART do not look like that. Four spans of
face #28623, singular values relative to the largest, ascending, after the 18 exact zeros:

    6e-12 5e-10 1e-9 2e-8 3e-8 6e-8 9e-7 2e-6 7e-6 2e-5 4e-5 6e-5 7e-5 9e-5
    6e-11 4e-10 2e-9 6e-9 2e-8 1e-7 6e-7 2e-6 8e-6 1e-5 5e-5 6e-5 8e-5 9e-5

A continuum at every scale. These spans are B-spline fits of analytic blend surfaces (the
file is a HOOPS export; Parasolid blends are rolling-ball surfaces, written out as bicubic
approximations), and a surface that is a cylinder, torus or sweep to 1e-8 of its size has
moving planes to 1e-8 that are not moving planes exactly. In the 300-span sample, 230 of 232
bicubic spans have no clear gap at the floor; at floor 1e-8 they take 21 to 30 columns, at
floor 1e-11 18 to 26. The lab reproduces this: a bicubic least-squares fit of a 60 degree
cylinder patch with control-point noise of 1e-8 of its size has its 19th direction at 1.9e-12,
with noise 1e-6 at 2.7e-10, with noise 1e-4 at 1.2e-7.

So "the exact implicit of this span" has two readings at double precision: the degree-18
surface of the bicubic as written (the square matrix), or the lower-degree surface the bicubic
was fitted to (the floor-based matrix, which treats near-syzygies at the floor as syzygies;
withdrawn as a field reading: it is not the span's surface).
They agree on the patch to 1e-9 relative and disagree away from it.

## 4. The exact degree-18 surface is not usable away from the patch

The square matrix's determinant is the exact implicit equation of the bicubic. Sign changes of
it along lines through the padded box of each span, 300-span sample of THE PART:

| crossings sorted by what they cross | count |
| --- | --- |
| the patch | 8,543 |
| the span's own sheet, continued within half a span past its edges | 4,826 |
| the span's own sheet, from parameters 0.5 to 3 spans out | 2,996 |
| another sheet | 6,158 |
| spans with another sheet inside their padded box | 154 of 300 |
| nearest other sheet to the patch, in span sizes: min / p10 / p50 / p90 | 0.020 / 0.022 / 0.056 / 0.645 |

The lab explains what those are. With no noise the fits are exactly degenerate surfaces
(the cylinder fit is an exact extrusion of a cubic, the torus and sphere fits exact products
of univariate fits), their null space is 27 to 39 wide at 1e-16, and 18 of those directions
give a determinant with extraneous factors: 54 to 111 "other" crossings that are artifacts.
With noise of 1e-6 or more the surfaces are generic, the determinant is exact, and there are
no other sheets in the box at all (cylinder, torus and sphere alike, at 1e-6, 1e-5, 1e-4,
1e-3, 1e-2). At noise 1e-8, the regime THE PART sits in, the sphere fit shows 51 other
crossings at 0.023 span sizes from the patch where the floor-based matrix does not drop rank
(ratio 0.11), the torus fit none. That is the ill-conditioning of a near-degenerate variety:
its extra real sheets are placed by roundoff-level detail of the coefficients, and a sheet 2%
of the span away is present or absent depending on it. Nothing in double precision settles
which, and nothing in the field could trust it. (The full run reports, at each other-sheet
point of THE PART, whether the floor-based matrix drops rank there; section 6.)

Two consequences for the fence idea:

- A box or hull fence around a span, with subdivision, assumes the extended surface stays out
  of the box except through the patch. For the exact degree-18 surface of a fitted span that
  assumption fails for half the spans of THE PART and cannot be made good by subdivision (the
  surface is the same; the twin sheet sits 2% of the span from the patch).
- *Withdrawn (see the note at the top).* This bullet called the floor-based matrix "the
  reading that behaves". Its zero set is the fitted analytic surface to 1e-9 relative, not the
  bicubic as written, so it is not an exact field of the span. The measurements stand as
  measurements: in the lab it has no other sheets in the box for noise 1e-6 and up, and on
  THE PART its zeros away from the patch are 479 of 102,900 box samples in 73 of 300 spans.

## 5. Proposal: what a span leaf would be (not adopted)

*Proposal, 2026-10-25 (Claude, for Derek's review). Not adopted.* Verdict against the four
guarantees first: the leaf it named was the floor-based matrix, which is withdrawn (see the
note at the top), so this proposal has no exact leaf for fitted spans; its separators are the
open part, and the measurements above say a box fence is not one.

- *Withdrawn:* "the leaf of a B-spline face is one floor-based M-rep per span". Fitted spans
  have no exact signed leaf in this document.
- The span's extent is not a box. The candidates that are exact along the span's own
  boundary are the ruled surfaces of its normals along its four knot lines (bidegree (5,1)
  for a bicubic, implicitizable the same way, degree at most 10); where two spans of one face
  meet they share that surface exactly, since both contain the knot line and are tangent along
  it. That is the construction Shapiro and Vossler's separation problem asks for, written for
  the one case a face's own structure gives for free. It is valid inside the reach of the
  span (before the normals cross) and that reach is a measurable number per span.
- Faces meeting at an angle need no separator: the neighbor's leaf is one, by the edge's
  convexity label (max or min), which is topology.
- Not covered and not proposed: tangent joins between faces (fillet to wall), seams and poles
  of periodic surfaces, and what to do past the reach. Those are where the literature stops.

## 6. The full run

`icam-check spans` over all 36,856 spans of THE PART, 2,621 s on 12 threads (the builds alone
are 114 ms each, 70 CPU-minutes; the rest is the measurement). The binary of this run predates
the floor-based rank check at other-sheet points, so that column is absent here.

| | |
| --- | --- |
| spans by degree | 28,522 bicubic, 8,310 of bidegree (1,3), 24 of (3,1); none refused |
| bicubic spans with no clear gap at the floor | 28,468 of 28,522 |
| columns taken at floor 1e-8, bicubic | 21 to 31 (mode 24: 7,802 spans); (1,3): 6 to 12 |
| moving-plane residual | p50 2.5e-9, max 1.1e-8 |
| rank drop on the patch, worst sample per span | p50 1.2e-12, p99 5.8e-10, max 3.0e-9 |
| spans whose own extended surface crosses the patch again (corank 2, a second real preimage found by Newton) | 40 (267 flagged by the singular-value ratio) |
| floor-based M-rep: box samples at a zero over 2% of the span from the patch | 68,409 of 12,641,608; 9,385 spans with one |
| square determinant: spans with another real sheet of the degree-18 surface in their padded box | 17,620 of 36,856 |
| nearest other sheet to the patch, span sizes | min 0.020, p10 0.022, p50 0.042, p90 0.509 |
| singular-value kernel, one thread | 11,671 points/s |

The sample's picture holds at full scale. Two numbers sharpen it. First, 40 spans are crossed
by their own extended surface (the sheet reached from parameters outside the span's square
passes through the patch itself): there no fence of any shape separates the span from its
extension, only a parameter-domain condition would, and that is a trim test. Second, the
floor-based matrix still has zeros away from the patch in a quarter of the spans by the
singular-value test; whether those are zeros of the fitted analytic surface's own extension
(a cylinder's far side, a torus's inner ring, which the sorting by real preimage within three
spans does not reach), or near-degeneracy noise in the singular-value test, or sheets, the
run as built cannot say. The 300-span rerun with the rank check at other-sheet points is the
next measurement, and a sign-carrying evaluator for the non-square matrix is the tool missing
for it.

## 7. The STEP Tools suite

`icam-check spans test-parts/steptools 40`: the 55 STEP Tools AP203 sample files kept in
`test-parts/` (see its README). Ten carry B-spline faces: 12,653 spans, 3,136 of them rational,
in 17 bidegrees from (1,1) to (5,11), written by CADDS 5, Pro/E, CATIA, ACIS and Unigraphics.
Forty spans per file, an even subsample, 369 in all; 12 minutes, 709 s of it the 21 measured
(5,11) spans of st203-bapl, whose moving-plane systems take up to 265 columns.

| | |
| --- | --- |
| spans refused | 0 of 369 |
| rank drop on the patch, worst sample per span | p50 1.1e-14, p99 5.5e-11, max 9.6e-11 |
| spans crossed by their own extended surface | 0 |
| spans with no clear null-space gap | 0 of 194 at bidegree (2,2) and below; 81 of 120 bicubic; 21 of 21 at (5,11); 3 of 4 at (5,5) |

| File | Writer | Spans | Measured | No gap | Most columns | Square det: spans with another sheet in the box | Floor M-rep ratio there, median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bull | CADDS 5 | 69 | 40 | 0 | 12 | 25 | 1.7e-1 |
| filler | CADDS 5 | 33 | 33 | 0 | 12 | 27 | 1.3e-1 |
| tork | CADDS 5 | 1,955 | 40 | 6 | 88 | 25 | 9.6e-2 |
| piston | | 348 | 40 | 0 | 12 | 33 | 8.0e-2 |
| moon_buggy | | 346 | 40 | 2 | 30 | 34 | 9.1e-2 |
| moon_buggy_asm | Pro/E | 16 | 16 | 0 | 36 | 7 | 2.3e-1 |
| rear | Pro/E | 7,600 | 40 | 39 | 37 | 10 | 8.0e-4 |
| jack_in_the_box | ACIS | 520 | 40 | 39 | 23 | 5 | 1.4e-5 |
| ph4m3-st | ACIS | 611 | 40 | 2 | 26 | 15 | 2.9e-14 |
| st203-bapl | CATIA | 1,155 | 40 | 29 | 265 | 33 | 1.0e-3 |

Three things the suite adds to THE PART:

1. **The construction holds across writers and degrees.** No refusals, the matrix drops rank
   on every patch to 1e-10 relative, rational spans and spans with base points included, up to
   110 rows at (5,11).
2. **There are two kinds of span, and they want different kernels.** Designed spans of low
   degree (the CADDS 5 files, piston, moon_buggy and moon_buggy_asm, most of them rational
   bidegree (2,2) and below) have a
   clean null-space gap, but often more columns than rows: exact conics, tori and spheres
   written as rational patches, whose implicit degree is below 2 d1 d2, or base points. For
   them the floor-based matrix reads the lower-degree surface, and the square determinant
   takes m of the r directions and picks up extraneous factors. Whether the lower-degree
   surface is the span's was settled in section 8: a sphere written with any weight other than
   the irrational square root of a half is not exactly a sphere, and is read as one when it
   differs from it only by rounding. The "other sheets" it shows in
   their boxes are those factors; the exact M-rep does not drop rank there (ratio 0.08 to 0.4,
   against 1e-14 on the patch). Fitted spans (ACIS jack_in_the_box, Pro/E rear, CATIA
   st203-bapl, the bicubics and the degree-5 spans) have no gap, as on THE PART.
3. **The sign is the open part, not the zero set.** The exact M-rep of a span with more columns
   than rows has no determinant; its evaluation function (the product of singular values, or
   det(M M^T)) is a sum of squares and carries no sign. So for the very spans where the matrix
   is exact without any floor, the field does not yet have an inside and an outside. For those
   that are exact conics, tori and spheres, the analytic surface the importer already reads
   (`Surf::Sphere`, `Surf::Torus`) is the signed kernel; recognizing a rational patch as one is
   an import-time test on the control net, not on the B-rep. For base points in general, a
   square matrix at a lower moving-plane degree (Busé §3.2 notes the critical degree drops by
   the degree of a curve through the base points) or moving quadrics (Sederberg-Chen) are the
   candidates. Neither is built.

True extra sheets of an exact surface inside a span's box do occur: the spans where the floor
M-rep also vanishes at the square determinant's crossings, bidegree (2,3) and (3,2) of ph4m3-st
(ratio 7e-14 and 2e-17) and the (5,1) spans of st203-bapl (6e-11), nearest 2% of the span size
from the patch. The 116 spans whose floor M-rep has zeros over 2% of the span from the patch
mix those with the exact surfaces' own far side (a rational quarter torus reaches the rest of
its torus only through parameters far outside the span, beyond the three spans the sorting
checks), which this run cannot separate.

Kernel cost grows with degree: the (5,11) matrix is 110 x 110 against 18 x 18 for a bicubic,
about 220 times the arithmetic per determinant.

## 8. Analytic surfaces read from B-spline faces

**Decision, 2026-10-25 (Derek).** A span that differs from an analytic surface only by the
rounding of its numbers is that surface, and its field is that surface's: a sphere's is the
distance from its center less its radius ("Can we treat a span as a perfect sphere when the only
difference is rounding in the file's numbers?" "yes"). The line between rounding and a fit is
3e-9 of the span's size ("That's approximately where every CAD kernel tries to build their
surfaces to, but that can be checked online"), with the tightest published kernel resolution as
a second, absolute ceiling.

Why the definition was needed: no rational sphere is exactly a sphere in a file or in memory,
because the square root of a half has no finite decimal or binary form. The exact surface of the
written numbers is a quartic that double precision does not pin down (its second least singular
value is within a factor of 2 of the least), and its candidates put false sheets a thousandth of a
radius inside the sphere. Writers' numbers carry more than printing: tork writes 3.78 as
3.779999999999759 and zero as 2.220446E-16; moon_buggy writes cos 30 degrees correct to 8 digits.

### Verdict against the four guarantees

1. **Fast and parallel.** The leaf is the analytic primitive the compiler already builds for
   STEP-native faces (`Prim::Sphere` and the rest), a closed form per point. The exact B-rep
   reference, which is what changed here, went from 4 to 47,849 distances a second on bull, 5 to
   4,384 on piston and 12 to 18,671 on tork (12 threads; table below).
2. **Exact.** Every face read as an analytic surface is within 3e-9 of each span's size and
   1e-8 mm of it, by a certified bound over the whole span (the surface's implicit equation
   composed with the span is a Bernstein polynomial; its largest coefficient bounds it; no
   sampling). Against the raw B-spline reading, the exact distance agrees to 1.4e-13 of the part
   size, with no inside or outside reading changed beyond the file tolerance.
3. **No holes.** One decision per face at import, the line above, which Derek set; it sits in a
   decade that no span of 49,509 occupies and is stricter than every published kernel resolution.
   A face that fails stays a B-spline face; nothing is refused.
4. **No B-rep logic.** Recognition reads the face's own control net and nothing else. The
   orientation comes from the span's Su x Sv and the face's sense flag.

### Measurements

Distance from each B-spline span to its nearest plane, sphere, circular cylinder or circular
cone, all 49,509 spans of the STEP Tools suite and THE PART (`icam-check spanshape`, raw
reading):

| Relative to the span's size | Spans |
| --- | --- |
| 1e-17 to 1e-9 | 1,158 (bull, filler, tork, piston, moon_buggy, ph4m3-st) |
| 1e-9 to 1e-8 | 0 |
| 1e-8 to 1e-6 | 36 (rear, st203-bapl, ph4m3-st) |
| 1e-6 and up | 48,315 |

| Absolute line | Source | Spans within it |
| --- | --- | --- |
| 1e-7 mm | Open CASCADE Precision::Confusion | 1,160 |
| 1e-6 mm | ACIS SPAresabs (model units, normally mm) | 1,181 |
| 1e-5 mm | Parasolid linear resolution, 1e-8 m | 1,248 (32 of them THE PART's fits) |
| 1e-3 mm | CATIA version 5 modeler resolution | 4,372 |

Every span the line accepts is within 1e-8 mm of its surface, 10 times inside the tightest of
these.

The importer with recognition against the raw reading (`icam-check analytic`, random points of
each solid's padded box, both readings interleaved):

| File | B-spline faces read as analytic | Distance difference (part size) | Reference distances/s, raw to analytic |
| --- | --- | --- | --- |
| bull | 34 of 36 (15 cylinders, 19 planes) | 2.5e-14 | 4 to 47,849 |
| tork | 421 of 529 (264 cylinders, 146 planes, 6 cones, 5 spheres) | 1.2e-13 | 12 to 18,671 |
| piston | 44 of 66 (22 cylinders, 15 planes, 7 cones) | 1.4e-13 | 5 to 4,384 |
| moon_buggy | 50 of 77 (28 planes, 14 cylinders, 8 spheres) | 1.2e-14 | 15 to 23 |
| filler | 17 of 19 | 4.8e-15 | 4 to 4 |
| st203-bapl | 4 of 104 (planes) | 0 | 3 to 4 |
| THE PART | 0 of 776 | none | none |

On moon_buggy the raw reading put two points outside the solid's own box inside it (a sign
error at a vertex of the B-spline reading, the file's known Lipschitz failure); the analytic
reading has them outside.

Labels: ISO 10303-42 gives a B-spline surface a `surface_form`. In this set only filler, piston
and tork write QUADRIC_SURF (no kind, center or radius) and jack_in_the_box one TOROIDAL_SURF;
everything else, THE PART included, writes UNSPECIFIED. The numbers are the only source.

### Not covered yet

- Tori (degree 4: bull's 14 spans, most of piston's), and the quadrics that are not a sphere,
  circular cylinder or cone: elliptic, hyperbolic and parabolic cylinders in the suite, and THE
  PART's 16 exact spans (8 hyperboloids of one sheet, 8 hyperbolic paraboloids). Their
  equations are exact; their leaves are not built.
- Cones whose span reaches the apex: the certified bound needs a positive height above it.
- Fitted spans: all 776 B-spline faces of THE PART. No exact signed leaf exists for them in this
  document.

### Withdrawn

The far-out test of sections 7 and the span polynomial (a candidate must vanish 128 spans out
along the span's extension): evaluated that far, the residual's mixed terms amplify the
arithmetic's rounding by about 1e12, so exact spans of real files fail it (tork's chamfer plane
scored 2.4e-4 while lying on its plane to 1.4e-10). Recognition does not use it.

Sources: [Open CASCADE Precision](https://dev.opencascade.org/doc/occt-7.8.0/refman/html/classPrecision.html),
[ACIS tolerance variables](http://www.q-solid.com/ACIS_Docs_R17/online/SPAacisuserTechArticles/SPAacisuser_totol.htm),
[Parasolid default precision, Bentley](https://DOCS.BENTLEY.COM/LiveContent/web/MicroStation%20Help-v25/en/AdvancedUnitSettings.html),
[CATIA version 5 modeler resolution, IBM HD91883](https://www.ibm.com/support/pages/apar/HD91883).

## 9. Tori and the other quadrics

*2026-10-26 (Derek: "Yeah, let's fill those in").* Same standard as section 8: a face is the
surface when every span is within 3e-9 of its size and 1e-7 mm of it, by a certified bound.

### Tori

The axis of a surface of revolution is the line every normal line meets, a linear condition on
the axis's Plücker coordinates; the tube is the circle the span's points make in the meridian
half-plane. Both are read off the first span's points and certified on every span: the torus
equation factors as (s - r)(s + r)((rho + R)^2 + h^2 - r^2), s the distance to the tube's center
circle, so for a ring torus the distance |s - r| is at most |G| / (r (R^2 - r^2)). Tori are a STEP
kind, so the importer turns these faces into torus faces and the compiler's torus primitive takes
them unchanged.

| File | Torus faces | All B-spline faces now analytic |
| --- | --- | --- |
| bull | 2 | 36 of 36 |
| filler | 2 | 19 of 19 |
| moon_buggy | 18 | 68 of 77 |
| piston | 20 | 64 of 66 |
| tork | 63 | 484 of 529 |

Across the suite 675 of 977 B-spline faces are now read as planes, spheres, cylinders, cones and
tori; the exact distance of both readings agrees to 1.4e-13 of the part size with no inside or
outside reading changed.

### Ellipsoids, hyperboloids, paraboloids and the noncircular cylinders

`crates/icam-core/src/quadric.rs` keeps them in canonical form in a local frame,
G = q0 u^2 + q1 v^2 + q2 w^2 + l w + c, and gives their exact signed distance: the foot solves the
Lagrange condition, a polynomial of degree at most 6 in one unknown whose real roots are all
found (between consecutive roots of its derivative a polynomial is monotone), plus the points
where a denominator vanishes. Tests compare it with a dense sampling of each kind's
parameterization refined around its best point: it is never farther and never more than 1e-6
short, on ellipsoids, hyperboloids of one sheet, both paraboloids, and elliptic, hyperbolic and
parabolic cylinders.

Recognition reads candidates off the first span's degree-2 polynomial in its eigenframe, with
two, one or no eigenvalues set to zero and the leftover linear part kept or dropped (cylinders
before paraboloids before the central kinds), and certifies each on every span: with |G| <= g on a
span and |grad G| >= gamma over its box, the surface is within 2 g / gamma when
4 |A| g <= gamma^2. Where a box holds the quadric's center the gradient bound fails, and the span
is certified quarter by quarter, to four levels. Two arithmetic facts enter every bound: a
candidate whose frame origin is more than 3e-9 / epsilon span sizes from the span is not
representable at the line, and working in a frame |o| span sizes away adds 16 epsilon (1 + |o|) to
the bound.

These faces stay B-spline faces for the B-rep reference's trims and carry the quadric
(`Face3::quadric`), oriented so G grows along the B-spline's Su x Sv.

| File | Faces read as quadrics | Exact distance against the B-spline's, 1e-3 of a span off |
| --- | --- | --- |
| jack_in_the_box | 8 elliptic cylinders | 2.9e-13 mm |
| moon_buggy | 1 elliptic, 1 hyperbolic, 1 parabolic cylinder | 1.8e-12 mm |
| moon_buggy_asm | the same three | 1.1e-11 mm |
| rear | 3 hyperbolic paraboloids | 2.3e-9 mm |

Found on the way: moon_buggy's face #625 is a hyperbolic cylinder close to its asymptotes, and a
rounding-sized linear term (3e-14) first read it as a paraboloid whose vertex was 6e17 mm away,
where its own coordinates held no digits; jack_in_the_box's elliptic cylinders were first read
as ellipsoids, paraboloids and hyperboloids 0.59 mm off, through the same far-origin rounding
fooling the bound. Trying the cylinder reading before the paraboloid one, the representability
limit and the rounding term in the bound removed all of them.

THE PART: no whole face is one quadric. 1,896 of its spans are quadrics on their own (1,090
hyperboloids of one sheet, 797 hyperbolic paraboloids, 9 hyperboloids of two sheets; 0.14 mm to
over 20 mm across; their exact distance agrees with the B-spline's to 2.4e-9 mm), inside 103
faces whose other spans are fitted; no face is a quadric in every span. Using them takes leaves
per span and the separators along knot lines of section 5.

### Verdict against the four guarantees, for the quadric primitive

1. **Fast and parallel.** G, which carries the sign and the zero set, is a degree-2 polynomial:
   155 to 232 million points a second on one thread in f64. The exact distance is a per-point
   computation over independent points, 74,000 to 159,000 a second on one thread (6 to 14 us),
   but not straight-line: the root solve's steps depend on the point. A fixed-step GPU form is not
   built.
2. **Exact.** The foot is the nearest critical point among all of them, checked against dense
   sampling; recognized faces agree with the B-spline's own distance to 2.3e-9 mm or better.
3. **No holes.** The decision is section 8's line; a face that fails stays a B-spline face.
4. **No B-rep logic.** Recognition reads the face's control net; the primitive reads its own
   coefficients.

### Checks

- The importer's corpus check: 49 of 55 files pass, as before, and no file changes status. Its
  pair rate on the exact reference: filler 4.6 to 12,398 a second, tork 0.4 to 1,265 (tori
  included), bull 20,447, piston 5,129.
- The compiler's own check (`icam-check frep3`, 2,000 points): filler, every face now analytic,
  has no sign error and its exact field is exact at all of them. bull has 10 sign errors in the
  per-cell field, 9 in 147 cells left without a formula at the depth limit (the compiler refused
  1,495 formulas for two sheets of one surface in a cell, a rule it applies to analytic
  primitives and not to B-spline faces) and 1 in a probed cell. Whether reading bull's two torus
  faces as tori caused those cells is not settled: the same check on the build before
  recognition had not finished after more than 40 minutes, because that compile calls the
  B-spline reference, and was stopped. piston did not finish within 900 s.

### Not wired, and still not covered

- The compiler has no quadric primitive. Adding one edits `frep3.rs`, which the guard allows only
  in a commit that lowers the counts it is slated for: that is the compiler's replacement, and the
  primitive here is ready for it. Tori need nothing; they are torus faces.
- Cones whose span reaches the apex, apple and lemon tori written as B-splines, and fitted
  spans: all 776 B-spline faces of THE PART stay B-spline faces.

## 10. Checking without the B-rep reference

*2026-10-26 (Derek: "We gotta find a way to evaluate these things that doesn't rely on the
legacy brep path if it's gonna be this painfully slow", and "I've always hated the over
reliance on regression tests when sometimes the code we had before was bad").*

The slow code in every check so far was one function, the exact distance to the trimmed B-rep
(`Solid3::sdf`): per query point, the nearest point on each nearby face's surface (a branch and
bound over a B-spline face's Bezier patches), a trim walk to decide whether that point is on the
face, and the edges and vertices when it is not. It runs at 4 to 12 points a second on files
with B-spline faces and 9 on THE PART, and it is not ground truth: it read two points outside
moon_buggy's own box as inside. The legacy compiler calls it too, for cell signs and its sampled
certificate, which is why its compiles take minutes.

`icam-check surfcheck` checks a field against its solid's exact surfaces instead. Reference data
is made once per face: points on the face's exact surface, kept when the face's own trim test
calls them inside and they lie a tenth of the face's size clear of its edges, each with its
outward normal and an offset delta under a quarter of its distance to the edges, a quarter of
its least radius of curvature and 1e-4 of the solid's box diagonal. A field is then checked by
evaluating it alone: zero on the surface, plus and minus delta at the offsets, and, with no
reference at all, the 1-Lipschitz bound on random pairs and a positive value outside the box.

| Check | Old way | Surface samples |
| --- | --- | --- |
| The importer's readings, the whole STEP Tools suite | minutes per file through the B-rep reference | 1 min 51 s for all 55 files, 18,410 samples of converted faces |
| THE PART | 9 reference distances a second | 29,217 samples in 20.5 s, 44 s with both reads |
| A compiled field, bull | the reference at every point | the check under a second; the 20.6 s compile is the legacy compiler |

Every face the importer reads as a plane, sphere, cylinder, cone, torus or other quadric is exact
at every sample of the B-spline face it was read from: on the surface within 1.5e-10 mm, at plus
and minus delta within 1.5e-10 mm of the exact distance, no sign wrong.

The legacy per-cell field (`SolidFrep::eval`, kept as a baseline): filler exact at all 768
samples; bull exact at 99.1%, with 9 sign errors, all on plane faces (none of 96 samples on its
torus faces, none of 480 on its cylinders); both break the Lipschitz bound on 4 to 6% of random
pairs, the jumps between cells its own notes describe.

The sampling still runs each face's trim test once per sample: that is the reference data, made
once. The fields checked run on the CPU over all threads because the compiler has no GPU
evaluator; the checks themselves are a few field evaluations per point.

## 11. Where the local rules have no formula

*2026-10-26 (Derek: "it sounds like we need to create rules for holes or we brick the input
which sounds like it's not a robust import method. Discuss", then "start").*

A small enough ball about any point of a valid solid holds one face, one edge or one vertex, so
a field built from local rules has no holes once every edge and every vertex has a rule.
`icam-check gaps <file|dir>` counts where the compiler has none, from each solid's topology and
its face normals along the edges. Nothing is compiled and no distance is evaluated. An edge's
kind is the compiler's own test: tangent when the sine of the angle between the two normals is
under 1e-6, otherwise convex or concave by which way the second face turns. It is read at five
points along each edge and just off each vertex, on the importer's reading of the faces.

### Vertices

The compiler gives a formula to a vertex whose edges are all convex (max) or all concave (min).
Every other vertex goes to probes.

| Vertices | STEP Tools suite | THE PART |
| --- | --- | --- |
| All | 24,245 | 4,476 |
| Every edge one kind: a formula today | 7,537 (31%) | 962 (21%) |
| Three edges, convex and concave mixed | 4,864 (20%) | 653 (15%) |
| Three edges, one of them tangent | 8,744 (36%) | 1,857 (41%) |
| Every edge tangent | 1,475 (6%) | 322 (7%) |
| Four or more edges, convex and concave mixed | 525 (2%) | 35 (1%) |
| Any other vertex at a tangent edge | 1,100 (5%) | 647 (14%) |
| Faces not closing into one fan around the vertex | 0 | 0 |

The most common vertex without a formula is the end of a fillet: two convex edges and one
tangent edge, 6,829 in the suite and 1,683 in THE PART.

### Edges

| Edges | STEP Tools suite | THE PART |
| --- | --- | --- |
| All | 37,358 | 6,964 |
| Convex at the middle | 21,685 | 3,793 |
| Concave at the middle | 7,638 | 1,054 |
| Tangent at the middle | 8,016 | 2,105 |
| Normals opposite (knife) | 19 | 12 |
| Kind changes along the edge, convex to concave | 228 | 288 |
| Kind changes along the edge, through a tangent reading | 201 | 589 |

The tangent line sits inside THE PART's data. Its nominally tangent edges have normals that
agree to anything from 1e-3 to 1e-10. By decade of the sine at the edge's middle:

| Sine | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THE PART, edges | 52 | 385 | 414 | 136 | 92 | 283 | 634 | 248 |
| Suite, edges | 293 | 197 | 160 | 77 | 54 | 58 | 106 | 384 |

The 1e-6 line splits one population, and 877 of THE PART's edges change kind along their
length. A line that decides structure is what guarantee 3 forbids.

### Features too close to separate

| Closer than | Short edges | Vertex pairs not joined by an edge | Faces with two edges this close |
| --- | --- | --- | --- |
| The file's own tolerance, suite | 33 | 12 | 55 |
| The file's own tolerance, THE PART | 0 | 0 | 0 |
| A cell's reach at the compiler's floor, suite | 6,414 | 15,245 | 3,344 |
| A cell's reach at the compiler's floor, THE PART | 4 | 0 | 6 |

Lengths and gaps are measured at nine points along each edge. The features under a file's own
tolerance are in three files, clip, socks and st203-bapl. By the file's own declaration each is
one feature, so they are input defects, to be reported with their locations. The large counts at
the floor come from the floor: the compiler refines no further than ten times the declared
tolerance, and twelve suite files declare 0.03 to 0.8 mm, which puts their floor at 0.3 to 8 mm.
That is the compiler's choice, not a property of the parts.

### What this says about the rules

- **The gap is rules, not geometry.** 100 features in three files are closer than their own
  files allow; 16,708 of the suite's vertices and 3,514 of THE PART's have no rule.
- **One edge rule needs no kind.** Take each face's distance on its own side of the plane
  through the edge spanned by the edge's tangent and the two faces' mean normal. For planar
  faces that plane is where the two distances are equal, so the rule is exactly max at a convex
  edge and min at a concave one; at a tangent edge it is the compiler's tangent separator. It is
  continuous through tangency and has no threshold. It fails only at a knife edge, where the mean
  normal vanishes. For curved faces it departs from max and min by second-order terms near the
  edge, which surfcheck has to measure.
- **Two small vertex rules cover most of the rest.** A three-edge vertex is settled by its edge
  kinds: the faces of the odd edge combine first (min for a concave odd edge, max for a convex
  one), then the third face with the other operation. A vertex with one tangent edge splits on
  that edge's separator into two sides whose edges are not tangent. With today's rule these
  cover 87% of the suite's vertices and 78% of THE PART's.
- **The general vertex needs the polygon theorem** (Dobkin, Guibas, Hershberger and Snoeyink,
  1988: every simple polygon has a monotone min/max formula that uses each edge once): 13% of
  the suite's vertices and 22% of THE PART's, all-tangent vertices included.

## 12. Trim loops in the parameter domain

*2026-10-26 (Derek: the file's declared tolerance is the accuracy line at edges; the route is
exact pointwise distance to the exact surfaces with trims that cannot fail or run away).*

The per-cell rule work of section 11 stopped there: every route needs edge information
somewhere, and the trim test the importer used (`Solid3::classify`, paths walked across the
face with crossings counted, up to twelve retries, an undecided answer treated as outside) fails
the intent of guarantee 4 twice over. Measured while sampling faces: 15,548 undecided answers in
6.1 million tests on the suite, 21% on moon_buggy_asm, 382 on THE PART; and the trimmed
distance search gives up after 50,000 steps.

### The construction

The files carry no parameter-domain curves (PCURVE count zero in THE PART, moon_buggy_asm and
rear), so each B-spline face's loops are built at import, `crates/icam-core/src/trim2.rs`:

- Every exact piece of every edge is sampled, each sample's surface parameters come from the
  certified closest-point search, and a cubic is fitted through them in the domain, its ends
  shared exactly with its neighbors.
- A piece is kept only under a proof: with the span patch P/W and the piece c(t), the
  composition S(c(t)) is a polynomial in t whose Bernstein coefficients bound its distance from
  the edge e(t) over the whole piece. Where that bound is loose, both curves are split exactly
  and the halves bounded, to six levels. The line is the file's tolerance plus what the file
  itself gets wrong there: the edge's own offset from the surface (largest sampled, which never
  exceeds the true one) and the gap between consecutive edges at a vertex (the loop closes
  across it). A piece that fails is split and refitted, to twelve levels; a face that still fails
  is refused with the edge, the bound and the points.
- A piece crossing a knot line is split where it crosses, chosen among the places the samples
  change span by the largest excursion (a sample a rounding error past one line, next to a run of
  samples well past another, is not the crossing). The split point is the exact preimage of the
  3D split point and is never moved onto the line: each child holds it inside its own span and
  the certificate judges that. Where that holds a shared point in one child's span and not the
  other's, the loop opens by a rounding-sized gap; every such gap is closed by a straight
  segment, split at knot lines and certified against the one 3D point both ends stand for.
- Chains going once around a periodic direction close a period away; seams need no case.

The inside test is crossing parity, nothing else: pieces are split until monotone in both
parameters, and a point's answer is a known reference's answer flipped once per crossing of the
path between them (a vertical leg then a horizontal one), with the half-open rule at piece ends
so a corner on the path counts once. A tree over the domain holds in each cell a reference point
(off-center and off every dyadic fraction, so it does not sit on an edge along a knot line), its
answer settled from the parent's through a path inside the parent, and the pieces entering the
cell. The root's answer comes from outside the domain; on a doubly periodic domain, from a point
offset from one piece whose nearest loop point is proven to be that piece. A query is one descent
and a few bounded root solves.

### What was tried and withdrawn

- **The side of the nearest loop point** (tangent side on a piece, wedge at a corner) was the
  first inside test. It is exact everywhere except where it is not decidable: bull's face #713
  has a fillet meeting a line tangentially in the domain, a corner with a 180 degree turn whose
  side the rule read from a 1e-6 fit-noise component. Parity has no such decision.
- **Snapping a split point onto its knot line.** A root found on the fitted curve at a sample
  that sat on the line by rounding moved the point 1.3e-4 in u, and the children could never
  certify. Removed; the clamp inside each child plus the bridge does the job with a proof.
- **The B-rep distance as the judge of disagreements.** It uses the old trim test for its face
  feet, so it sides with the old test by construction. Replaced for the machinery by a plain
  polygon crossing count on the same loop (32 segments per piece, no tree, no monotone pieces,
  no root solves); disagreements with the old test are reported with their distance from the
  face's edges, and those within two tolerances are not judged.

### Results (`icam-check trim2`, 200 random parameters per B-spline face, raw reading)

| | STEP Tools suite | THE PART |
| --- | --- | --- |
| B-spline faces | 976 | 776 |
| Loops built | 950 | 776 |
| Refused | 26 (rear 6, tork 16, piston 2, moon_buggy 2) | 0 |
| Build, all faces | 16.5 s | 16.2 s |
| Pieces, cells | 60,246, 103,410 | 58,971, 109,336 |
| Largest certified deviation | 2.98e-2 mm | 7.44e-3 mm |
| Faces whose edges miss each other at a vertex by more than the file's tolerance | 45 | 0 |
| Tests | 190,000 | 155,200 |
| Tree against the polygon count on the same loop | 0 differ | 0 differ |
| Old test undecided | 50 | 0 |
| Old and new disagree | 2,210 (1,976 of them on jack_in_the_box) | 197 (74 within two tolerances of an edge) |
| Time per test, old / new | 172 us / 1.3 us | 186 us / 0.96 us |

On jack_in_the_box the two tests disagree at 52% of the points of its closed surfaces of
revolution; the B-rep distance, biased toward the old test, still calls the new one right at 88
of the 96 points it judged. The remaining refusals are at the floor of files declaring 1e-6 mm:
pieces whose certified deviation lands between 1.0e-6 and 1.2e-6, and bridges of 1e-9 in a
parameter that cost 1.1e-6 mm on surfaces with derivatives near 1,000 mm per unit parameter.

### Not done

- The old trim test is still what the B-rep distance (`Solid3::sdf`) uses; the certified loops
  are a second answer beside it. The exact pointwise evaluator that reads the loops, the exact
  surfaces and the edges is the next step, and then the old test and the per-cell compiler go.
- Analytic faces keep the old test; their loops in their own parameters are the same
  construction with closed-form preimages.
- Build time is the closest-point search, twelve per fitted piece: rear took 154 s before the
  crossing fix and 7 s after; a Newton continuation from the previous sample would cut it again.

## 13. Fixed Newton steps per brick, measured on THE PART

*2026-10-26 (Derek: "How bad is the branchy search honestly?", then "measure").*

A GPU kernel runs every point of a brick through the same instructions only if the nearest
point on a B-spline face comes from a fixed number of Newton steps, seeded once per brick. Fitted
spans have no closed-form nearest point, so this is the step that decides whether the evaluator
can be straight-line inside a brick. `icam-check newton` measures it against the certified search
(`NurbsSurf::closest`, exact to 1e-9 mm) on every B-spline face of THE PART: 12 bricks per face
and size, centered within a brick's size of a random point of the face inside its certified trim
loops, 32 random points per brick, 816,000 points in all. Bricks are 8 cells on a side, so the
sizes 0.5, 2 and 8 mm are the cache's cells of 1/16, 1/4 and 1 mm.

The step is a projected Newton step on the squared distance: the full Hessian where it is
positive definite, else Gauss-Newton; a parameter on its bound whose step points out of the
domain is held and the step taken in the other alone. A point counts as settled when one more
step would move it less than 1e-9 mm. The certified search was never beaten: no fixed-step answer
was ever nearer than it, so every difference below is a fixed-step error.

| Seeding, 6 steps | Brick 0.5 mm | Brick 2 mm | Brick 8 mm |
| --- | --- | --- | --- |
| One seed per face, exact to 1e-9 mm | 99.988% | 99.382% | 89.834% |
| One seed per face, settled but wrong | 0.003% | 0.184% | 4.807% |
| One seed per candidate span, exact to 1e-9 mm | 99.997% | 99.961% | 97.861% |
| One seed per candidate span, settled but wrong | 0.001%, worst 4.2e-4 mm | 0.001%, worst 9.9e-3 mm | 0.029%, worst 0.54 mm |
| Candidate spans per brick, mean and most | 6.9, 88 | 19.4, 150 | 34.4, 256 |
| Some candidate not settled | 0.25% | 3.5% | 19% |

A candidate span is one whose control-point box lies within the brick center's distance plus
the brick's diagonal, which every nearest point of a point in the brick satisfies.

Two things found on the way:

- **Clamping both parameters after a full two-parameter step leaves a false resting point.**
  With it, 8.2% of the points in 0.5 mm bricks settled wrong, nearly all on the domain's border.
  Holding the bounded parameter fixes it. The library's patch Newton spends an iteration latching
  the hold without moving, so a one-iteration settle test passes points that would still move.
- **One seed per face is not enough.** A face can hold several local nearest points for one
  brick; seeding each candidate span finds them.

What it says about the kernel:

- **Straight-line inside a brick is feasible at the fine levels.** Per-span seeds and six fixed
  steps give exact distances at 99.997% of near-surface points in 0.5 mm bricks.
- **Undetected errors are not zero.** About one point in 100,000 settles on a wrong local
  nearest point. A per-point certificate has to turn every one into a flag: each candidate span
  ruled out by its lower bound (box and support-plane bounds, both straight-line), and a second
  nearer point inside the winning span ruled out by fixed-depth subdivision. Not built.
- **The flag rate is overstated.** "Some candidate not settled" flags points for spans that could
  never win; flagging only spans whose lower bound is under the best distance would cut both the
  flags and the work. Not built.
- **The cost is the candidate count.** Six steps over 7 to 34 spans, each step a rational bicubic
  evaluation with second derivatives, is about 17,000 to 83,000 operations per point by count:
  tens of millions of points a second on the T1000 by arithmetic alone. Not measured on the GPU.
- **The fallback is the certified search,** 0.8 to 1 ms per point per thread on THE PART.

## 14. Trim loops for analytic faces, and what they exposed

*2026-10-26 (Derek: "go" on step 1 of the import route).*

### Analytic surfaces as exact rational B-splines

Each plane, cylinder, cone, sphere and torus face is written as an exact rational B-spline over
its extent (`trim2::Analytic`), so section 12's construction and certificate apply unchanged.
Circles are rational quadratic arcs (weights 1, cos(phi), 1 for an arc of half angle phi, the
middle control point at 1 / cos(phi)); surfaces of revolution take the products of a circle and
a meridian, weights multiplied. On such an arc tan(alpha / 2) = tan(phi / 2) (2 t - 1), alpha
measured from the arc's middle, so a point's parameters are closed form. Su x Sv is the
surface's own normal in every form, so the face's sense orients its loops as for a B-spline
face. A unit test checks, over a grid of parameters on every kind (ring, horn, apple and lemon
tori included), that each point is on the surface to 1e-12, that the inverse returns its
parameters to 1e-10 except along a collapsed direction, and the orientation.

- Planes: their frame's (x, y) over a padded rectangle about the face.
- Cylinders and cones: (arc parameter around the axis, height), the height range padded about
  the face's and kept on the apex's own side.
- Spheres: (longitude, latitude) arcs, with the poles placed as far from the face's boundary as
  49 candidate axes allow. A sphere's parametrization is singular only at its poles and where
  they lie is free; a boundary through or within 1e-10 of a pole lost track of how many times
  the curve winds around.
- Tori: around the axis, and around the tube from the outer equator for a ring or horn torus;
  an apple or lemon keeps the meridian arc on the axis's own side, in equal arcs of at most 90
  degrees between the two angles where it meets the axis.

### What it exposed in the construction (each fix applies to B-spline faces too)

- **Loops were split at every edge a face uses twice and each run closed on its own start.** A
  cone face of THE PART reaches several holes through slit edges (an edge from the outer
  boundary into a hole, with the face on both sides); the run between two slits was closed by a
  segment drawn 41 mm across the face, and the "vertex gap" allowance, measured across the slit,
  let the certificate pass it. Boundary uses are now linked into cycles by vertex identity:
  after the twice-used edges are dropped, each use is followed by the next one that starts at
  the vertex it ends at. THE PART's largest vertex gap went from 41 mm to 7.7e-5 mm and its
  largest certified deviation from 41 mm to 7.4e-3 mm. No file had a 41 mm gap; the pairing did.
- **The reference answer came from a point outside the parameter domain, taken as outside the
  face.** That holds for every side but one the surface collapses on (a sphere's pole, a lemon
  torus's tip), which a face can reach through with no edge there: 12 tori and 8 spheres of
  THE PART read inside out. The reference is now a point offset from a loop piece's middle,
  settled to have that piece as its nearest loop point, inside exactly when it is on the
  face's side by the loops' direction; it is taken from the piece's copy inside the domain, on
  either side of the piece, and its offset never shrinks below a millionth of the domain, so the
  side is never decided at rounding scale (it once settled one unit in the last place inside a
  domain border).
- **Periodic copies were sized by the domain's diagonal over the period**, mixing an arc
  parameter with a height in millimeters: a long cylinder asked for 1.6 GB. Copies now reach one
  period either side of the domain along each periodic direction.
- **A loop's closing piece took the copy of the loop's start nearest the piece's start**; a long
  closing piece that winds more than half a period was cut short. It now takes the copy nearest
  its own last sample.
- **A piece under 1e-10 of the surface's size is a point**: its two ends are reduced to one
  period copy before a segment between them is certified.

### A judge independent of both tests

Beside every edge of every face: a step off the edge's middle into and out of the face, along
the outward normal crossed with the edge's direction in its loop (the file's convention: the
face lies to the left of its loops seen from outside), put back on the surface. It is wrong
where the step leaves a narrow face or where an edge runs along the surface's own border (the
outward step projects back onto the face), and there both tests miss alike; on the faces turned
inside out it was the one judge that told which test was right.

### Results (`icam-check trim2`, 200 random parameters per face; analytic faces with ICAM_TRIM2_ANALYTIC)

| | Suite, B-spline | Suite, analytic | THE PART, B-spline | THE PART, analytic |
| --- | --- | --- | --- | --- |
| Faces with certified loops | 969 of 976 | 13,854 of 13,867 | 776 of 776 | 1,816 of 1,819 |
| Tree against the polygon count | 0 differ | 0 differ | 0 differ | 0 differ |
| Old and new disagree | 1.14% | 0.54%, 14,852 of 14,890 within two tolerances of an edge | 0.13% | 0.019% |
| Old test undecided | 50 | 8,797 | 0 | 21 |
| Beside the edges, new wrong / old wrong | 739 / 2,474 | 7,636 / 8,250 | 655 / 2,036 | 130 / 144 |
| Per test, old / new | 244 us / 1.3 us | 7.7 us / 2.2 us | 168 us / 0.8 us | 14 us / 1.5 us |
| Build, all faces of that kind | 30 s | 31 s | 20 s | 20 s |

Refused, each with the edge, the bound and the points: THE PART's 3 analytic faces (closing
pieces of loops on periodic surfaces: two pieces 1e-10 to 1e-8 mm long whose ends land a period
apart, one piece split twelve times at one knot line) and 20 suite faces at the floor of files
declaring 1e-6 mm (rear, tork). With topology pairing, 100 suite faces have edges that miss each
other at a vertex by more than their file's tolerance, the largest 5.8e-3 mm (rear 29, tork 16
among them): the files' own defects.

### Collapsed sides and the parameter path

*Same day, before the evaluator.* Two more causes of refusals, both fixed for every face kind:

- **Sides the surface collapses on.** A domain side that maps to one point (a cone's apex, a
  sphere's pole, an apple or lemon torus's tip) is found at build. An edge point within the
  file's tolerance of that point is put on the side, since the file does not say which way round
  it the edge goes; along the side the loop runs the way the face's orientation requires, which
  is exact because the whole side is one point.
- **Period copies near a pole.** Near a pole the parameter around it turns fast, and with a dozen
  samples a piece could change copy between two of them, after which every point was a full turn
  off. Each edge piece's parameter path is now refined once, until consecutive points differ by
  under an eighth of a period, and every sample, split point and loop closure takes its copy from
  that path.

A bisection along the edge for the exact knot-line crossing was tried and withdrawn: with or
without the path it raised THE PART's refusals (to 7 and 10).

| | Suite, B-spline | Suite, analytic | THE PART, B-spline | THE PART, analytic |
| --- | --- | --- | --- | --- |
| Faces with certified loops | 969 of 976 | 13,860 of 13,867 | 776 of 776 | 1,818 of 1,819 |
| Tree against the polygon count | 0 differ | 0 differ | 0 differ | 0 differ |
| Build, all faces | about 30 s | about 30 s | 48 s (20 s before the path) | 48 s |

Still refused: THE PART's sphere face #209236, a long piece whose end lies 3e-4 past a knot line
and whose fit stays 0.28 mm off while the crossing split shaves slivers from its end (halving
there fixed it but broke a near-pole face); the suite's 14 at the 1e-6 mm floor of rear and tork.

## 15. The exact evaluator

*2026-10-26 and 27 (Derek: "go" on step 2 of the import route).*

`solid3::Exact` is the solid's signed distance from its certified loops (sections 12 and 14).
The nearest boundary point of p is a critical point of the distance inside a face, an interior
point of an edge, or a vertex. Every face, edge and vertex goes into one box tree, and the
features are visited best first, each against the nearest found so far.

- **Analytic faces.** The distance on a whole plane, cylinder, cone, sphere or torus has one
  local minimum, its closed-form foot (on a torus the other three critical points are two
  saddles and a maximum; on a single nappe of a cone and on a lemon the same holds). The face
  contributes its foot when the foot is in the face by its loops, and nothing otherwise: its
  nearest point is then on its boundary, which its edges and vertices give. Where every point
  of a circle is nearest (p on the axis), the circle's ends in the face are on edges at the same
  distance, so no special case is needed. A cone's apex or a lemon's tip in the face, away from
  every vertex, is a point of its own with the mean normal around it.
- **B-spline faces.** A branch and bound over the surface's patches skips patches wholly
  outside the face (the loops classify a parameter box exactly: over the parameters where a
  monotone piece's u lies in the box, its v runs between its end values), and returns only
  critical points of the squared distance inside the face. A face point that is not one only
  bounds the search: next to a trim edge the loop runs up to the band beyond the true edge, and
  such a point could beat the edge by that much and take its sign from a curved face's normal,
  which can be wrong there. Points that are not critical are first carried on by Newton across
  spans, which ends on the critical point they were heading for.
- **The critical-point test.** A patch is not split when |S - p|^2 has no critical point on
  it: with Q = P - p w, g_s w^3 / 2 = w Q.Q_s - w_s Q.Q and alike in t are polynomials, so
  a g_s + b g_t along the gradient at the patch's middle has a Bernstein form, and when all its
  coefficients are positive (clear of rounding by a bound on the factors' products) the patch
  holds none. Without it, a nearest point on a trim edge with the untrimmed surface running on
  past it toward p keeps every patch along the edge above the bound, and the search splits them
  down to the settling distance: on one moon_buggy face 20,001 steps (the cap) against 48 with
  it. Patches touching a side the surface collapses on (a pole) are never cleared.
- **Knot lines of full multiplicity.** Interior knots of multiplicity at least the degree leave
  a surface only continuous in its parameters, with no derivative across them, and a minimum
  on such a line need not be a critical point of either side.
  Each such line inside a face is searched along like an edge (a one-dimensional branch and
  bound with its own critical-point test, critical points along the line only), its sign from
  the two sides' normals summed; patches beside it are then cleared like any other. st203-bapl
  writes most of its surfaces as piecewise Bezier (14 of 20 interior knots at full
  multiplicity): 170 capped searches before, none after, and 10 times faster.
- **Edges and vertices.** An edge's nearest point below the current best (its search stops at
  the bound), interior points only, signed by its two faces' normals summed; a vertex by its
  angle-weighted pseudo-normal. A face whose surface collapses to a point at the vertex (a pole,
  a cone's apex) has no normal there: its fan of limiting normals around the point (S_k x S_kw)
  is weighted by the angle its limiting tangent turns through, over the part of the side the
  face covers. Before this, a spike tip on moon_buggy took an arbitrary normal from the pole and
  read points 1,000 mm outside the part as inside.
- **Faces without certified loops** fall back to the old trim test, and the answer is marked
  unsure.

### What the check found, each fixed

`icam-check exact` checks the evaluator with no B-rep reference: a dense cloud of boundary
points (each face's parameter grid inside its loops, edge points, vertices) that no answer may
miss by more than the band (the file's tolerance plus the loops' certified deviation from their
edges, where the file's own edge offsets put a face next to its edge up to that much off); the
half-segment rule (the point halfway to the nearest point has half the distance, the same sign
and the same nearest point); the Lipschitz bound on random pairs; a positive value outside the
solid's box; and surfcheck's samples off the faces. Faults it found, in the evaluator and in
the loops:

1. **Cones closed by a vertex loop.** A drill-point cone runs from its circle to its apex, a
   VERTEX_LOOP. The face's range and box were built from its edges alone and stopped at the
   circle, and a range with no height made the sampled closure test call the height direction
   closed, so off-range points wrapped to either side of the loop at random (monster4: 14 bad
   samples). Vertex loops now bound a face's extent, and an analytic form closes where its kind
   does.
2. **Queries on a straight loop piece.** Round-number parts put queries exactly on their loops'
   axis-aligned pieces, where the crossing count's two legs used rules that disagreed (an 80 mm
   miss on iso14649-demo). The count now takes its whole path moved up by an infinitesimal e
   and right by e^2, both ends: a reference point can sit on a piece's line too (monster4: a
   reference offset from the middle of a 50 mm edge landed exactly on the line of a slot's side,
   and the face read as its complement). Pieces fitted to a straight edge also wobble about
   it by 1e-12, and the count takes each piece's ends as its range: every turning point now
   splits a piece, however small.
3. **Straddling by control boxes.** A parameter box was "straddling" when a piece's control
   box met it, and that box never shrinks: tiny patches stayed undecided down to rounding, where
   they multiplied four to a level (the capped searches). The curve itself is tested now.
4. **Lemon faces.** A DEGENERATE_TOROIDAL_SURFACE with select_outer false is the lemon. ISO
   10303-42 (2021, note 1 to the entity) puts its normal out of the lemon, opposite to Su x Sv;
   THE PART (HOOPS Exchange 2025.8) uses Su x Sv, so all 12 of its lemon faces were the
   complement of themselves, and 97 of 1,000 points outside its box read as inside. Which a
   file used shows at a lemon's tangent edges, where a valid solid's two outward normals agree
   and are never nearly opposite: the importer turns a lemon over when its nearly opposite
   neighbors (dot below -0.999) outnumber its nearly equal ones, and reports lemons with neither.
   THE PART: 12 turned over, none undecided; the suite has no lemons. The premise has an
   exception: two faces can touch tangentially with opposite normals in a valid solid (a cusp,
   item 6), so a lemon that met its neighbors only at cusps would be turned over wrongly; one
   with a cusp among tangent edges is decided by the majority.
5. **Last-bit wobble.** Even after item 2, a piece rising from v = 3 to 3 + 1e-15 was read as
   crossed by one leg of the count (from its ends) and not by the other (its value between them
   evaluated to exactly 3), and a rear torus face read as its own complement at a grid point on
   its edge's line; evaluating four equal control values need not return their value either.
   Loop control values are now snapped to a grid of 2^-40 of the domain's extent along each
   direction, counted from the domain's start, with values within half a step of either end put
   on that end (a piece that wobbled in its last bits becomes exactly constant; rounding to a
   grid keeps order and shared ends, and moves the loops by 2^-41 of the domain), and a constant
   evaluates to itself. A periodic direction is snapped only when its period is a whole number
   of steps (an analytic form's 4; a closed B-spline's period is not), since a loop closes on
   its start one period over. The snap made the beside-the-edge judge of section 12 count far
   more errors on B-spline faces (tork 332 to 891), all of one kind: a step out past an edge
   along the surface's own border lands on that border, the face's boundary, where either answer
   is right; before, the pieces there shared the closest-point search's last-bit rounding and
   two of the four sides happened to read outside. The judge now sets those points aside and
   reports how many (tork: 11 wrong of 3,968 judged, 1,660 on the border).
6. **Cusp edges.** A 1996 rear part has a torus bead touching a plane tangentially with opposite
   normals. The edge's two normals cancel, so a point nearest that edge has no sign from it (the
   half-segment rule caught a sign flip there); such answers are marked unsure.
7. **The old trim test as the sampler.** Surface samples are taken by the old test; where the
   loops say a sample is not in its face (THE PART: a B-spline face whose single loop encloses a
   small region of a narrow domain, two small spherical triangles), the geometry and the
   beside-the-edge judge side with the loops, and the check reports those samples as disputed.

### Results

`icam-check exact`, importer's reading. THE PART: 1,000 queries of each kind; the suite (56 files,
331 solids): 400 of each kind per solid.

| | THE PART | Suite |
| --- | --- | --- |
| Queries | 3,000 | 393,200 |
| Missed a nearer boundary point beyond the band, not marked unsure | 0 | 0 |
| Half-segment rule broken beyond the band, not marked unsure | 0 | 0 |
| Lipschitz bound broken by over twice the band | 0 of 1,000 pairs | 3, all in rear's inconsistent solid |
| Not positive outside the solid's box, beyond the band | 0 | 1, the same solid |
| Surface samples: off plus or minus delta, sign wrong | 0, 0 of 2,202 (13 disputed) | 0, 0 of 27,450 (6 disputed) |
| Searches stopped at their cap | 0 | 4 (moon_buggy, a far pole) |
| Against the old B-rep distance | 1 of 200 differ, the new nearer | 2,296 of 66,200 differ, 9 with the old nearer beyond the band (tork, the old test's error, checked against the raw B-spline) |
| Queries per second, all threads | 2,030 to 2,850 (the old distance: 11) | 19,350 |
| Without the critical-point test | 5 to 18 per second, 164 to 175 of 300 queries capped | |

The loops after this section's changes (`icam-check trim2`, 200 parameters per face): every
face built as before (THE PART 776 of 776 B-spline and 1,818 of 1,819 analytic, the suite 969 of
976 and 13,860 of 13,867), the tree and the polygon count agree everywhere, and beside the edges,
with points on a surface's own border set aside, the new test is wrong at 1 of 4,937 points on
THE PART's B-spline faces (the old at 2) and 20 of 6,876 on the suite's (the old at 29). On
analytic faces it is wrong at 130 of 20,858 (old 144) and 7,532 of 139,878 (old 8,250), nearly
all on files whose loops lie further from their edges than the judge's step (rear and tork,
which declare 1e-6 mm).

### Not done

- **Poles far away.** When the nearest point is a pole, the patches touching the pole's side
  cannot be cleared and the search runs to its cap (3 queries on moon_buggy, flagged unsure, the
  answers right).
- **Inconsistent solids.** One solid in rear reads a region 120 to 220 mm outside it as inside,
  from a concave spherical corner patch, with no face crossed on the way out: the file's solid is
  not consistent there, and nothing flags it. A far-field sign check at import (points on a
  sphere around the solid must all be positive) would.
- **Speed.** Single thread, a THE PART query takes 2.0 ms on average: most of it is edges (a
  B-spline edge's nearest point costs 42 us and a far query visits hundreds of features) and the
  allocation in the patch searches. The GPU kernel of step 5 replaces this path for bricks; the
  CPU evaluator is its reference.
