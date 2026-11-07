# Proofs of the kernel's claims

**Trial, started 2026-10-31 (Derek): use Lean proofs to make the kernel resolve as fast and as
correct as mathematically possible.** Not a decision. Nothing here changes a field method; the
first step is a measurement that says how much of the payoff is speed and how much is
correctness.

## Against the four guarantees

This adds no field or distance method. It proves the claims the exact evaluator and the brick
cache already rely on (docs/EXACT_FIELD.md, "The four guarantees"):

1. **Fast and parallel.** Proofs cost nothing at run time. A margin proven safe at a smaller
   value, or an unsure rule narrowed to the cases a proof cannot close, can only shrink the set
   of nodes the settle pass reads again.
2. **Exact where it claims to be.** Each claim below is stated in the code as a comment. A proof
   makes it checkable.
3. **No holes.** The GPU kernel's margins are hand-picked constants. A proof says, for each one,
   whether it only sends an answer to unsure (a speed setting) or can decide an answer wrongly
   (a hole).
4. **No brittle intersection or B-rep logic.** The sign rules for faces, edges and vertices are
   the inside/outside answer everything downstream relies on. Claim 4 is that rule.

## What a proof buys

Lean proves statements. It does not make code faster. Speed comes from two places: a margin
proven safe at a smaller value, and an unsure rule cut down to exactly the cases a proof cannot
close. The settle pass reads every unsure or bridged node again exactly, so both shorten it.

"Fastest possible" is not something these proofs can show. What they can show, test by test,
is the tightest bound that is still sound.

## The claims

### Claim 1: deciding boxes and bricks

`crates/icam-sim/src/partcache.rs`, the module comment and `exact_brick`. A box is decided whole
by the distance at its center plus and minus half its diagonal. A node reading of |d| past the
clamp settles every node within |d| minus the clamp. Both rest on the field being 1-Lipschitz,
which holds for a true signed distance. The lemmas are short over the reals; the work is in
stating exactly what the code assumes (strict or non-strict inequalities, the clamp against the
cell diagonal).

### Claim 2: signs by flood fill

Same file. A node the GPU knows only to lie past the clamp takes its sign from a neighbor within
a cell diagonal "which no surface separates from it". The condition that makes this sound is
|d| greater than the distance between the two nodes. Whether the code tests it with the right
inequality, against the right length, is the claim. Any wrong sign at a read node spreads
through this fill (see the bug under Claim 4).

### Claim 3: the GPU kernel's margins

`crates/icam-sim/src/partgpu.rs`, `PartGpu::new`: the margin is `8 * f32::EPSILON` times half
the part's diagonal, commented "a few ulps", which is a choice, not a bound. The kernel,
`crates/icam-core/src/solid3/gpu.wgsl`, uses it with more chosen constants:

| Line | Test | What a wrong value does |
| --- | --- | --- |
| 735, 799 | Newton stops at `moved <= 2e-7 \|\| step <= 0.125 * P.eps` | Stops early or late |
| 742, 806 | critical point within `1e-4 * dist + P.eps` | Accepts a point that is not critical, or rejects one that is |
| 881, 908 | `4.0 * P.tol + 16.0 * P.eps` | Decides near-surface answers |
| 962, 1678, 1738 | prune when `far_dist(...) < g(...) * 0.999 - P.eps` | A candidate pruned wrongly gives a wrong distance |
| 1355 | box pad `P.eps + 1e-6 * max(abs(lo), abs(hi))` | A box too small misses a feature |

The pruning tests come first: a wrongly pruned candidate is a hole, not a slowdown.

The float model: Lean has no mature IEEE 754 library. Proofs work over the reals with the
standard model, where each operation's result is the exact result times (1 + δ) with |δ| at
most u, and u comes per operation from the WGSL accuracy table. Addition and multiplication are
correctly rounded there; division and square root are not, and subnormals may flush to zero. For
a whole straight-line kernel, FPTaylor or Gappa compute a rigorous rounding bound automatically;
the Lean proof covers the rule that uses the bound.

### Claim 4: the sign from the nearest feature

`crates/icam-core/src/solid3.rs`, `sdf_within`: the sign is that of `(p - q) · n`, with a
face's normal, or an edge's or vertex's pseudo-normal; the answer is unsure when the search was
capped, a loop is uncertified, or an edge is a cusp (its normals cancel). For triangle meshes
this rule is a theorem (Bærentzen and Aanæs, "Signed distance computation using the angle
weighted pseudonormal", 2005). For curved faces trimmed by certified loops, with lines inside
faces where only position is continuous, it is not proven. The aim is a statement of exactly when the nearest feature's sign is
right; the cases the proof cannot close become the unsure rule, and nothing beyond them.

**The first case, measured 2026-10-29.** THE PART at 0.5 mm, the brick holding node
(-9.81133, 107.60228, -51.87650): the GPU reads -0.7961476, the true distance is +0.796145
(Face 16). Curve piece 27829 (an edge piece or a knot line) ties Face 16 within 2.2e-6 with the
wrong sign and wins the minimum. The check that reads every 29th node misses it, and far nodes
flood-filled from it inherit it. It shows without the build speed-ups too, so it is
older than that work. Two features at the same distance cannot have opposite true signs, so one
feature's sign rule is wrong there; Claim 4 says which.

## How the proofs connect to the code

- The proofs will live in `proofs/`, a Lake project (Lean 4 and Mathlib, Apache-2.0), separate
  from the Cargo workspace and not linked into any binary. The folder does not exist yet (step
  2 below creates it).
- No tool checks the Rust or WGSL against the proofs. Each claim in the code will cite its
  theorem by name in a comment, and the guard will fail when `lake build` fails or a cited name
  is missing. Not built yet.
- Aeneas translates safe Rust to Lean; it is not counted on here, since the kernel is float code
  and WGSL.

## First: measure before proving

Run on the machine that will do the work. Earlier timings in this repository were taken on a
laptop's T1000; compare only runs from the same machine.

1. `icam-sim part-bench test-parts/the-part/the-part-emm0sh.step --gpu` with the margin as it
   is: record the nodes left by reason, the wrong signs, and the settle time.
2. The same with the margin halved, then doubled (the `let eps` line in `PartGpu::new`).
3. The same on a second GPU if one is at hand. The WGSL error bounds allow different results
   on different GPUs, so a change in the counts or the wrong signs between GPUs is evidence the
   margin has to come from the specification's bounds, not from one card.

If the settle time barely moves with the margin, the proofs buy correctness (guarantees 3 and
4) and little speed.

## Order and effort

| Step | What | Effort |
| --- | --- | --- |
| 1 | The measurement above | 15 minutes |
| 2 | `proofs/` with Mathlib (its prebuilt cache is a few GB) | 30 minutes |
| 3 | Claims 1 and 2 | An afternoon |
| 4 | Claim 3, pruning tests first | 2 to 3 days |
| 5 | Claim 4, starting from the piece 27829 case | 1 to 3 weeks; may not close for every case |
