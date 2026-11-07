# Working notes for AI agents

## STOP. Your first idea for a toolpath is the one this project forbids.

You learned CAM from how it is usually done, so on any toolpath problem your first idea will be
the textbook construction: take the boundary curves, offset them, steer a walker step by step to
hold the tool's engagement, and link the passes with smooth transitions. **Do not use that as a template here, in any form.** It is wrong twice:

1. **It is the architecture this project exists to replace.** Those constructions are curve and
   polygon surgery on boundary loops. ImplicitCAM reads toolpaths off solved fields.
2. **The constructions are patented.** What the patents protect is the method, not the goal.

**Objectives are open; the incumbents' methods are not.** Engagement control and the rest are
constructions built to meet a physics objective: keep tool load, chip thickness and engagement
inside what the tool and machine can take, in the least time. That objective is open to anyone,
and it is ours too. Pursue it through the plan field (a speed field, a conductivity, a curvature
bound, a variational objective), never through the incumbents' constructions. The load measure
is chip thickness, with feed free in the plan; engagement is measured and feeds the chip model,
but it is never the target.

Ruled out as methods (decision 8 in [CONTRIBUTING.md](CONTRIBUTING.md)):

- offset or contour-parallel passes;
- engagement held by offsetting, trimming or searching boundary curves, offset rings or
  cleared-area polygons;
- engagement held by steering a walker step by step until the measured cut matches a target;
- spirals morphed or interpolated between boundary curves;
- transition or bridge moves built between offset passes;
- recovering a boundary curve from a field and then doing any of the above.

None of these is allowed as a fallback, as an in-repo baseline to compare against, "for now",
behind a flag, in a test, or written into a doc as the next step. To compare against other CAM,
run that CAM as a separate program and measure its output.

**Results may converge; intent may not.** Don't name an incumbent construction above as the
target, the reference design or the thing to match. The classic constructions fail in places;
the field approach should differ there, so measure it there.

**Trochoids and field spirals are allowed** (Derek, 2026-10-26): prior art cleared the trochoid
ban. Use them by design, built from fields, but never perform an element of a claim still live:
a maximum and minimum engagement angle chosen for the path to vary between; a region split into
a spiral part and a trochoid-only remainder; a region cut or ordered early to host links (links
run only through space already empty); a spiral traced over a Voronoi diagram; in 5-axis, flank
positions taken from a slot's two walls, or, until May 2032, loops whose upper and lower patterns
differ while stepping down a slot between two walls. Decision 8 in CONTRIBUTING.md has the
list.

If the field-native way is not obvious, finding it is the job, not a reason to fall back. Stop
and say so in the issue or pull request.

## One planner. Every query takes a pose.

Your second reflex will be to solve the easy case first: a vertical tool, a slab, a flat floor, a
part made of horizontal faces. This project did that, and it produced a part model only prismatic
parts fit, built from stored edge curves, which the whole roughing pipeline now has to be moved
off (the roadmap's "One planner" section). Derek, 2026-11-07: one planner targets simultaneous
5-axis, and everything else falls out as a subset of it.

- Every query, plan and model takes a pose: a position and a tool axis. 3-axis, 2.5D and 3+2 pass
  a held axis; none of them gets a branch with its own algorithm.
- There is no part model for a class of parts. Whatever the exact evaluator reads, the planner
  plans.
- Before you propose a change, say how simultaneous 5-axis calls it. If it doesn't, don't build
  it.

## The field is never computed by intersecting surfaces

An exact distance can be computed the B-rep way: find the nearest point on each surface, on each
curve where two surfaces meet (Newton onto the pair, slide along the crossing), at each point
where three meet, and keep the ones a sign test says are on the boundary. That is exact, and it
was built here on 2026-10-25, measured at 34 to 277 points per second, and discarded the same
day. It is the brittle intersection logic the project exists to replace, and it is not a kernel.
Decision 10 in [CONTRIBUTING.md](CONTRIBUTING.md):

- The field is the exact evaluator's signed distance (`solid3::Exact`, decisions 2 and 3): each
  face's nearest critical point when the trim loops certified at import put it in the face,
  otherwise its edges and vertices, read for their own distance and sign (Derek, 2026-10-27; the
  eikonal redistance from F planned before that is no longer the route).
- It never comes from solving where two model surfaces meet, from a per-point search over
  intersection curves or triple points, or from any trim test other than the certified loops.
- Before you offer any field or distance method for approval, write its verdict against the
  four guarantees: a straight-line parallel kernel with a measured rate; exact where it claims to
  be; no holes and no thresholds that decide structure; no surface-intersection or B-rep logic.
  If it fails one, say so and withdraw it. Do not offer it "with conditions".
- Nothing is grandfathered. Code that breaks a rule (the prismatic compiler's edge-curve
  read in `frep.rs`) is slated for replacement and listed in `docs/EXACT_FIELD.md`.
  Seeing it done there is not a reason you may do it; the guard reports it until it is gone.

## Think GPU first

Field queries are independent per point, cell, pixel or pose, so planning, checking and
visualization here should be embarrassingly parallel. Agents keep writing single-threaded CPU
loops over field samples instead. Decision 9 in [CONTRIBUTING.md](CONTRIBUTING.md):

- Before writing any loop over samples, cells, poses or pixels, decide whether it runs as a GPU
  kernel (wgpu and WGSL, as in `icam-gpu`). If it doesn't, say why in the pull request.
- Pick the formulation that parallelizes: fast sweeping or the fast iterative method over fast
  marching, Jacobi or multigrid over Gauss-Seidel, per-point walk-on-spheres, per-cell pruned
  trees compiled to straight-line WGSL.
- Where the CPU is right (a true sequential dependency, tiny inputs, the reference itself), use
  `rayon` over independent work, never a single thread by default.
- Visualize on the GPU: raymarch or sphere-trace the field in a shader instead of rasterizing
  images on the CPU.
- Keep the double-precision CPU implementation as the reference every kernel is tested against.

## Read CONTRIBUTING.md

It holds the rules; the rest of this page adds what an agent most often gets wrong here.

## The decisions are hard constraints

- The eleven design decisions in CONTRIBUTING.md are not preferences to weigh against a task.
  If a task seems to need a construction they forbid, stop and say so in the issue or pull
  request instead of building around it.
- Claude Code loads `.claude/settings.json`: all eleven decisions are injected at session
  start (copied verbatim from CONTRIBUTING.md), every edit under `crates/` and `docs/` is
  checked against them, and the tree is re-checked when you stop. A refusal names the decision
  it enforces. Other agents get none of this, so apply the decisions yourself and run
  `node tools/frep-guard/check.mjs` before every commit.
- Never commit with `--no-verify` or `-n`, set `core.hooksPath` to anything but `.githooks`,
  edit `tools/frep-guard/`, `.githooks/` or `.claude/settings.json`, or run `check.mjs
  --accept`. Those belong to the maintainer.
- Don't route around the guard by renaming an identifier, moving code into a file the rules
  don't cover, or editing through the shell. A pattern that blocks you is pointing at a
  decision; change the approach.
- Docs may record analysis, measurements and proposals. Open a proposal with
  `*Proposal, <date> (<author>, for Derek's review). Not adopted.*` and never write a
  forbidden construction into a doc as the plan or the next step.

## Generator work

- Apply the field-native test from CONTRIBUTING.md to every change in how paths are generated,
  and report the gates (gouges, continuity) with numbers.
- Check section 13.3 of `docs/ROUGHING_FEASIBILITY.md` before proposing a generator. Each
  anti-pattern there was built once already.
- Verify posted programs with `icam-stock`'s `ImplicitStock`, never with the generator's own
  in-process stock.

## Evidence

- Run the reproduction yourself and put the measured numbers in the pull request. Don't ask a
  human for evidence you can produce.
- When you time two methods, run them interleaved in one pass on the same machine. Timings
  taken in separate runs drift with load, power state and thermals, and can flip a verdict.

## Words

- No `Mesh`, `Tri`, `winding`, `polyline`, `tessellate` or `facet` identifiers in `crates/`;
  the guard reads names.
- Call any compared commercial CAM or CAD program "the baseline" or "commercial CAM", never by
  product name. Machines, controllers and G-code dialects may be named.
- Commit messages: `<area>: <what changed>, because <why>`, in one sentence.
