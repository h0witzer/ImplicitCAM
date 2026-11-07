# Contributing to ImplicitCAM

ImplicitCAM is a CAM kernel for subtractive machining that works on fields. The part, the
stock and the toolpath plan are scalar fields, and toolpaths are read off solved fields instead
of being built from offset curves and trimmed surfaces. [README.md](README.md) has the
overview and the current status. [docs/ARCHITECTURE_AND_ROADMAP.md](docs/ARCHITECTURE_AND_ROADMAP.md)
has the thesis, the history of the decisions and the milestones.

Help is welcome, from a single test part to a new plan field. This page covers the contributor
agreement, how to build and test, the design decisions every change keeps, how to propose
changing one, and where help is wanted.

Bugs and proposals go in [Issues](https://github.com/h0witzer/ImplicitCAM/issues); questions
and ideas go in [Discussions](https://github.com/h0witzer/ImplicitCAM/discussions).

## Before your first pull request: the CLA

ImplicitCAM is dual licensed. Everyone gets it under AGPL-3.0-only, and the copyright holder,
Derek Van Allen, sells commercial licenses for closed products and services. He can only
include your contribution in those commercial licenses with your permission, so every
contributor signs the [Contributor License Agreement](CLA.md) once. In short:

- You keep the copyright in your work and can use it however you like.
- You let Derek Van Allen use, change and relicense your contribution, commercial licenses
  included, and you grant a license to any of your patents it needs.
- You confirm the work is yours to give, or that your employer has agreed.
- In return, your contribution stays available to everyone as open source. Commercial licenses
  are offered alongside the AGPL, never in place of it.

The CLA bot comments on your first pull request. Reply with the sentence it asks for, and that
one signature covers all your later pull requests. A pull request can't merge until everyone
who committed to it has signed.

## Get it running

You need stable Rust and Node 18 or newer (Node runs the design guard).

```sh
git clone https://github.com/h0witzer/ImplicitCAM && cd ImplicitCAM
git config core.hooksPath .githooks      # the design guard runs on every commit
cargo test --workspace --release         # unit and pipeline tests
node tools/frep-guard/check.mjs          # the design guard over the whole tree
```

The GPU tests skip themselves on a machine without a GPU that wgpu can open. With one,
`icam-sim part-bench --gpu <part.step>` checks the exact evaluator's kernel against the CPU
reference, and `icam-check gpu` checks the prismatic contact kernel (being retired).

The reference papers are not in the repository: most are publisher copies we may not
redistribute. `node tools/fetch-references.mjs` downloads the open copies and checks their
hashes. [whitepaper-references/MANIFEST.md](whitepaper-references/MANIFEST.md) says why each
paper matters here.

## The design decisions

These are Derek's decisions, and this list is their one record (Derek, 2026-11-07): the guard's
session-start text and the edit-check prompts copy it verbatim, and section 9 of the roadmap
keeps the dated history of how each came to be. A change that contradicts one is not merged,
however good it is otherwise; to change a decision, see
[Proposing a change to a decision](#proposing-a-change-to-a-decision).

1. **F-rep is the only internal representation.** Every stage after import (planning, gouge
   and collision checks, engagement, medial axis, reach, certification) reads fields, never
   B-rep faces, trim loops or topology. The exact evaluator of decision 3 is the field: it reads
   the exact surfaces, certified loops, edges and vertices inside `solid3` and `trim2`, and
   every stage after it reads its answers (distance, nearest point, feature).
2. **The part's field is its exact signed distance, one global function** (Derek, 2026-10-27;
   rewrites "a B-rep compiles to one global F-rep", 2026-09-29). The signed distance to the
   solid's boundary is one function everywhere by definition, so no cell ever holds a formula
   of its own and the field has no seams. It is evaluated pointwise from the exact surfaces and
   the trim loops certified at import (decision 3). Planning reads exact values, from the
   evaluator and its GPU kernel; the sparse brick cache samples the evaluation for drawing and
   for finding coarse entries far from the surface, never for a decision near it (Derek,
   2026-11-07). No CSG tree is compiled:
   global CSG needs separating primitives that fitted B-spline faces cannot supply, and the
   suite has 8,016 tangent edges (THE PART 2,105) where that construction fails. Per-cell
   formulas chosen by probes, clamps or local rules stay reversed (2026-10-02): they put seams
   in the field.
3. **The exact evaluator is the field, and it has no holes** (Derek, 2026-10-27; no holes:
   Derek, 2026-11-07; rewrites "the exact B-rep is an import-time reference", 2026-09-30).
   `solid3::Exact` gives the signed distance: each face contributes its nearest critical point
   when its certified loops put that point in the face, and otherwise its edges and vertices
   answer, signed by outward normals and pseudo-normals. The accuracy line at an edge is the
   larger of the file's declared tolerance and the deviation the importer measures there
   (Derek, 2026-10-26 and 2026-11-07): face interiors are exact, and within that band either
   neighboring face's answer stands. Every query gets an answer: the import never refuses a
   face (an edge off its surface is certified to the deviation it achieves, and reported), a
   cusp edge carries a sign rule, and the search is bounded by construction, with no step cap
   and no "unsure" answer. The double-precision CPU evaluator is the reference every GPU kernel
   is tested against. The path-walking trim test (`solid3::classify`, `sdf`,
   `nurbs_face_closest`) is retired and is deleted in milestone 0 of the roadmap; until then it
   still runs inside `Exact` for a face whose loops were refused, a hole on the retirement
   list, never a precedent. `frep3` was deleted on 2026-10-30.
4. **Sign comes from topology, never from a winding number** (2026-09-29; restated by Derek,
   2026-11-07). At an edge the sign comes from its two faces' outward normals, at a vertex from
   the angle-weighted pseudo-normal. The earlier wording, "topology decides each operator, max
   at a convex edge, min at a concave one", belonged to the global CSG plan (decision 10) and
   has no operator left to decide.
5. **Exact input is never tessellated.** No triangles, meshes or facets are made from B-rep or
   spline input. Meshes enter only when the mesh is the input (a scan, an STL-only file),
   through a mesh importer.
6. **B-splines stay native.** They are never lowered to lines, arcs, biarcs or polylines,
   except by a post processor.
7. **The glued field and the pure field stay rejected** (Derek, 2026-10-25). Neither the glued
   local-formula proposal in [docs/FREP_GLUING.md](docs/FREP_GLUING.md) nor its pure field (F
   for the sign, a distance among surface feet, intersection curves and triple points) is built,
   under any name. On 2026-10-25 the pure field was recorded as adopted with conditions and
   rejected the same day against the four guarantees; the record is decision 10.
8. **The incumbents' toolpath constructions are not our template.** Engagement-controlled
   roughing as commercial CAM does it is a set of patented methods for a physics objective:
   keep tool load, chip thickness and engagement inside what the tool and machine can take,
   in the least time. The patents protect the methods, not the objective. The objective is
   open and it is ours too: pursue it through the plan field, with chip thickness as the load
   measure and feed free in the plan (Derek, 2026-10-26). Engagement is measured and feeds the
   chip model, but it is not the target, and no maximum and minimum engagement angle is chosen
   for the path to vary between. The methods are ruled out in any form, including fallbacks,
   in-repo baselines, flags, tests and plans written into docs:
   - offset or contour-parallel passes;
   - engagement held by offsetting, trimming or searching boundary curves, offset rings or
     cleared-area polygons;
   - engagement held by steering a walker step by step until the measured cut matches a
     target;
   - spirals morphed or interpolated between boundary curves;
   - transition or bridge moves built between offset passes;
   - recovering a boundary curve from a field and then doing any of these.

   Results may converge; intent may not. None of these constructions is ever the stated
   target, the reference design or the thing to match. They are also the curve machinery the
   F-rep replaces, and where they fail is where the field approach should differ.

   **Trochoids and field spirals are allowed** (Derek, 2026-10-26). Trochoids were ruled out
   for their patents, and prior art clears them: the engagement-limiting patents that claimed
   circular passes have expired. A plan may use trochoids by design, and spirals read off a
   solved field, built from fields like every other path (decision 1). They must not perform
   an element of a claim still live (roadmap section 10 has the rules beside these; the
   maintainer keeps the claim reading):
   - a maximum and a minimum engagement angle chosen for the path to vary between;
   - one portion of a region given to a diverging spiral and trochoids only to the rest;
   - a region cut, or ordered early, to host links: links run only through space that is
     already empty;
   - a Voronoi diagram anywhere in planning (Derek, 2026-11-07: wider than the claim, which
     recites a spiral traced around a wave over one); medial information comes from solved
     plan fields;
   - in 5-axis, flank positions taken from a slot's two walls, and, until May 2032,
     simultaneous 5-axis loops whose upper and lower patterns differ while stepping down a
     slot between two walls.

   Spirals whose stepover rises and falls along each loop rest on prior art: Bieterman and
   Sandstrom published field spirals with that property in 2003 (Derek, 2026-10-26).

9. **Field work is built for the GPU first** (Derek, 2026-10-24). Field queries are
   independent per point, cell, pixel or pose, so toolpath planning, checking and
   visualization should be embarrassingly parallel. Before writing a loop over samples,
   cells, poses or pixels, decide whether it runs as a GPU kernel (wgpu and WGSL, as in
   `icam-gpu`); if it doesn't, say why in the pull request. A single-threaded CPU loop over
   independent samples is never the default.
   - Pick formulations that parallelize: fast sweeping or the fast iterative method over
     fast marching, Jacobi or multigrid over Gauss-Seidel, per-point walk-on-spheres,
     straight-line WGSL generated per tool, machine and span record.
   - Kernels run single precision in local frames, with rounding folded into the
     certification margin. The double-precision CPU implementation is the reference every
     kernel is tested against, and borderline gouge verdicts are re-decided on the CPU
     (roadmap section 9).
   - Where the CPU is right (a true sequential dependency, tiny inputs, the reference
     itself), use `rayon` over independent work.
   - Visualize on the GPU: raymarch or sphere-trace the field in a shader.

10. **The four guarantees. The goal is pure implicit CSG; it is blocked; the compromise is the
    certified loops** (Derek, 2026-10-25, rewritten 2026-10-27 and 2026-11-07;
    [docs/EXACT_FIELD.md](docs/EXACT_FIELD.md)). The goal was to turn B-rep constructs into pure
    F-rep ones through an implicit CSG, with no trims and no intersection tests: trims and
    intersections are what make B-rep brittle, and an implicit CSG has no holes or seams by
    construction. B-rep logic is never rebadged as an F-rep planner. The goal is blocked for
    fitted B-spline faces: they have no exact implicit (a bicubic patch implicitizes to degree
    18 with extra sheets) and no separating primitives at tangent edges (8,016 in the suite,
    2,105 on THE PART). The compromise is the exact evaluator over trim loops certified at
    import (decision 3). A rigorous global route for B-spline faces, if one is found, is
    preferred. The project rests on fast parallel evaluation of exact true fields with no holes
    and no brittle surface-intersection or B-rep logic. Guarantee 4 means that nothing in the
    architecture may open a hole, a seam or a failure in the exact inside/outside test
    everything downstream relies on, and no edge finding whose search time explodes (Derek,
    2026-10-26).
    - **Trim loops** are built once at import (fitted to the edges' preimages, certified to the
      larger of the file's tolerance and the measured deviation) and read at a query by exact
      crossing parity in a cell tree: always an answer, no retry, no threshold. That is the only
      trim test allowed. Edges and vertices are read at a query for their own distance and sign.
    - **A per-point search** over faces, edges and vertices is allowed in the CPU evaluator,
      bounded by box lower bounds and the critical-point test, bounded by construction with no
      step cap, and its measured rate is reported.
    - **The distance** is the exact evaluator's (decision 3). Planning reads it exactly, through
      the GPU kernel; the brick cache is for drawing and coarse entries (decision 2). Until
      2026-10-27 the plan was an eikonal redistance from F's zero set; that is no longer the
      route.
    - **Still out:** Newton onto surface pairs or triples, a slide along a crossing curve,
      bisection for where an exposed piece ends, an on-the-boundary test by a sign change across
      an epsilon step, a path-walking trim test, and a per-point search over intersection curves
      or triple points. The pure field of [docs/FREP_GLUING.md](docs/FREP_GLUING.md) was built
      that way and rejected on 2026-10-25.
    - **The evaluator, judged against the guarantees** (2026-10-27, restated 2026-11-07).
      Guarantee 1 is met by its GPU kernel (THE PART in 1 mm bricks in 2.3 s; the CPU reference
      is a per-point search at 2,030 to 2,850 queries a second on all threads). Guarantee 2 is
      met on THE PART (every `icam-check exact` check clean); the suite has 3 Lipschitz breaks
      and 1 sign failure in one inconsistent solid (`rear`), unflagged, on the retirement list.
      Guarantee 3 is not yet met: refused loops (26 suite faces, 1 on THE PART), cusp edges and
      the search cap are holes, closed in milestone 0, and one wrong sign carried by the brick
      cache (`docs/PROOFS.md`, piece 27829) is open. Guarantee 4, as Derek reads it, is met.
    - Before offering any field or distance method for approval, judge it in writing against
      the four guarantees: it evaluates as a straight-line parallel kernel (give the measured
      points per second); it is exact where it claims to be; it has no holes (no unresolved
      cells, no refused parts behind fallbacks, no thresholds that decide structure); it has no
      surface-intersection or B-rep logic.
    - A method that fails one is not offered for approval, with or without conditions, and
      "approved with conditions" never licenses building it.

11. **One planner, built for simultaneous 5-axis** (Derek, 2026-11-07). Every query, plan and
    model takes a pose: a position and a tool axis. 3-axis, 2.5D and 3+2 are the same code with
    the axis held, never their own code path, planner or part model; a restriction is a value
    passed in. No part model exists for a class of parts: whatever the exact evaluator reads,
    the planner plans. Every proposal says how simultaneous 5-axis calls it; if it doesn't, it
    is out. The prismatic part model (`frep.rs`, `prismatic.rs`, `part.rs`, the references in
    `query.rs`) and the pipeline built on it are being retired (roadmap, milestone 0); edits
    that only lower that code's counts pass, new held-axis code does not.

### The guard

[tools/frep-guard](tools/frep-guard) makes the decisions mechanical wherever a pattern can see
them. It flags B-rep internals outside import (runtime code may build and query `solid3::Exact`,
decision 3), the old B-rep evaluation and probe selection in the compiler, the B-rep's own
loops in the field, winding numbers, tessellation and mesh names, and lowered splines, and
surface-intersection solvers and B-rep geometry in the compiler (decision 10).
Comments and strings are ignored. Decisions 8, 9 and 11 are about method, not names, so no
pattern catches them: Claude Code's edit check covers 8 and 11, and review covers all three.

- **Nothing is grandfathered** (Derek, 2026-10-25). `baseline.json` holds no allowances. Code
  that breaks a rule is slated for replacement and listed in
  [docs/EXACT_FIELD.md](docs/EXACT_FIELD.md); the guard reports it at every session start and
  stop until it is replaced. It is never a precedent: that another file does it is not a reason
  you may.
- A file over a rule can be edited only to lower its count, and a commit that touches it must
  lower every count it is over on (the guard compares against HEAD), so replacement lands step
  by step and nothing new gets in.
- The pre-commit hook checks staged files; CI checks the whole tree on every pull request.
- Never commit with `--no-verify`, and don't rename things or move code to get past a
  pattern. If the guard blocks you, change the approach, or open an issue and ask.
- Changes under `tools/frep-guard/`, `.githooks/` and `.claude/settings.json` need Derek's
  review; `check.mjs --accept` is his alone and is not expected to be used again.

The guard reads identifiers, so names such as `Mesh`, `Tri`, `winding` or `polyline` in
`crates/` fail it even in harmless code. Name things for what they are in field terms.

## Principles for generator work

The decisions say what the kernel may not do. These say what a good change to path generation
looks like. They come from Part I of the roadmap and section 13 of
[docs/ROUGHING_FEASIBILITY.md](docs/ROUGHING_FEASIBILITY.md).

**The field-native test.** Every change to how paths are generated is judged by it:

1. The physics shapes the path, not only its feed.
2. Every decision is a continuous function of fields: no argmax on a flat curve, no threshold
   that switches structure, no greedy "nearest" choice, no feedback of a field's own
   derivatives without regularization.
3. Engagement, links and order are read from the plan (spacing, curvature, saddles, T), not
   discovered by walking.
4. The gates hold. Gouges: none, measured with the exact stock on the posted program.
   Continuity: a 5 µm wall change or a 0.1% physics change moves at most 5% of the path by
   more than 1 mm. Report beside them the share of the cut past the stable depth, the
   retracts, and the time as a share of the physics bound. Scores are absolute, never
   relative to a baseline program.
5. Engineering polish that is not a field advantage (lead-ins, arc fitting for output, the
   links between cuts) is copied from known-good practice, not reinvented. The cutting path
   itself is never smoothed after the fact (Derek, 2026-10-24): a path that is wrong is fixed
   in the plan field that produced it.

The anti-patterns already hit, with what each one cost, are in section 13.3. Read them before
proposing a generator.

**Also:**

- **Feed follows the load; the path never does** (Derek, 2026-11-07). Feed is scheduled along
  the path from the cut model, point by point. The path is never moved, step by step, to hold a
  load: that is decision 8's banned servo.
- **Bounds and exact values.** The stock's field is a bound (min and max of exact sweeps), safe
  for gouge and collision checks; compute an exact value where a precise boundary is needed, and
  never read a bound's magnitude as a distance (roadmap section 5).
- **Grids solve plan fields; they decide nothing near the surface** (Derek, 2026-11-07). The
  part and the stock stay exact. A plan field may be solved on a grid; reach, clearance, gouge
  and a wall's position are read from exact queries, never from a sampled grid (roadmap section
  6).
- **Verification reads separate inputs, not separate code** (Derek, 2026-11-07). The checker
  reads the posted program, never the planner's moves, against the exact part through
  `icam-sweep`'s certified sweeps, the same stock the planner uses; `icam-sweep` is itself tested
  against brute force. `icam-stock`'s `ImplicitStock` is a 3-axis-only checker and retires with
  the prismatic code.
- **Objectives yes, incumbent constructions no.** Decision 8 applies to every generator
  change: physics objectives, with chip thickness as the load measure, shape the plan field;
  the incumbents' constructions never shape the path, and trochoids and field spirals stay
  clear of the live claims listed there.

## Proposing a change to a decision

Decisions can change, but only on purpose and only by Derek. A plan written on 2026-09-30
reversed decision 2 without saying so, and code was built on it. This process exists so that
doesn't happen again. On 2026-10-25 a field proposal was put to Derek as approve, approve with
conditions or reject, with its decision violations listed as fixable; "approved with conditions"
was then taken as license to build intersection logic. So, before the choice is offered at all:

- State the verdict against the four guarantees of decision 10, one line each, with the measured
  evaluation rate. A proposal that fails one is withdrawn, not offered with conditions.
- Say how simultaneous 5-axis calls it (decision 11).
- Code that is still on the retirement list is not a precedent; don't lean on it.

1. Open an issue titled `Proposal: <what>`. Name the decision it affects, what you would do
   instead, and the measurements that show why.
2. For a larger design, add a document under `docs/` that opens, under its title, with
   `*Proposal, <date> (<your name>, for Derek's review). Not adopted.*`
   [docs/FREP_GLUING.md](docs/FREP_GLUING.md) is the example. Docs may record
   analysis, measurements and proposals. They may not state a forbidden construction as the
   plan or the next step.
3. No code that depends on the proposal merges until Derek accepts it. When he does, the
   decision is recorded on this page with his name and the date, the roadmap's section 9 gets
   the dated history, and the guard (`rules.mjs`, `.claude/settings.json`) is regenerated from
   this page.

## Code provenance and licenses

- **Don't copy or translate code from GPL, LGPL or AGPL projects, or from proprietary
  software.** Copyleft code written by others can't be included in the commercial licenses,
  and proprietary code isn't ours to use. Reading open-source code and papers for ideas is
  fine. Write your own implementation and cite the paper or project in a comment or in the
  pull request.
- **Dependencies** must be permissive (MIT, Apache-2.0, BSD, ISC, Zlib and the like) or
  MPL-2.0. No GPL, LGPL or AGPL dependencies. Name each new dependency in the pull request and
  say why it is needed.
- **Test data** (parts, programs, tool and material data) must be yours or openly licensed;
  say which. Don't submit a customer's or an employer's parts without written permission.
  Published cutting data, such as a tool maker's catalog, is fine with a citation.
- **Reference papers** are listed in `whitepaper-references/references.json` and never
  committed as PDFs.
- **Patents.** If you know of a patent that covers what you're building, name it in the issue
  (number, owner, and the claim element that matters). The default response is to find a
  method it doesn't claim.
- **Product names.** Name machines, controllers and G-code dialects as needed (Fanuc,
  MultiCAM). Don't name the commercial CAM or CAD product a comparison program came from, in
  code, docs, tests or commits. It is "the baseline" or "commercial CAM".

## Where help is wanted

**Test parts and benchmarks.** The work is measured on real parts, and more variety finds more
bugs. A useful case has:

- the part as STEP. B-spline faces are welcome. Any solid exercises the exact evaluator and its
  checks; until milestone 0 of the roadmap lands, roughing runs only on prismatic parts in
  their machining orientation.
- the stock, the tools (type, diameter, corner radius, flutes) and the material.
- optionally, a program for the part from any CAM (the baseline) in a dialect `icam-post`
  reads, with its tool table.

Open an issue titled `Test part: <name>` and attach the files as a zip, with a line saying
they're yours or how they're licensed. Measured results are as useful as the files: what
`icam-check` reports, run times, anything that looks wrong.

**Posts and machine dialects.** `icam-post` reads 3-axis mill programs in Fanuc-style and
MultiCAM dialects. To add a dialect to the reader, include a short program you wrote or posted
yourself, a test, and a citation of the public controller documentation you used. The writer
is being retooled for the one planner (milestone 1 of the roadmap) and isn't open for
contributions yet. Until then,
open an issue describing your controller's dialect, with the documentation section, so the
design covers it. A post processor is the one place B-splines may be lowered to lines and
arcs (decision 6); its file is added to the guard's allowed list when it exists.

**Field-native research.** The ranked list of open generator work is section 13.4 of
[docs/ROUGHING_FEASIBILITY.md](docs/ROUGHING_FEASIBILITY.md), and the milestones to
simultaneous 5-axis are in Part III of the roadmap. Start with a `Proposal:` issue: the idea,
how it passes the field-native test, and how you will measure it. Spikes and measurements are
welcome before code. A proposal whose method comes down to one decision 8 rules out is closed,
however it is dressed. Its objective may still be a good one.

**Not open yet: the instruments, command line, posting, finish allowance and tool
assignment.** The maintainer is retooling them. Bug reports are welcome; please don't send pull
requests in those areas yet.

## Sending a change

1. **Open an issue first** for anything bigger than a bug fix, and wait for a reply before you
   build a large change.
2. **Claim the issue with a draft pull request** before you start, so two people don't build
   the same thing. Branch from `main`, run `git commit --allow-empty -m "claim: #N"`, push,
   and open a draft titled `<summary> (#N)`. The lowest-numbered open pull request for an
   issue wins. Mark yours ready for review when it's done.
3. **Keep each pull request to one topic.**
4. **Write commit messages that say what changed and why**, in one sentence of the form
   `<area>: <what changed>, because <why>`. For example: `icam-stock, icam-motion: measure
   every sample's engagement at once against the stock as it stood at its own sweep count,
   because the stock before a step does not depend on the order the samples are read in`.
5. **Show measurements.** For geometry and generator work, give the measured numbers, the
   expected ones and where the expected ones came from. Generator changes report the gates
   and the measures from the field-native test. When you time two methods, run them
   interleaved in one pass on the same machine.
6. **Pass CI:** the design guard, the build and the tests, plus the CLA check.

**Tests.** New behavior comes with a test that measures it against a value you can derive: a
closed form, an independent computation, or the exact reference.

**Style.** There is no rustfmt or clippy gate yet. Match the formatting and naming of the code
around your change.

## AI agents

AI coding agents are welcome. [AGENTS.md](AGENTS.md) has their working rules. Claude Code picks
up the project hooks in `.claude/settings.json` automatically: they load the decisions at
session start, check each edit against them, and refuse `--no-verify`. The person who opens a
pull request is responsible for it meeting this page, whatever wrote the code.

## License

By contributing you agree to the [CLA](CLA.md), and your contribution is published under
AGPL-3.0-only like the rest of ImplicitCAM.
