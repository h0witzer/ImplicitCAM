# ImplicitCAM

A functional-representation (F-rep) CAM kernel for subtractive machining, written in Rust.
The part, the stock and the toolpath plan are scalar fields over space. Toolpaths are read off
solved fields (an eikonal or elliptic plan, a gradient, a constrained walk) instead of being
built from combinatorial B-rep or mesh geometry: no offset loops to trim, no trimmed surfaces
to intersect, no tessellation of exact input.

ImplicitCAM is being built as one planner for simultaneous 5-axis: it plans the tool's position
and axis together, and 3-axis, 2.5D and 3+2 machining are the same planner with the axis held.
It is not built yet. The thesis, the decisions behind it and the milestones are in
[docs/ARCHITECTURE_AND_ROADMAP.md](docs/ARCHITECTURE_AND_ROADMAP.md).

## Why fields

What a field gives that curve-based CAM cannot (section 13 of
[docs/ROUGHING_FEASIBILITY.md](docs/ROUGHING_FEASIBILITY.md)):

- **Queries anywhere, before cutting:** distance to the part, stock height, channel width,
  wall thickness. An objective can be scored at every point.
- **Topology for free:** fronts merge and split without curve surgery. A plan's saddles are
  where passes split, so they are designed link sites.
- **The plan is solved, not constructed:** an eikonal with a speed field, or an elliptic plan
  with designed conductivity, so the physics of the cut shapes the path, not only its feed.
- **Parallel by construction:** every field query is independent per point, so the part's
  evaluator, the stock's sweeps and the viewer run as GPU kernels, with a double-precision
  CPU reference behind each one; planning is built the same way.
- **Gradients and continuity:** the plan and the stock are differentiable, so the time a plan
  costs can be descended, and a small change to the part moves the path by a small amount.

## Status

Research code. `icam-rough` roughs prismatic parts end to end today (plans, posts, measures
and checks a program) on the part model being retired; the one planner's command line arrives
with milestone 1. Nothing here should drive a machine.

What works today:

- STEP import that keeps every face on its exact surface (planes, cylinders, cones, spheres,
  tori, rational B-splines) and every B-spline native, never tessellated.
- The part's exact signed distance (`solid3::Exact`) for any solid the importer reads, from
  trim loops certified at import, on the CPU (the reference) and the GPU, sampled into a sparse
  brick cache.
- Tools as exact solids of revolution, posed anywhere, moved by any serial machine, each move's
  sweep certified; the stock as the box less every sweep (`icam-sweep`), and an independent
  exact stock that checks programs for gouges (`icam-stock`), read through a G-code reader
  (Fanuc-style and MultiCAM dialects).
- A viewer that draws the part and the machined stock from their fields (`icam-sim`).
- The roughing plans (max-cut, physics, harmonic, ratio, conductive, objective, walls), finish
  allowance and tool assignment.

Being replaced: the roughing pipeline still reads a part model that only prismatic parts fit
(`frep.rs`, `prismatic.rs`). Milestone 0 of the roadmap replaces it with pose-general queries
against the exact part; until then the design guard reports its edge-curve read in `frep.rs`.

## Layout

| Path                     | Contents                                                                    |
|--------------------------|-----------------------------------------------------------------------------|
| `crates/icam-core`       | Exact geometry (B-splines, curves, surfaces, solids), STEP import, the exact evaluator and its GPU kernel, tool profiles |
| `crates/icam-sweep`      | Poses, motions, machines, tool solids, certified sweeps and the stock they leave |
| `crates/icam-query`      | Queries against the part for planning                                       |
| `crates/icam-motion`     | Physical cut limits and feeds: chip, power, deflection, chatter, machine    |
| `crates/icam-strategy`   | Plans solved as fields, and paths read off them                             |
| `crates/icam-rough`      | Roughing end to end: plans, posting, measuring and checking                 |
| `crates/icam-finish`     | A variable finish allowance so the finisher sees constant load              |
| `crates/icam-assign`     | Assigning material to tools                                                 |
| `crates/icam-stock`      | The independent exact checker of a program's stock                         |
| `crates/icam-post`       | G-code in and out: the program reader every instrument measures with        |
| `crates/icam-sim`        | The viewer: the part and the machined stock drawn from their fields         |
| `crates/icam-gpu`        | GPU kernels for the prismatic contact query (being replaced)                |
| `crates/icam-check`      | Command-line checks: the exact evaluator, its trim loops and spans, the prismatic GPU kernel |
| `docs/`                  | Architecture and roadmap, the roughing feasibility study, proposals         |
| `tools/frep-guard`       | The check that keeps B-rep constructions out of the F-rep path              |
| `whitepaper-references/` | The reference papers: a verified list and a fetch tool (no PDFs)            |

## Building

Stable Rust. Node 18 or newer for the design guard and the reference fetch tool.

```sh
cargo test --workspace --release              # unit and pipeline tests
git config core.hooksPath .githooks           # run the design guard on every commit
node tools/frep-guard/check.mjs               # or run it over the whole tree by hand
cargo run --release -p icam-check -- exact part.step  # the exact evaluator's checks on a part
cargo run --release -p icam-sim -- part part.step     # view a part drawn from its exact field
node tools/fetch-references.mjs               # download open copies of the reference papers
```

The GPU tests skip themselves without a GPU that wgpu can open. With one,
`icam-sim part-bench --gpu part.step` checks the evaluator's kernel against the CPU reference.

## Contributing

Help is welcome, from a test part to a new plan field. [CONTRIBUTING.md](CONTRIBUTING.md)
covers building and testing, the design decisions every change keeps, the contributor
agreement and where help is wanted. In short:

1. Sign the [Contributor License Agreement](CLA.md). The CLA bot asks on your first pull
   request.
2. Keep the eleven design decisions; the guard checks the ones a pattern can see on every
   commit and in CI, and Claude Code's edit check covers the rest.
   Decision 8 is the one most often broken: **the incumbents' toolpath constructions (offset
   passes, engagement held by step-by-step steering, spirals morphed between boundary curves)
   are not our template.** Their physics objectives are fair game through field methods; their
   patented methods are not. Trochoids and field spirals are allowed, clear of the live claims
   decision 8 lists.
3. Open an issue before larger work, and claim it with a draft pull request.

AI coding agents are welcome; their working notes are in [AGENTS.md](AGENTS.md).

## License

Copyright 2026 Derek Van Allen.

ImplicitCAM is free software under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`); see [LICENSE](LICENSE). Running a modified version as a network service
obliges you to offer its users the source, the same as shipping it.

Using ImplicitCAM in a closed-source product or service needs a separate commercial license from
the copyright holder, Derek Van Allen. Contact [@h0witzer](https://github.com/h0witzer).

Contributions are accepted under the [Contributor License Agreement](CLA.md), which lets the
copyright holder include them in those commercial licenses. Contributors keep their copyright,
and every accepted contribution stays available as open source.
