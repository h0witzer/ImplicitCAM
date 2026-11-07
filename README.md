# ImplicitCAM

A functional-representation (F-rep) CAM kernel for subtractive machining, written in Rust.
The part, the stock and the toolpath plan are scalar fields over space. Toolpaths are read off
solved fields (an eikonal or elliptic plan, a gradient, a constrained walk) instead of being
built from combinatorial B-rep or mesh geometry: no offset loops to trim, no trimmed surfaces
to intersect, no tessellation of exact input.

High-efficiency 3-axis roughing and finishing come first. Simultaneous 5-axis is the intended
endpoint, and the core types are shaped for it from the start. The thesis, the decisions
behind it and the phases are in
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
- **Parallel by construction:** every field query is independent per point, so planning,
  checking and visualization are built as GPU kernels, with a double-precision CPU
  reference behind each one.
- **Gradients and continuity:** the plan and the stock are differentiable, so the time a plan
  costs can be descended, and a small change to the part moves the path by a small amount.

## Status

Research code. There is no end-to-end command line yet, and nothing here should drive a
machine.

What works today:

- STEP import that keeps every face on its exact surface (planes, cylinders, cones, spheres,
  tori, rational B-splines) and every B-spline native, never tessellated.
- Prismatic solids compile to one global F-rep: 1-Lipschitz, zero set exactly on the
  boundary, with interval pruning. General solids have an exact signed-distance reference
  that checks the compiler; the general compiler is open work.
- The tool contact query on the CPU (the reference) and on the GPU (wgpu, WGSL), checked
  against each other by `icam-check`.
- An independent, exact swept-tool stock that checks any 3-axis program for gouges, read
  through a G-code reader (Fanuc-style and MultiCAM dialects).
- The roughing plans (max-cut, physics, harmonic, ratio, conductive, objective, walls).

In progress: the measuring instruments, the command line and posting, finish allowance and
tool assignment.

## Layout

| Path                     | Contents                                                                    |
|--------------------------|-----------------------------------------------------------------------------|
| `crates/icam-core`       | Exact geometry (B-splines, curves, surfaces, solids), STEP import, the F-rep compiler, the reference contact query |
| `crates/icam-gpu`        | GPU kernels for the contact query (wgpu, WGSL)                              |
| `crates/icam-check`      | Command-line checks: GPU against CPU, compiled F-rep against the exact reference |
| `crates/icam-post`       | G-code in and out: the program reader every instrument measures with        |
| `crates/icam-stock`      | The machined stock: the independent exact checker and the in-process stock  |
| `crates/icam-query`      | Queries against the part field for planning                                 |
| `crates/icam-motion`     | Physical cut limits and feeds: chip, power, deflection, chatter, machine    |
| `crates/icam-strategy`   | Roughing strategies read off solved plan fields                             |
| `docs/`                  | Architecture and roadmap, the roughing feasibility study, proposals         |
| `tools/frep-guard`       | The check that keeps B-rep constructions out of the F-rep path              |
| `whitepaper-references/` | The reference papers: a verified list and a fetch tool (no PDFs)            |

## Building

Stable Rust. Node 18 or newer for the design guard and the reference fetch tool.

```sh
cargo test --workspace --release              # unit and pipeline tests
git config core.hooksPath .githooks           # run the design guard on every commit
node tools/frep-guard/check.mjs               # or run it over the whole tree by hand
cargo run --release -p icam-check -- frep part.step   # compile a part's F-rep and check it
node tools/fetch-references.mjs               # download open copies of the reference papers
```

`icam-check gpu` needs a GPU that wgpu can open; the GPU test skips itself without one.

## Contributing

Help is welcome, from a test part to a new plan field. [CONTRIBUTING.md](CONTRIBUTING.md)
covers building and testing, the design decisions every change keeps, the contributor
agreement and where help is wanted. In short:

1. Sign the [Contributor License Agreement](CLA.md). The CLA bot asks on your first pull
   request.
2. Keep the nine design decisions; the guard checks them on every commit and in CI.
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
