// frep-guard rules: the B-rep constructions this repository has decided against, as
// patterns over Rust code (comments and string contents are stripped first).
//
// Every rule cites the decision it enforces. The decisions live in CONTRIBUTING.md, the one
// record (Derek, 2026-11-07); this file only makes them mechanical. DECISIONS below and the
// prompt hooks in .claude/settings.json are copied from CONTRIBUTING.md verbatim by
// sync-decisions.mjs; never edit them by hand, change CONTRIBUTING.md and rerun it.
//
// There are no allowances (baseline.json, 2026-10-25): code over a rule is slated for
// replacement and stays reported until it is replaced. Nothing may be added, and an
// edit to such a file may only lower its count.

/** The decisions, in the words the session-start hook injects. */
export const DECISIONS = [
  "F-rep is the only internal representation. Every stage after import (planning, gouge and collision checks, engagement, medial axis, reach, certification) reads fields, never B-rep faces, trim loops or topology. The exact evaluator of decision 3 is the field: it reads the exact surfaces, certified loops, edges and vertices inside `solid3` and `trim2`, and every stage after it reads its answers (distance, nearest point, feature).",
  "The part's field is its exact signed distance, one global function (Derek, 2026-10-27; rewrites \"a B-rep compiles to one global F-rep\", 2026-09-29). The signed distance to the solid's boundary is one function everywhere by definition, so no cell ever holds a formula of its own and the field has no seams. It is evaluated pointwise from the exact surfaces and the trim loops certified at import (decision 3). Planning reads exact values, from the evaluator and its GPU kernel; the sparse brick cache samples the evaluation for drawing and for finding coarse entries far from the surface, never for a decision near it (Derek, 2026-11-07). No CSG tree is compiled: global CSG needs separating primitives that fitted B-spline faces cannot supply, and the suite has 8,016 tangent edges (THE PART 2,105) where that construction fails. Per-cell formulas chosen by probes, clamps or local rules stay reversed (2026-10-02): they put seams in the field.",
  "The exact evaluator is the field, and it has no holes (Derek, 2026-10-27; no holes: Derek, 2026-11-07; rewrites \"the exact B-rep is an import-time reference\", 2026-09-30). `solid3::Exact` gives the signed distance: each face contributes its nearest critical point when its certified loops put that point in the face, and otherwise its edges and vertices answer, signed by outward normals and pseudo-normals. The accuracy line at an edge is the larger of the file's declared tolerance and the deviation the importer measures there (Derek, 2026-10-26 and 2026-11-07): face interiors are exact, and within that band either neighboring face's answer stands. Every query gets an answer: the import never refuses a face (an edge off its surface is certified to the deviation it achieves, and reported), a cusp edge carries a sign rule, and the search is bounded by construction, with no step cap and no \"unsure\" answer. The double-precision CPU evaluator is the reference every GPU kernel is tested against. The path-walking trim test (`solid3::classify`, `sdf`, `nurbs_face_closest`) is retired and is deleted in milestone 0 of the roadmap; until then it still runs inside `Exact` for a face whose loops were refused, a hole on the retirement list, never a precedent. `frep3` was deleted on 2026-10-30.",
  "Sign comes from topology, never from a winding number (2026-09-29; restated by Derek, 2026-11-07). At an edge the sign comes from its two faces' outward normals, at a vertex from the angle-weighted pseudo-normal. The earlier wording, \"topology decides each operator, max at a convex edge, min at a concave one\", belonged to the global CSG plan (decision 10) and has no operator left to decide.",
  "Exact input is never tessellated. No triangles, meshes or facets are made from B-rep or spline input. Meshes enter only when the mesh is the input (a scan, an STL-only file), through a mesh importer.",
  "B-splines stay native. They are never lowered to lines, arcs, biarcs or polylines, except by a post processor.",
  "The glued field and the pure field stay rejected (Derek, 2026-10-25). Neither the glued local-formula proposal in docs/FREP_GLUING.md nor its pure field (F for the sign, a distance among surface feet, intersection curves and triple points) is built, under any name. On 2026-10-25 the pure field was recorded as adopted with conditions and rejected the same day against the four guarantees; the record is decision 10.",
  "The incumbents' toolpath constructions are not our template. Engagement-controlled roughing as commercial CAM does it is a set of patented methods for a physics objective: keep tool load, chip thickness and engagement inside what the tool and machine can take, in the least time. The patents protect the methods, not the objective. The objective is open and it is ours too: pursue it through the plan field, with chip thickness as the load measure and feed free in the plan (Derek, 2026-10-26). Engagement is measured and feeds the chip model, but it is not the target, and no maximum and minimum engagement angle is chosen for the path to vary between. The methods are ruled out in any form, including fallbacks, in-repo baselines, flags, tests and plans written into docs: - offset or contour-parallel passes; - engagement held by offsetting, trimming or searching boundary curves, offset rings or cleared-area polygons; - engagement held by steering a walker step by step until the measured cut matches a target; - spirals morphed or interpolated between boundary curves; - transition or bridge moves built between offset passes; - recovering a boundary curve from a field and then doing any of these. Results may converge; intent may not. None of these constructions is ever the stated target, the reference design or the thing to match. They are also the curve machinery the F-rep replaces, and where they fail is where the field approach should differ. Trochoids and field spirals are allowed (Derek, 2026-10-26). Trochoids were ruled out for their patents, and prior art clears them: the engagement-limiting patents that claimed circular passes have expired. A plan may use trochoids by design, and spirals read off a solved field, built from fields like every other path (decision 1). They must not perform an element of a claim still live (roadmap section 10 has the rules beside these; the maintainer keeps the claim reading): - a maximum and a minimum engagement angle chosen for the path to vary between; - one portion of a region given to a diverging spiral and trochoids only to the rest; - a region cut, or ordered early, to host links: links run only through space that is already empty; - a Voronoi diagram anywhere in planning (Derek, 2026-11-07: wider than the claim, which recites a spiral traced around a wave over one); medial information comes from solved plan fields; - in 5-axis, flank positions taken from a slot's two walls, and, until May 2032, simultaneous 5-axis loops whose upper and lower patterns differ while stepping down a slot between two walls. Spirals whose stepover rises and falls along each loop rest on prior art: Bieterman and Sandstrom published field spirals with that property in 2003 (Derek, 2026-10-26).",
  "Field work is built for the GPU first (Derek, 2026-10-24). Field queries are independent per point, cell, pixel or pose, so toolpath planning, checking and visualization should be embarrassingly parallel. Before writing a loop over samples, cells, poses or pixels, decide whether it runs as a GPU kernel (wgpu and WGSL, as in `icam-gpu`); if it doesn't, say why in the pull request. A single-threaded CPU loop over independent samples is never the default. - Pick formulations that parallelize: fast sweeping or the fast iterative method over fast marching, Jacobi or multigrid over Gauss-Seidel, per-point walk-on-spheres, straight-line WGSL generated per tool, machine and span record. - Kernels run single precision in local frames, with rounding folded into the certification margin. The double-precision CPU implementation is the reference every kernel is tested against, and borderline gouge verdicts are re-decided on the CPU (roadmap section 9). - Where the CPU is right (a true sequential dependency, tiny inputs, the reference itself), use `rayon` over independent work. - Visualize on the GPU: raymarch or sphere-trace the field in a shader.",
  "The four guarantees. The goal is pure implicit CSG; it is blocked; the compromise is the certified loops (Derek, 2026-10-25, rewritten 2026-10-27 and 2026-11-07; docs/EXACT_FIELD.md). The goal was to turn B-rep constructs into pure F-rep ones through an implicit CSG, with no trims and no intersection tests: trims and intersections are what make B-rep brittle, and an implicit CSG has no holes or seams by construction. B-rep logic is never rebadged as an F-rep planner. The goal is blocked for fitted B-spline faces: they have no exact implicit (a bicubic patch implicitizes to degree 18 with extra sheets) and no separating primitives at tangent edges (8,016 in the suite, 2,105 on THE PART). The compromise is the exact evaluator over trim loops certified at import (decision 3). A rigorous global route for B-spline faces, if one is found, is preferred. The project rests on fast parallel evaluation of exact true fields with no holes and no brittle surface-intersection or B-rep logic. Guarantee 4 means that nothing in the architecture may open a hole, a seam or a failure in the exact inside/outside test everything downstream relies on, and no edge finding whose search time explodes (Derek, 2026-10-26). - Trim loops are built once at import (fitted to the edges' preimages, certified to the larger of the file's tolerance and the measured deviation) and read at a query by exact crossing parity in a cell tree: always an answer, no retry, no threshold. That is the only trim test allowed. Edges and vertices are read at a query for their own distance and sign. - A per-point search over faces, edges and vertices is allowed in the CPU evaluator, bounded by box lower bounds and the critical-point test, bounded by construction with no step cap, and its measured rate is reported. - The distance is the exact evaluator's (decision 3). Planning reads it exactly, through the GPU kernel; the brick cache is for drawing and coarse entries (decision 2). Until 2026-10-27 the plan was an eikonal redistance from F's zero set; that is no longer the route. - Still out: Newton onto surface pairs or triples, a slide along a crossing curve, bisection for where an exposed piece ends, an on-the-boundary test by a sign change across an epsilon step, a path-walking trim test, and a per-point search over intersection curves or triple points. The pure field of docs/FREP_GLUING.md was built that way and rejected on 2026-10-25. - The evaluator, judged against the guarantees (2026-10-27, restated 2026-11-07). Guarantee 1 is met by its GPU kernel (THE PART in 1 mm bricks in 2.3 s; the CPU reference is a per-point search at 2,030 to 2,850 queries a second on all threads). Guarantee 2 is met on THE PART (every `icam-check exact` check clean); the suite has 3 Lipschitz breaks and 1 sign failure in one inconsistent solid (`rear`), unflagged, on the retirement list. Guarantee 3 is not yet met: refused loops (26 suite faces, 1 on THE PART), cusp edges and the search cap are holes, closed in milestone 0, and one wrong sign carried by the brick cache (`docs/PROOFS.md`, piece 27829) is open. Guarantee 4, as Derek reads it, is met. - Before offering any field or distance method for approval, judge it in writing against the four guarantees: it evaluates as a straight-line parallel kernel (give the measured points per second); it is exact where it claims to be; it has no holes (no unresolved cells, no refused parts behind fallbacks, no thresholds that decide structure); it has no surface-intersection or B-rep logic. - A method that fails one is not offered for approval, with or without conditions, and \"approved with conditions\" never licenses building it.",
  "One planner, built for simultaneous 5-axis (Derek, 2026-11-07). Every query, plan and model takes a pose: a position and a tool axis. 3-axis, 2.5D and 3+2 are the same code with the axis held, never their own code path, planner or part model; a restriction is a value passed in. No part model exists for a class of parts: whatever the exact evaluator reads, the planner plans. Every proposal says how simultaneous 5-axis calls it; if it doesn't, it is out. The prismatic part model (`frep.rs`, `prismatic.rs`, `part.rs`, the references in `query.rs`) and the pipeline built on it are being retired (roadmap, milestone 0); edits that only lower that code's counts pass, new held-axis code does not.",
];

// File classes, by repo-relative path with forward slashes.
const IMPORT = /^crates\/icam-core\/src\/(step|solid3|surface3|curve3|prismatic|trim2)\.rs$/;
const COMPILER = /^crates\/[^/]+\/src\/(.*\/)?frep[^/]*\.rs$/;
const CHECKER = /^crates\/icam-check\//;
const GPU = /^crates\/icam-gpu\//;
// Exact paths only, added by Derek when such a module exists, so code cannot dodge a
// rule by moving to a file with a lucky name. None exist yet.
const MESH_IMPORTERS = new Set([]); // where meshes, winding numbers and STL may appear
const POST_PROCESSORS = new Set([]); // where splines may be lowered to lines and arcs
const RUST = /^crates\/.*\.rs$/;

const ROADMAP = "CONTRIBUTING.md, the design decisions; docs/ARCHITECTURE_AND_ROADMAP.md section 9 has the history";

export const RULES = [
  {
    id: "brep-outside-import",
    applies: (p) => RUST.test(p) && !IMPORT.test(p) && !COMPILER.test(p) && !CHECKER.test(p),
    patterns: [
      /\bsolid3::(?!(?:Exact|ExactStats|Sd|Feature|Solid3)\b)/g,
      /\bstep::(?!read\b)/g,
      /\b(?:Face3|Brep|BFace|OEdge|EdgeRec)\b/g,
    ],
    why: "Runtime code may build and query the exact evaluator (step::read, Solid3::from_step, solid3::Exact and its answers) but not reach into the B-rep: faces, edges, vertices and loops are the evaluator's business (a pattern sees the types and paths; field access on a Solid3 is for review).",
    decision: `2026-10-27 (Derek): the exact evaluator is the field; every stage after it reads its answers (CONTRIBUTING.md decisions 1 and 3, ${ROADMAP}).`,
  },
  {
    id: "brep-eval-in-compiler",
    applies: (p) => COMPILER.test(p),
    patterns: [/\.(?:sdf|sdf_among|classify|nurbs_face_closest)\s*\(/g],
    why: "The old path-walking reference (sdf, sdf_among, classify, nurbs_face_closest) is retired and supplies nothing to field code. The field is solid3::Exact over the certified loops; compare against the old reference only in icam-check or a #[cfg(test)] module.",
    decision: `2026-09-30 (Derek), rewritten 2026-10-27: the exact evaluator is the field and the path-walking trim test retires (CONTRIBUTING.md decision 3, ${ROADMAP}).`,
  },
  {
    id: "distance-by-intersection",
    applies: (p) => RUST.test(p) && !IMPORT.test(p) && !CHECKER.test(p),
    // Names of surface-meeting solvers and exposure tests; a machine's sliding joint
    // (Link::Slide) and a UI slider are not these.
    patterns: [/\b(?:meet2|meet3|project2|\w*[Ss]urf\w*[Ii]ntersect\w*|\w*[Ii]ntersect\w*[Ss]urf\w*|[Ss]si\w*|\w*[Bb]isect\w*|\w*[Ee]xposed\w*|\w*slide_along\w*|\w*crossing_curve\w*|\w*[Tt]riple_?[Pp]oint\w*)\b/g],
    why: "Solving where surfaces meet (Newton onto surface pairs or triples, slides along a crossing curve, bisection for where an exposed piece ends) is the brittle intersection logic the field replaces, and a per-point search over such pieces is not a kernel. The distance is the exact evaluator's (solid3::Exact: critical points in faces through certified loops, edges, vertices), read exactly by planning through its GPU kernel.",
    decision: `2026-10-25 (Derek): the four guarantees; the pure field was built this way and rejected (docs/EXACT_FIELD.md, ${ROADMAP}).`,
  },
  {
    id: "brep-geometry-in-compiler",
    applies: (p) => COMPILER.test(p),
    patterns: [/\.(?:face_distance|edge_frame|use_of)\s*\(/g, /\.curve\.(?:at|closest|bbox|start|end|bez_pieces|plane_crossings)\s*\(/g, /\.verts\s*\[/g],
    why: "A trim test (face_distance), a frame read off a stored edge curve, or a stored curve's or vertex's geometry decides nothing in the compiler: the surfaces and the topology (which faces an edge joins, which way it turns) are all it reads. Existing uses are slated for replacement, never a precedent.",
    decision: `2026-10-25 (Derek): no B-rep checks in the compiler (docs/EXACT_FIELD.md, ${ROADMAP}).`,
  },
  {
    id: "formula-by-probe",
    applies: (p) => COMPILER.test(p),
    patterns: [/\b(?:[Pp]robe\w*|PROBE\w*|candidates?)\b/g],
    why: "Choosing a cell's formula by sampling probes against the B-rep is the per-cell design that was reversed. The field is one global function, the exact signed distance; no cell holds a formula of its own.",
    decision: `2026-10-02 (Derek): the F-rep must be one global function; the 2026-09-30 per-cell plan is reversed (CONTRIBUTING.md decision 2, ${ROADMAP}).`,
  },
  {
    id: "brep-in-compiled-field",
    applies: (p) => COMPILER.test(p),
    // Checked structurally in check.mjs: a pub struct whose fields name a B-rep type.
    structural: "pub-struct-holds-brep",
    patterns: [],
    why: "A compiled field (the prismatic compiler, being retired) must be self-contained: nothing evaluated at runtime may hold or reach the B-rep. Runtime code reads the exact evaluator's answers, never B-rep types.",
    decision: `2026-09-29 (Derek): B-rep is an import format; 2026-10-27: the exact evaluator is the field and every stage after it reads its answers (CONTRIBUTING.md decisions 1 and 3, ${ROADMAP}).`,
  },
  {
    id: "trim-loops-in-field",
    applies: (p) => COMPILER.test(p) || GPU.test(p),
    patterns: [/\.loops\b/g, /\.use_of\s*\(/g, /\b(?:in_face|point_in_face|inside_face|trim_test)\b/g],
    why: "The B-rep's own loops (.loops, use_of) and walked point-in-face tests stay out of field code. The only trim test is the certified parameter-domain loops (trim2), read through solid3::Exact.",
    decision: `2026-09-30 (Derek), rewritten 2026-10-27: the certified loops are the only trim test (CONTRIBUTING.md decision 10, ${ROADMAP}).`,
  },
  {
    id: "winding-number",
    applies: (p) => RUST.test(p) && !MESH_IMPORTERS.has(p),
    patterns: [/\b\w*[Ww]inding\w*\b/g],
    why: "The sign comes from topology (the faces' outward normals at an edge, the pseudo-normal at a vertex), never from a winding count. Winding numbers belong only in a mesh or point-cloud importer.",
    decision: `2026-09-29 (Derek), restated 2026-11-07: sign from topology, never a winding number (CONTRIBUTING.md decision 4, ${ROADMAP}).`,
  },
  {
    id: "tessellate-exact-input",
    applies: (p) => RUST.test(p) && !MESH_IMPORTERS.has(p),
    // egui's own UI tessellation (its panels' text and widgets, never part geometry) is
    // not an input being tessellated; it is exempt only when called by its full path,
    // egui::Context::tessellate, so no other tessellator can pass by a lucky name
    // (Derek, 2026-10-30).
    patterns: [/(?<!\begui::Context::)\b\w*(?:[Tt]essellat|[Tt]riangulat|[Ff]acet)\w*\b/g, /\b(?:Triangle|Tri|TriMesh|Mesh|mesh|stl|Stl|STL)\b/g],
    why: "An exact input is never tessellated first. Meshes enter only through a mesh importer, and only when the mesh is the input.",
    decision: `2026-09-29 (Derek): an exact input is never tessellated first; an earlier demo's triangle-shadow clearance is the anti-pattern (CONTRIBUTING.md decision 5, ${ROADMAP}).`,
  },
  {
    id: "spline-lowered",
    applies: (p) => RUST.test(p) && !POST_PROCESSORS.has(p),
    patterns: [/\b\w*(?:[Bb]iarc|to_arcs|arc_fit|fit_arcs?|to_lines|[Pp]olyline|[Pp]olygoni[sz]|[Ll]ineari[sz])\w*\b/g],
    why: "B-splines stay exact until a post processor forces a lower form. Use the native rational Bezier pieces (bspline.rs, curve3.rs).",
    decision: `2026-09-30 (Derek): "B-splines stay native"; most CAM engines lower splines to lines and arcs, ours does not (CONTRIBUTING.md decision 6, ${ROADMAP}).`,
  },
];

/** Files Claude may not change without Derek approving the prompt. */
export const GUARD_FILES = [
  /^tools\/frep-guard\//,
  /^\.githooks\//,
  /^\.claude\/settings(\.local)?\.json$/,
];
