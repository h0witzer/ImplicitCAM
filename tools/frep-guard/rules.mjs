// frep-guard rules: the B-rep constructions this repository has decided against, as
// patterns over Rust code (comments and string contents are stripped first).
//
// Every rule cites the decision it enforces. The decisions live in
// docs/ARCHITECTURE_AND_ROADMAP.md (section 9 amendments); this file only makes them
// mechanical. The prompt hook in .claude/settings.json restates DECISIONS for the
// cases a pattern cannot see (a per-cell formula looks like a pruned tree in code);
// keep the two in step when a decision changes.
//
// There are no allowances (baseline.json, 2026-10-25): code over a rule is slated for
// replacement and stays reported until it is replaced. Nothing may be added, and an
// edit to such a file may only lower its count.

/** The decisions, in the words the session-start hook injects. */
export const DECISIONS = [
  "F-rep is the only internal representation (2026-09-29). Every stage after import (planning, gouge and collision checks, engagement, medial axis, reach, certification) reads fields, never faces, loops or B-rep topology. The exact evaluator (solid3::Exact, decision 3) is the field: it reads the exact surfaces, certified loops, edges and vertices inside solid3 and trim2, and every stage after it reads its answers (distance, nearest point, feature).",
  "The part's field is its exact signed distance, one global function (Derek, 2026-10-27; rewrites 'compiled to ONE global F-rep', 2026-09-29). It is evaluated pointwise from the exact surfaces and the trim loops certified at import (solid3::Exact over trim2), and the global field for planning and viewing is that evaluation sampled into the sparse brick cache. No CSG tree is compiled. Per-cell formulas chosen by probes, clamps or local rules stay reversed (2026-10-02; frep3.rs: 244 seams up to 945 mm on a test nest).",
  "The exact evaluator is the field; the old trim walks retire (Derek, 2026-10-27; rewrites 'solid3 is the import-time reference, not the runtime path', 2026-09-30). solid3::Exact gives the signed distance: a face's nearest critical point when its certified loops put it in the face, else its edges and vertices, signed by outward normals and pseudo-normals. The file's declared tolerance is the accuracy line at edges (2026-10-26). What it cannot decide it marks unsure, never guesses (refused loops, cusp edges, a search at its cap). The double-precision CPU evaluator is the reference every GPU kernel is tested against. The path-walking trim test (solid3::classify, sdf, nurbs_face_closest) and frep3 are retired: no new code uses them outside icam-check comparisons.",
  "B-rep topology decides each operator (max at a convex edge, min at a concave one), never a winding count (2026-09-29).",
  "An exact input is never tessellated first (2026-09-29); triangle-shadow clearance is the named anti-pattern. Meshes enter only when the mesh is the input (a scan, an STL-only file).",
  "B-splines stay native end to end (2026-09-30); only a post processor may lower them to lines and arcs.",
  "The glued local-formula proposal (docs/FREP_GLUING.md) stays rejected: no fade, cap or glue terms, no one-leaf evaluation, anywhere outside icam-check's comparison baselines. The exact field (FREP_GLUING.md's pure field) is adopted with conditions (Derek, 2026-10-25; docs/EXACT_FIELD.md): F evaluates as its whole formula, in value and not only sign; queries read compiled data and F only, never a vertex, edge curve, face list, loop or solid3 call; formulas come from rules, never from the B-rep's answers (no probes, no certificate checked against the reference, no solid3 signs for empty cells, no point-in-face operator choice). The tests listed there must stay green.",
  "STOP before writing any toolpath code: your first idea will be the textbook construction commercial CAM uses, and it is not our template (2026-10-23, revised 2026-10-26). The patents protect methods, not objectives: physics objectives (tool load, chip thickness, stability, time) are ours too and may shape the plan field by field methods, with chip thickness as the load measure and feed free in the plan; engagement is measured, never the target. The constructions are ruled out in any form, including fallbacks, in-repo baselines, flags, tests and plans: offset or contour-parallel passes; engagement held by offsetting, trimming or searching boundary curves, offset rings or cleared-area polygons; engagement held by steering a walker step by step until the measured cut matches a target; spirals morphed between boundary curves; bridge moves between offset passes; recovering a boundary curve from a field and doing any of these. Results may converge; intent may not (never name an incumbent construction as the target). Trochoids and field spirals are allowed (Derek, 2026-10-26: prior art clears them), built from fields, but never perform an element of a live claim: a maximum and minimum engagement angle chosen for the path to vary between; a region split into a spiral part and a trochoid-only remainder; a region cut or ordered early to host links (links run only through space already empty); a spiral traced over a Voronoi diagram; in 5-axis, flank positions taken from a slot's two walls, or, until May 2032, loops whose upper and lower patterns differ while stepping down a slot between two walls (CONTRIBUTING.md decision 8). If the field-native way is not obvious, stop and ask Derek; never fall back.",
  "GPU first (2026-10-24). Field queries are independent per point, cell, pixel or pose, so planning, checking and visualization should be embarrassingly parallel. Before writing any loop over samples, cells, poses or pixels, decide whether it runs as a GPU kernel (wgpu/WGSL, as in icam-gpu); if not, say why. Never default to a single-threaded CPU loop over independent samples: pick parallel formulations (fast sweeping or fast iterative over fast marching, Jacobi or multigrid over Gauss-Seidel, per-point walk-on-spheres), use rayon where the CPU is right, and visualize fields in shaders. The double-precision CPU implementation stays the reference every kernel is tested against.",
  "The four guarantees, and no intersection logic (Derek, 2026-10-25, rewritten 2026-10-27; docs/EXACT_FIELD.md). The project rests on fast parallel evaluation of exact true fields with no holes and no brittle surface-intersection or B-rep logic. Guarantee 4 means nothing in the architecture may open holes or failures in the exact inside/outside test everything downstream relies on, and no edge finding whose search time explodes (Derek, 2026-10-26). Allowed (2026-10-27): trim loops built once at import, certified within the file's tolerance, read at a query by exact crossing parity (trim2), the only trim test; edges and vertices read at a query for their own distance and sign; a per-point search over faces, edges and vertices in the CPU evaluator, bounded by box lower bounds and the critical-point test, with a step cap whose hit marks the answer unsure. The distance is the exact evaluator's, sampled into the brick cache (no longer an eikonal redistance from F). Still NEVER: Newton onto surface pairs or triples, sliding along a crossing curve, bisection for where an exposed piece ends, an on-the-boundary test by a sign change across an epsilon step, a path-walking trim test, a per-point search over intersection curves or triple points. The pure field of docs/FREP_GLUING.md was built exactly that way and rejected on 2026-10-25. Before offering ANY field or distance method for approval, judge it in writing against the four guarantees: (1) it evaluates as a straight-line parallel kernel, with the measured points per second; (2) it is exact where it claims to be; (3) it has no holes (no unresolved cells, no refused parts hidden as fallbacks, no thresholds that decide structure); (4) it has no surface-intersection or B-rep logic. A method that fails one is not offered for approval, with or without conditions, and 'approved with conditions' never licenses building it. The exact evaluator was judged so on 2026-10-27 before Derek adopted it: guarantee 1 met only with the step-5 GPU kernel, guarantee 3 met except flagged holes.",
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

const ROADMAP = "docs/ARCHITECTURE_AND_ROADMAP.md section 9";

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
    why: "Solving where surfaces meet (Newton onto surface pairs or triples, slides along a crossing curve, bisection for where an exposed piece ends) is the brittle intersection logic the field replaces, and a per-point search over such pieces is not a kernel. The distance is the exact evaluator's (solid3::Exact: critical points in faces through certified loops, edges, vertices), sampled into the brick cache.",
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
    why: "Choosing a cell's formula by sampling probes against the B-rep is the per-cell design that was reversed. A cell holds a pruned restriction of the one global tree.",
    decision: `2026-10-02 (Derek): the F-rep must be one global function; the 2026-09-30 per-cell plan is reversed (${ROADMAP}, "version 2 plan").`,
  },
  {
    id: "brep-in-compiled-field",
    applies: (p) => COMPILER.test(p),
    // Checked structurally in check.mjs: a pub struct whose fields name a B-rep type.
    structural: "pub-struct-holds-brep",
    patterns: [],
    why: "The compiled field must be self-contained: nothing evaluated at runtime may hold or reach the B-rep. Builders that hold &Solid3 must stay private to compilation.",
    decision: `2026-09-29 amendment: B-rep is an import format, compiled to one global F-rep (${ROADMAP}).`,
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
    why: "Operators come from B-rep topology (max at a convex edge, min at a concave one), not from a winding count. Winding numbers belong only in a mesh or point-cloud importer.",
    decision: `2026-09-29 amendment: "The B-rep topology decides the operator, not a winding count" (${ROADMAP}).`,
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
    decision: `2026-09-29 amendment: "An exact input is never tessellated first; an earlier demo's triangle-shadow clearance is the anti-pattern" (${ROADMAP}).`,
  },
  {
    id: "spline-lowered",
    applies: (p) => RUST.test(p) && !POST_PROCESSORS.has(p),
    patterns: [/\b\w*(?:[Bb]iarc|to_arcs|arc_fit|fit_arcs?|to_lines|[Pp]olyline|[Pp]olygoni[sz]|[Ll]ineari[sz])\w*\b/g],
    why: "B-splines stay exact until a post processor forces a lower form. Use the native rational Bezier pieces (bspline.rs, curve3.rs).",
    decision: `2026-09-30 (Derek): "B-splines stay native"; most CAM engines lower splines to lines and arcs, ours does not (${ROADMAP}).`,
  },
];

/** Files Claude may not change without Derek approving the prompt. */
export const GUARD_FILES = [
  /^tools\/frep-guard\//,
  /^\.githooks\//,
  /^\.claude\/settings(\.local)?\.json$/,
];
