#!/usr/bin/env node
// fetch-references.mjs - keep whitepaper-references/ in step with references.json.
//
// Written 2026-09-29. The list lives in whitepaper-references/references.json, not in this
// file, and MANIFEST.md is generated from it. To add a paper, add an entry there and rerun.
//
// What this tool will and will not do
// -----------------------------------
// It downloads only copies that authors, their institutions, open archives (arXiv), or
// the publisher have put on the open web. It sends no credentials, no cookies, and no
// proxy or resolver, and it never touches a paid endpoint. Entries with access.kind
// "none" have no open copy; the tool fetches nothing for them and says so.
//
// Entries with access.kind "browser" ARE open, but their host answers scripted clients
// with a bot challenge (ACM and Wiley use Cloudflare, HAL uses Anubis proof-of-work).
// The tool does not solve, evade, or replay those challenges. It prints the URL; you
// open it in a browser and save the PDF under the listed file name, and the next run
// hashes it like any other file.
//
// For access.kind "http" it tries Node's fetch first, with browser User-Agent and Accept
// headers. If that returns something other than a PDF, it retries the same URL once with
// the system curl and says so on stdout. That is a different HTTP client, not a way
// around a challenge: if curl also gets HTML, the entry is recorded as failed.
//
// Usage (from the ImplicitCAM root)
//   node tools/fetch-references.mjs            fetch what is missing, hash everything, rewrite MANIFEST.md
//   node tools/fetch-references.mjs --verify   no network: re-hash what is on disk, compare, exit 1 on mismatch
//   node tools/fetch-references.mjs --accept   like the default, but accept changed hashes instead of failing
//
// Nothing is repaired quietly. A file whose hash no longer matches the recorded one is
// reported and left alone unless --accept is passed.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const DIR = join(ROOT, 'whitepaper-references');
const LIST = join(DIR, 'references.json');
const MANIFEST = join(DIR, 'MANIFEST.md');

const MIN_PDF_BYTES = 16 * 1024;
const TIMEOUT_MS = 180_000;
const HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  Accept: 'application/pdf,application/octet-stream;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.9',
};

const GROUPS = [
  ['eval', 'Fast evaluation of implicit fields', 'Interval and affine arithmetic, Lipschitz bounds, tape and tree pruning, tracing. The hot loop: how fast and how safely a field can be queried.'],
  ['dist', 'Distance fields: representation, construction, exactness', 'F-rep and R-functions, which operations keep a true distance (the Euclidean crux), sparse and adaptive storage, mesh to field import.'],
  ['pde', 'Field and PDE solvers on implicit domains', 'Eikonal fronts, heat-method distances, and grid-free Monte Carlo solvers that run directly on an SDF. The engines behind field plans.'],
  ['sim', 'Stock and machining simulation', 'Dexel, tri-dexel, voxel, and distance-field stock; the in-process stock model and the checker.'],
  ['path', 'Toolpath planning with fields', 'Offsets as level sets, PDE and medial-axis spirals, and the 2025-2026 implicit-field multi-axis planners.'],
  ['mach', 'Machining physics and motion', 'Engagement control, chatter, and trajectory limits. What the objective function has to respect.'],
];
const PRIORITY = { core: 'core', supporting: 'supporting', background: 'background' };

const args = new Set(process.argv.slice(2));
const VERIFY = args.has('--verify');
const ACCEPT = args.has('--accept');
const today = new Date().toISOString().slice(0, 10);

const data = JSON.parse(readFileSync(LIST, 'utf8'));
const refs = data.references;
let problems = 0;

function sha256(buf) { return createHash('sha256').update(buf).digest('hex'); }
function pdfGate(buf) {
  if (buf.length < MIN_PDF_BYTES) return `only ${buf.length} bytes`;
  if (buf.subarray(0, 5).toString('latin1') !== '%PDF-') return 'no %PDF- header (got ' + JSON.stringify(buf.subarray(0, 15).toString('latin1')) + ')';
  if (!buf.subarray(Math.max(0, buf.length - 4096)).toString('latin1').includes('%%EOF')) return 'no %%EOF in the last 4096 bytes (truncated?)';
  return null;
}

async function viaFetch(url) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { headers: HEADERS, redirect: 'follow', signal: ctl.signal });
    return { status: r.status, buf: Buffer.from(await r.arrayBuffer()) };
  } finally { clearTimeout(t); }
}
function viaCurl(url, out) {
  const h = Object.entries(HEADERS).flatMap(([k, v]) => ['-H', `${k}: ${v}`]);
  execFileSync('curl', ['-sSL', '--max-time', String(TIMEOUT_MS / 1000), ...h, '-o', out, url], { stdio: ['ignore', 'ignore', 'pipe'] });
  return readFileSync(out);
}

async function download(ref) {
  const dest = join(DIR, ref.file);
  const tmp = join(tmpdir(), `fetch-ref-${process.pid}-${ref.n}.part`);
  let buf, why;
  try {
    const r = await viaFetch(ref.access.url);
    buf = r.buf;
    why = r.status >= 400 ? `HTTP ${r.status}` : pdfGate(buf);
  } catch (e) { why = 'fetch error: ' + e.message; }
  if (why) {
    console.log(`  #${ref.n} Node fetch did not return a PDF (${why}); retrying once with the system curl`);
    try { buf = viaCurl(ref.access.url, tmp); why = pdfGate(buf); } catch (e) { why = 'curl error: ' + String(e.message).split('\n')[0]; }
  }
  if (existsSync(tmp)) rmSync(tmp);
  if (why) return why;
  writeFileSync(dest + '.part', buf);
  renameSync(dest + '.part', dest);
  return null;
}

function pageCounts(files) {
  if (!files.length) return {};
  const py = 'import json,sys\nfrom pypdf import PdfReader\nout={}\nfor p in sys.argv[1:]:\n  try: out[p]=len(PdfReader(p).pages)\n  except Exception as e: out[p]=None\nprint(json.dumps(out))';
  for (const exe of ['python', 'python3', 'py']) {
    try { return JSON.parse(execFileSync(exe, ['-c', py, ...files], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })); } catch { /* try next */ }
  }
  console.log('  page counts skipped: no python with pypdf on PATH');
  return {};
}

// 1. Fetch what is missing.
if (!VERIFY) {
  for (const ref of refs) {
    if (ref.access.kind !== 'http' || existsSync(join(DIR, ref.file))) continue;
    console.log(`#${ref.n} fetching ${ref.file}`);
    const err = await download(ref);
    if (err) { console.log(`  #${ref.n} FAILED: ${err}. Open ${ref.access.url} by hand.`); ref.lastError = `${today}: ${err}`; }
    else { console.log(`  #${ref.n} ok`); delete ref.lastError; }
  }
}

// 2. Hash everything on disk and compare with what was recorded.
const present = refs.filter(r => existsSync(join(DIR, r.file)));
const pages = pageCounts(present.map(r => join(DIR, r.file)));
for (const ref of refs) {
  const p = join(DIR, ref.file);
  if (!existsSync(p)) {
    if (ref.got) { console.log(`#${ref.n} MISSING ${ref.file} (recorded sha256 ${ref.got.sha256.slice(0, 12)}...)`); if (VERIFY) problems++; }
    continue;
  }
  const buf = readFileSync(p);
  const gate = pdfGate(buf);
  if (gate) { console.log(`#${ref.n} NOT A VALID PDF ${ref.file}: ${gate}`); problems++; continue; }
  const now = { sha256: sha256(buf), bytes: buf.length, pages: pages[p] ?? null, verified: today };
  if (ref.got && ref.got.sha256 !== now.sha256) {
    console.log(`#${ref.n} CHANGED ${ref.file}: recorded ${ref.got.sha256.slice(0, 12)}..., on disk ${now.sha256.slice(0, 12)}...`);
    if (!ACCEPT) { problems++; continue; }
  }
  ref.got = { ...now, via: ref.got && ref.got.sha256 === now.sha256 ? ref.got.via : (ref.access.kind === 'http' ? 'tool' : 'by hand') };
}

// 3. PDFs in the folder that no entry claims.
const claimed = new Set(refs.map(r => r.file.toLowerCase()));
const unlisted = readdirSync(DIR).filter(f => f.toLowerCase().endsWith('.pdf') && !claimed.has(f.toLowerCase()));
for (const f of unlisted) console.log(`UNLISTED ${f}: add an entry to references.json (or ask Claude to "add it to the collection")`);

// Papers on disk that no distilled skill covers yet (data.skills lists which entries each skill holds).
const distilled = new Set((data.skills || []).flatMap(s => s.entries));
const undistilled = refs.filter(r => existsSync(join(DIR, r.file)) && !distilled.has(r.n));
for (const r of undistilled) console.log(`NOT DISTILLED #${r.n} ${r.file}: fold it into a skill (book-to-skill Mode 4)`);

// 4. Status per entry, then write references.json and MANIFEST.md.
function status(ref) {
  const on = existsSync(join(DIR, ref.file)) && ref.got;
  if (on) return ref.access.kind === 'http' ? 'acquired' : ref.access.kind === 'browser' ? 'saved from browser' : 'supplied by hand';
  if (ref.access.kind === 'http') return ref.lastError ? 'fetch failed' : 'not fetched yet';
  if (ref.access.kind === 'browser') return 'browser download needed';
  return 'no open copy';
}
const counts = {};
for (const r of refs) { r.status = status(r); counts[r.status] = (counts[r.status] || 0) + 1; }

if (!VERIFY) {
  writeFileSync(LIST, JSON.stringify(data, null, 2) + '\n');
  writeFileSync(MANIFEST, renderManifest());
  console.log(`wrote references.json and MANIFEST.md (${refs.length} entries)`);
}
console.log(Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ') + (unlisted.length ? `, ${unlisted.length} unlisted` : ''));
if (problems) { console.log(`${problems} problem(s); see above`); process.exit(1); }

function cell(s) { return String(s ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' '); }
function idLink(r) {
  if (r.doi) return `[${r.doi}](https://doi.org/${r.doi})`;
  if (r.arxiv) return `[arXiv:${r.arxiv}](https://arxiv.org/abs/${r.arxiv})`;
  return 'none';
}
function getIt(r) {
  if (r.access.kind === 'none') return cell(r.access.note);
  return `[${r.access.kind === 'browser' ? 'open in browser' : 'open copy'}](${r.access.url}). ${cell(r.access.note)}`;
}
function renderManifest() {
  const L = [];
  L.push('# Reference papers - manifest', '');
  L.push(`Generated by \`tools/fetch-references.mjs\` from \`references.json\` on ${today}. Do not edit this`,
    'file by hand; edit `references.json` and rerun the tool.', '');
  L.push('## What this collection is for', '',
    'The state of the art in two things ImplicitCAM depends on: evaluating implicit fields and signed',
    'distance fields fast and safely, and CAM methods that already work in, or would improve inside,',
    'an implicit framework. Each entry says in one line why it matters here. A verified manifest, a',
    'fetch tool, and knowledge-base skills distilled from the papers (listed under "Skills distilled',
    'from this set").', '');
  L.push('## The PDFs are not meant to be redistributed', '',
    '`whitepaper-references/*.pdf` is ignored by `whitepaper-references/.gitignore`. ImplicitCAM is',
    'meant to be open source, and most of these are publisher PDFs or author copies posted under',
    'terms that let the authors share them, not us. What belongs in the repository is `references.json`,',
    'this manifest, and the tool. Anyone can rebuild the folder with `node tools/fetch-references.mjs`',
    'and check the bytes against the sha256 column.', '');
  L.push('## How the list was produced', '',
    'Seeded 2026-09-29 from the ImplicitCAM design docs, the works-cited list of the Gemini report',
    '`Functional Implicit and SDF Kernels for Subtractive CAM.pdf`, and targeted searches for 2024-2026',
    'work. Every entry with a DOI was resolved against the Crossref REST API (authors, title, venue,',
    'volume, issue, pages, year); arXiv entries against the arXiv API; the one JCGT paper against',
    'jcgt.org, since JCGT issues no DOIs. Open-access locations came from OpenAlex, then from author',
    'and lab pages, and every "open copy" URL was checked to return a real PDF. Where Crossref gives an',
    'online-first year that differs from the issue year, the citation uses the issue year and says so.', '');
  L.push('## Status vocabulary', '', '| status | meaning |', '| --- | --- |',
    '| `acquired` | the tool fetched an open copy and it passed the PDF gate (`%PDF-` header, at least 16,384 bytes, `%%EOF` in the last 4,096 bytes). |',
    '| `saved from browser` | open, but the host blocks scripted clients; saved by hand from a browser, then hashed by the tool. |',
    '| `browser download needed` | open (ACM since 2026-01-01, HAL, Wiley open access, PNAS), but the host answers scripts with a bot challenge. Open the link, save the PDF under the listed file name, rerun the tool. |',
    '| `no open copy` | OpenAlex lists none and none was found. Buy, borrow, or use an institutional subscription; drop the file in and rerun. |',
    '| `supplied by hand` | a closed paper that has been bought or borrowed and dropped in. The tool hashes it but cannot re-acquire it. |',
    '| `fetch failed` / `not fetched yet` | an open copy that did not arrive; the error is in `references.json` under `lastError`. |', '');
  L.push('## Adding a paper', '',
    '1. Drop the PDF in this folder (any name). The tool reports it as UNLISTED.',
    '2. Ask Claude to "add it to the collection": it resolves the citation, adds an entry to',
    '   `references.json` with the next free number, renames the file, and reruns the tool.',
    '3. Entry numbers are permanent. Do not renumber when inserting; new entries take the next number.', '');
  const total = refs.length;
  L.push('## Summary', '', `${total} entries. ` + Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ') + '.', '');
  const core = refs.filter(r => r.priority === 'core');
  L.push(`Priority: ${core.length} core, ${refs.filter(r => r.priority === 'supporting').length} supporting, ${refs.filter(r => r.priority === 'background').length} background. Core means it bears directly on a current ImplicitCAM decision or phase.`, '');
  if (data.skills && data.skills.length) {
    L.push('## Skills distilled from this set', '',
      'Personal Claude Code skills in `~/.claude/skills/<name>/`, outside every repository for the same',
      'reason the PDFs are: they are derived from publisher material. Regenerate rather than copy them.',
      'When a paper lands on disk, fold it into its skill with `book-to-skill` Mode 4 and add its number here',
      '(the tool prints NOT DISTILLED for any paper on disk that no skill lists).', '',
      '| Skill | Built | Entries |', '| --- | --- | --- |');
    for (const s of data.skills) L.push(`| \`${s.name}\` | ${s.built} | ${s.entries.join(', ')} |`);
    L.push('');
  }
  for (const [g, title, blurb] of GROUPS) {
    const rows = refs.filter(r => r.group === g).sort((a, b) => (a.priority === b.priority ? a.year - b.year : ['core', 'supporting', 'background'].indexOf(a.priority) - ['core', 'supporting', 'background'].indexOf(b.priority)));
    if (!rows.length) continue;
    L.push(`## ${title}`, '', blurb, '', '| # | Citation | ID | Why it matters here | Priority | Status | Get it |', '| --- | --- | --- | --- | --- | --- | --- |');
    for (const r of rows) L.push(`| ${r.n} | ${cell(r.citation)} | ${idLink(r)} | ${cell(r.why)} | ${PRIORITY[r.priority] || r.priority} | ${r.status} | ${getIt(r)} |`);
    L.push('');
  }
  L.push('## Fingerprints', '', 'Byte counts are bare integers on purpose: `--verify` compares them. Pages were counted with pypdf.', '',
    '| # | Local file | sha256 | Bytes | Pages | Via | Verified |', '| --- | --- | --- | --- | --- | --- | --- |');
  for (const r of refs) {
    if (!r.got) { L.push(`| ${r.n} | \`${r.file}\` (absent) | - | - | - | - | - |`); continue; }
    L.push(`| ${r.n} | \`${r.file}\` | \`${r.got.sha256}\` | ${r.got.bytes} | ${r.got.pages ?? '-'} | ${r.got.via} | ${r.got.verified} |`);
  }
  L.push('');
  return L.join('\n');
}
