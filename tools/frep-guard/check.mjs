#!/usr/bin/env node
// frep-guard: refuses B-rep constructions where the F-rep is required.
//
// The rules and the decisions they enforce are in rules.mjs. Existing uses are
// allowed per file and rule by baseline.json; anything beyond that allowance is a
// violation. One checker, several entry points:
//
//   node tools/frep-guard/check.mjs                       report the tree; exit 1 if over
//   node tools/frep-guard/check.mjs --staged              git pre-commit (.githooks/pre-commit)
//   node tools/frep-guard/check.mjs --tighten             lower allowances to what is left
//   node tools/frep-guard/check.mjs --accept --reason ".." Derek only: allow the current tree
//   node tools/frep-guard/check.mjs --hook <name>         Claude Code hooks (.claude/settings.json):
//       pre-edit       PreToolUse Edit|Write|MultiEdit: deny an edit that adds a violation
//       bash           PreToolUse Bash|PowerShell: deny skipping git hooks, ask before
//                      touching the guard
//       stop           Stop: re-check the tree, catching edits made through the shell
//       session-start  SessionStart: put the decisions in context, including after compaction

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { DECISIONS, GUARD_FILES, RULES } from "./rules.mjs";

const ROOT = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));
const BASELINE = path.join(ROOT, "tools", "frep-guard", "baseline.json");
const BASELINE_REL = "tools/frep-guard/baseline.json";
const BREP_TYPE = /\b(?:Solid3|Face3|Brep|BFace|OEdge|EdgeRec)\b|\b(?:solid3|step)::/;
const ACCEPT_CMD = 'node tools/frep-guard/check.mjs --accept --reason "..."';

// ---------------------------------------------------------------- scanning

const isIdent = (c) => c !== undefined && /[A-Za-z0-9_]/.test(c);

/** Rust source with comments and string contents blanked, line breaks kept. */
export function strip(src) {
  const out = [];
  const n = src.length;
  const blank = (c) => (c === "\n" ? "\n" : " ");
  let i = 0;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === "/" && d === "/") {
      while (i < n && src[i] !== "\n") (out.push(" "), i++);
      continue;
    }
    if (c === "/" && d === "*") {
      let depth = 0;
      do {
        if (src[i] === "/" && src[i + 1] === "*") (depth++, out.push(" ", " "), (i += 2));
        else if (src[i] === "*" && src[i + 1] === "/") (depth--, out.push(" ", " "), (i += 2));
        else (out.push(blank(src[i])), i++);
      } while (i < n && depth > 0);
      continue;
    }
    if ((c === "r" || (c === "b" && d === "r")) && !isIdent(src[i - 1])) {
      let j = i + (c === "b" ? 2 : 1);
      let hashes = 0;
      while (src[j] === "#") (hashes++, j++);
      if (src[j] === '"') {
        for (let k = i; k <= j; k++) out.push(src[k]);
        i = j + 1;
        const close = '"' + "#".repeat(hashes);
        while (i < n && !src.startsWith(close, i)) (out.push(blank(src[i])), i++);
        for (let k = 0; k < close.length && i < n; k++) (out.push(src[i]), i++);
        continue;
      }
    }
    if (c === '"') {
      out.push('"');
      i++;
      while (i < n && src[i] !== '"') {
        if (src[i] === "\\") (out.push(" "), i++);
        if (i < n) (out.push(blank(src[i])), i++);
      }
      if (i < n) (out.push('"'), i++);
      continue;
    }
    if (c === "'") {
      const m = /^'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'/.exec(src.slice(i, i + 14));
      if (m) {
        out.push("'", ..." ".repeat(m[0].length - 2), "'");
        i += m[0].length;
        continue;
      }
    }
    out.push(c);
    i++;
  }
  return out.join("");
}

/** Index just past the bracket matching the one at `open`, or -1. */
function matchBracket(code, open) {
  const pair = { "{": "}", "(": ")" };
  const want = pair[code[open]];
  let depth = 0;
  for (let i = open; i < code.length; i++) {
    if (code[i] === code[open]) depth++;
    else if (code[i] === want && --depth === 0) return i + 1;
  }
  return -1;
}

/** Violations in one file: { rule, line, text, src }. Test modules are not scanned. */
export function scan(rel, text) {
  const rules = RULES.filter((r) => r.applies(rel));
  if (rules.length === 0) return [];
  const code = strip(text.replace(/\r\n/g, "\n"));
  const lines = code.split("\n");
  const raw = text.replace(/\r\n/g, "\n").split("\n");
  let end = lines.findIndex((l) => /^\s*#\[cfg\(test\)\]/.test(l));
  if (end < 0) end = lines.length;
  const hits = [];
  for (const r of rules) {
    for (let li = 0; li < end; li++) {
      for (const re of r.patterns) {
        for (const m of lines[li].matchAll(re)) hits.push({ rule: r.id, line: li + 1, text: m[0], src: raw[li].trim() });
      }
    }
    if (r.structural === "pub-struct-holds-brep") {
      const head = lines.slice(0, end).join("\n");
      for (const m of head.matchAll(/\bpub(?:\([^)]*\))?\s+(?:struct|enum)\s+(\w+)[^{(;]*([{(;])/g)) {
        const open = m.index + m[0].length - 1;
        if (head[open] === ";") continue;
        const close = matchBracket(head, open);
        const body = head.slice(open, close < 0 ? head.length : close);
        if (BREP_TYPE.test(body)) {
          const line = head.slice(0, m.index).split("\n").length;
          hits.push({ rule: r.id, line, text: `pub ${m[1]} holds ${body.match(BREP_TYPE)[0]}`, src: raw[line - 1].trim() });
        }
      }
    }
  }
  return hits.sort((a, b) => a.line - b.line);
}

const countBy = (hits) => hits.reduce((o, h) => ((o[h.rule] = (o[h.rule] || 0) + 1), o), {});

// ---------------------------------------------------------------- files and baseline

function rustFiles() {
  const dir = path.join(ROOT, "crates");
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { recursive: true })
    .map((f) => String(f).split(path.sep).join("/"))
    .filter((f) => f.endsWith(".rs") && !f.split("/").includes("target"))
    .map((f) => `crates/${f}`);
}

function relPath(p) {
  if (!p) return null;
  const abs = path.resolve(process.env.CLAUDE_PROJECT_DIR || ROOT, p);
  const rel = path.relative(ROOT, abs);
  if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) return null;
  return rel.split(path.sep).join("/");
}

function loadBaseline(text) {
  try {
    const b = JSON.parse(text ?? readFileSync(BASELINE, "utf8"));
    return { allow: b.allow || {}, log: b.log || [], note: b.note };
  } catch {
    return { allow: {}, log: [] };
  }
}

function saveBaseline(b) {
  const sorted = Object.fromEntries(Object.keys(b.allow).sort().map((k) => [k, b.allow[k]]));
  const note =
    "Per-file allowances for frep-guard rules (tools/frep-guard/rules.mjs). There are none (Derek, 2026-10-25): " +
    "code over a rule is slated for replacement, not grandfathered; the tree reports it until it is replaced, a commit that touches it must lower its count, " +
    "and an edit may only lower it. The replacement list is in docs/EXACT_FIELD.md.";
  writeFileSync(BASELINE, JSON.stringify({ note, allow: sorted, log: b.log }, null, 2) + "\n");
}

/** Every (file, rule) over its allowance, given file contents. */
function overages(files, baseline) {
  const out = [];
  for (const [rel, text] of files) {
    const hits = scan(rel, text);
    const counts = countBy(hits);
    const allow = baseline.allow[rel] || {};
    for (const [rule, n] of Object.entries(counts)) {
      if (n > (allow[rule] || 0)) out.push({ rel, rule, n, allowed: allow[rule] || 0, hits: hits.filter((h) => h.rule === rule) });
    }
  }
  return out;
}

const treeFiles = () => rustFiles().map((rel) => [rel, readFileSync(path.join(ROOT, rel), "utf8")]);

// ---------------------------------------------------------------- messages

const ruleOf = (id) => RULES.find((r) => r.id === id);

function explain(rule) {
  const r = ruleOf(rule);
  return `  Why: ${r.why}\n  Decision: ${r.decision}`;
}

function formatOver(over, limit = 6) {
  return over
    .map((o) => {
      const shown = o.hits.slice(0, limit).map((h) => `    line ${h.line}: \`${h.text}\`   ${h.src.slice(0, 110)}`);
      if (o.hits.length > limit) shown.push(`    ... and ${o.hits.length - limit} more`);
      return `[${o.rule}] ${o.rel}: ${o.n} found, ${o.allowed} allowed\n${shown.join("\n")}\n${explain(o.rule)}`;
    })
    .join("\n\n");
}

const NO_WORKAROUND =
  "Do not work around this: no shell edits, no renaming to dodge the pattern, no moving the code to a file the rule skips. " +
  "Rewrite it as an F-rep construction. That another file still does this is no license: that code is slated for replacement " +
  "(docs/EXACT_FIELD.md). If the task truly needs this, stop and ask Derek.";

// ---------------------------------------------------------------- hook entry points

function readStdin() {
  try {
    return JSON.parse(readFileSync(0, "utf8") || "{}");
  } catch {
    return {};
  }
}

const emit = (obj) => process.stdout.write(JSON.stringify(obj));
const preToolUse = (decision, reason) =>
  emit({ hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: decision, permissionDecisionReason: reason } });

function applyEdits(old, edits) {
  let text = old;
  for (const e of edits) {
    const os = e.old_string ?? "";
    const ns = e.new_string ?? "";
    let t = text;
    if (!t.includes(os)) {
      t = t.replace(/\r\n/g, "\n");
      if (!t.includes(os.replace(/\r\n/g, "\n"))) return null;
    }
    const o = t === text ? os : os.replace(/\r\n/g, "\n");
    text = e.replace_all ? t.split(o).join(ns) : t.replace(o, () => ns);
  }
  return text;
}

function hookPreEdit(input) {
  const ti = input.tool_input || {};
  const rel = relPath(ti.file_path);
  if (!rel) return;
  if (GUARD_FILES.some((re) => re.test(rel))) {
    return preToolUse(
      "ask",
      `frep-guard protects its own rules and wiring (${rel}). Changing it changes what counts as a B-rep construction in the F-rep path, so Derek must approve this edit himself.`,
    );
  }
  if (!RULES.some((r) => r.applies(rel))) return;
  const abs = path.join(ROOT, rel);
  const old = existsSync(abs) ? readFileSync(abs, "utf8") : "";
  let next;
  if (input.tool_name === "Write") next = ti.content ?? "";
  else if (Array.isArray(ti.edits)) next = applyEdits(old, ti.edits);
  else next = applyEdits(old, [ti]);
  if (next === null) return; // the edit will fail on its own

  const before = scan(rel, old);
  const after = scan(rel, next);
  const nb = countBy(before);
  const na = countBy(after);
  const allow = loadBaseline().allow[rel] || {};
  const over = [];
  for (const [rule, n] of Object.entries(na)) {
    const limit = Math.max(nb[rule] || 0, allow[rule] || 0);
    if (n <= limit) continue;
    // Show the hits that were not there before.
    const seen = new Map();
    for (const h of before.filter((h) => h.rule === rule)) seen.set(h.text + h.src, (seen.get(h.text + h.src) || 0) + 1);
    const added = after.filter((h) => {
      if (h.rule !== rule) return false;
      const k = h.text + h.src;
      if (seen.get(k)) return seen.set(k, seen.get(k) - 1), false;
      return true;
    });
    over.push({ rel, rule, n, allowed: limit, hits: added.length ? added : after.filter((h) => h.rule === rule) });
  }
  if (over.length === 0) return;
  preToolUse(
    "deny",
    `frep-guard blocked this edit: it adds B-rep constructions where the F-rep is required.\n\n${formatOver(over)}\n\n${NO_WORKAROUND}`,
  );
}

function hookBash(input) {
  const cmd = String(input.tool_input?.command || "");
  // Flags are read with quoted text blanked, so a commit message that mentions one passes.
  const bare = cmd.replace(/"(?:\\.|[^"\\])*"|'[^']*'/g, '""');
  if (/\bgit\b/.test(bare) && (/--no-verify\b/.test(bare) || /\bgit\b[^|;&]*\bcommit\b[^|;&]*\s-n(?=\s|$)/.test(bare))) {
    return preToolUse("deny", "frep-guard: skipping git hooks is not allowed in this repository; the pre-commit hook is the last check that no B-rep construction reached the F-rep path. Fix what it reports instead.");
  }
  if (/core\.hooksPath/i.test(bare) && !/^\s*git\s+config\s+(--local\s+)?core\.hooksPath\s+\.githooks\s*$/.test(bare)) {
    return preToolUse("deny", "frep-guard: core.hooksPath must stay .githooks, where the F-rep pre-commit check lives.");
  }
  if (/disableAllHooks/.test(cmd)) {
    return preToolUse("deny", "frep-guard: turning hooks off would turn off the F-rep checks. Ask Derek if hooks are getting in the way.");
  }
  if (/check\.mjs[^|;&]*--accept\b/.test(cmd)) {
    return preToolUse("ask", "frep-guard: --accept raises the allowance for B-rep constructions in the F-rep path. Only Derek accepts that; approve only if he asked for this exact change.");
  }
  const touchesGuard = /frep-guard|\.githooks|\.claude[\\/]settings/.test(cmd);
  const writes =
    /(?<![\d&=-])>>?\s*(?!\/dev\/null|\$null|&)|\bsed\s+-i|\bperl\s+-i|\btee\b|\b(?:mv|cp|rm|del)\b|\b(?:Set|Add|Clear)-Content\b|\bOut-File\b|\b(?:Remove|Move|Copy|New|Rename)-Item\b|\bgit\s+(?:checkout|restore|rm|mv|stash)\b|\bnode\s+-e\b|\bpython\w*\s+-c\b|WriteAll|writeFileSync/;
  if (touchesGuard && writes.test(cmd) && !/check\.mjs\s+--tighten\b/.test(cmd)) {
    return preToolUse("ask", "frep-guard: this command looks like it changes the guard's own rules or wiring. Derek must approve that himself.");
  }
}

function hookStop(input) {
  const over = overages(treeFiles(), loadBaseline());
  if (over.length === 0) return;
  const report = formatOver(over, 4);
  if (input.stop_hook_active) {
    return emit({ systemMessage: `frep-guard: the tree still has B-rep constructions beyond the baseline. The pre-commit hook will refuse them.\n${report}` });
  }
  emit({
    decision: "block",
    reason: `frep-guard: the working tree now has B-rep constructions where the F-rep is required (edits made through the shell are checked here too).\n\n${report}\n\n${NO_WORKAROUND} Remove them before finishing, or tell Derek plainly that you added them and why.`,
  });
}

function hookSessionStart() {
  const lines = [
    "ImplicitCAM F-rep rules. These are Derek's decisions; do not re-litigate them, and do not let a plan quietly reverse them:",
    ...DECISIONS.map((d) => `- ${d}`),
    "tools/frep-guard enforces these. It blocks edits that add B-rep constructions to the F-rep path, re-checks the tree when you stop, and refuses commits. If a task seems to need a prohibited construction, stop and ask Derek. Never route around the hooks.",
  ];
  const over = overages(treeFiles(), loadBaseline());
  if (over.length) lines.push(`Currently over the baseline: ${over.map((o) => `${o.rel} [${o.rule}] ${o.n}/${o.allowed}`).join("; ")}.`);
  try {
    const hp = execFileSync("git", ["config", "core.hooksPath"], { cwd: ROOT, encoding: "utf8" }).trim();
    if (hp !== ".githooks") throw new Error();
  } catch {
    lines.push("The git pre-commit guard is not installed in this clone. Install it by running: git config core.hooksPath .githooks");
  }
  emit({ hookSpecificOutput: { hookEventName: "SessionStart", additionalContext: lines.join("\n") } });
}

// ---------------------------------------------------------------- command line

function git(args) {
  return execFileSync("git", args, { cwd: ROOT, encoding: "utf8", maxBuffer: 64 << 20 });
}

/** The file's content at HEAD, or null for a new file. */
function atHead(rel) {
  try {
    return git(["show", `HEAD:${rel}`]);
  } catch {
    return null;
  }
}

function staged() {
  const names = git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).split("\n").filter(Boolean);
  const baseline = names.includes(BASELINE_REL) ? loadBaseline(git(["show", `:${BASELINE_REL}`])) : loadBaseline();
  const files = names.filter((f) => RULES.some((r) => r.applies(f))).map((f) => [f, git(["show", `:${f}`])]);
  // The ratchet (Derek, 2026-10-25): code over a rule is slated for replacement, not
  // allowed. A commit may touch such a file only if it lowers every count the file is
  // over on, against HEAD, so replacement lands step by step and nothing new gets in.
  const over = overages(files, baseline).filter((o) => {
    const head = atHead(o.rel);
    const before = head === null ? 0 : countBy(scan(o.rel, head))[o.rule] || 0;
    return !(before > 0 && o.n < before);
  });
  if (over.length === 0) return 0;
  process.stderr.write(
    `frep-guard refused this commit: it keeps or adds B-rep constructions where the F-rep is required.\n\n${formatOver(over)}\n\n` +
      `A file over a rule may be committed only when the commit lowers that count (it is slated for replacement, see docs/EXACT_FIELD.md). ` +
      `Remove them, or lower the count and commit that step.\n`,
  );
  return 1;
}

function report() {
  const baseline = loadBaseline();
  const files = treeFiles();
  const over = overages(files, baseline);
  const counts = files.map(([rel, text]) => [rel, countBy(scan(rel, text))]).filter(([, c]) => Object.keys(c).length);
  for (const [rel, c] of counts) {
    const allow = baseline.allow[rel] || {};
    for (const [rule, n] of Object.entries(c)) console.log(`${n > (allow[rule] || 0) ? "OVER " : "ok   "} ${rel} [${rule}] ${n} found, ${allow[rule] || 0} allowed`);
  }
  for (const [rel, allow] of Object.entries(baseline.allow)) {
    const c = Object.fromEntries(counts)[rel] || {};
    for (const [rule, n] of Object.entries(allow)) {
      if ((c[rule] || 0) < n) console.log(`less  ${rel} [${rule}] ${c[rule] || 0} found, ${n} allowed (run --tighten to lock in)`);
    }
  }
  if (over.length) {
    console.log(`\n${formatOver(over)}`);
    return 1;
  }
  console.log(counts.length ? "\nfrep-guard: nothing beyond the baseline." : "frep-guard: clean.");
  return 0;
}

function tighten() {
  const b = loadBaseline();
  const counts = Object.fromEntries(treeFiles().map(([rel, text]) => [rel, countBy(scan(rel, text))]));
  let changed = 0;
  for (const [rel, allow] of Object.entries(b.allow)) {
    for (const [rule, n] of Object.entries(allow)) {
      const now = counts[rel]?.[rule] || 0;
      if (now < n) (now ? (allow[rule] = now) : delete allow[rule], changed++);
    }
    if (Object.keys(allow).length === 0) delete b.allow[rel];
  }
  if (changed) {
    b.log.push({ date: new Date().toLocaleDateString("sv"), change: "tighten", reason: "Lowered allowances to the remaining uses." });
    saveBaseline(b);
  }
  console.log(changed ? `frep-guard: lowered ${changed} allowance(s).` : "frep-guard: nothing to tighten.");
  return 0;
}

function accept(reason) {
  if (!reason) {
    console.error('frep-guard: --accept needs --reason "why Derek allows this".');
    return 2;
  }
  const b = loadBaseline();
  const allow = {};
  for (const [rel, text] of treeFiles()) {
    const c = countBy(scan(rel, text));
    if (Object.keys(c).length) allow[rel] = c;
  }
  b.allow = allow;
  b.log.push({ date: new Date().toLocaleDateString("sv"), change: "accept", reason });
  saveBaseline(b);
  console.log(`frep-guard: baseline set to the current tree (${Object.keys(allow).length} file(s)).`);
  return 0;
}

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const value = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : undefined);
const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) try {
  if (flag("--hook")) {
    const name = value("--hook");
    const input = name === "session-start" ? {} : readStdin();
    ({ "pre-edit": hookPreEdit, bash: hookBash, stop: hookStop, "session-start": hookSessionStart })[name]?.(input);
    process.exit(0);
  }
  if (flag("--staged")) process.exit(staged());
  if (flag("--tighten")) process.exit(tighten());
  if (flag("--accept")) process.exit(accept(value("--reason")));
  process.exit(report());
} catch (e) {
  process.stderr.write(`frep-guard crashed: ${e?.stack || e}\n`);
  // Hooks fail open (the pre-commit check is the backstop); the commit check fails closed.
  process.exit(flag("--hook") ? 0 : 1);
}
