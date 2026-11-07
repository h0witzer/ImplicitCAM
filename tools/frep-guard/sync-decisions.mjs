#!/usr/bin/env node
// sync-decisions.mjs - copy the design decisions from CONTRIBUTING.md, verbatim, into the
// guard's session-start text (rules.mjs DECISIONS) and the four edit-check prompts in
// .claude/settings.json. CONTRIBUTING.md is the one record of the decisions (Derek,
// 2026-11-07); run this after changing it, and commit all three files together.
//
//   node tools/frep-guard/sync-decisions.mjs            rewrite rules.mjs and settings.json
//   node tools/frep-guard/sync-decisions.mjs --check    exit 1 if either is out of step (CI)

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const CONTRIBUTING = path.join(ROOT, "CONTRIBUTING.md");
const RULES = path.join(ROOT, "tools", "frep-guard", "rules.mjs");
const SETTINGS = path.join(ROOT, ".claude", "settings.json");

/** The numbered decisions under "## The design decisions", as plain text, one string each. */
export function decisions(md = readFileSync(CONTRIBUTING, "utf8")) {
  const start = md.indexOf("\n## The design decisions");
  const end = md.indexOf("\n### The guard", start);
  if (start < 0 || end < 0) throw new Error("CONTRIBUTING.md: the design decisions section was not found");
  const lines = md.slice(start, end).split(/\r?\n/);
  const out = [];
  for (const line of lines) {
    const m = /^(\d+)\.\s+(.*)$/.exec(line);
    if (m) {
      if (Number(m[1]) !== out.length + 1) throw new Error(`CONTRIBUTING.md: decision ${m[1]} out of order`);
      out.push(m[2]);
    } else if (out.length && /^\s{3,}\S/.test(line)) {
      // A continuation or a sub-bullet of the current decision.
      const t = line.trim();
      out[out.length - 1] += /^- /.test(t) ? ` ${t}` : ` ${t}`;
    } else if (out.length && line.trim() === "") {
      // A blank line inside the list separates decisions; nothing to add.
    } else if (out.length && !/^\s/.test(line) && line.trim() !== "") {
      // Prose after the list ends it.
      break;
    }
  }
  return out.map(plain);
}

/** Markdown to plain text: bold off, links to their text, whitespace collapsed. */
function plain(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function numbered(list) {
  return list.map((d, i) => `${i + 1}. ${d}`).join("\n");
}

/** The four edit-check prompts: one preamble and one epilogue each around the same list. */
function prompts(list) {
  const body = numbered(list);
  const code = "You are the design check for ImplicitCAM, an F-rep (implicit field) CAM kernel. Derek, the owner, has made these decisions (CONTRIBUTING.md, copied here verbatim).";
  const docs = "You are the design check for ImplicitCAM's design documents. Derek, the owner, has made these decisions (CONTRIBUTING.md, copied here verbatim). A plan written into the docs must not quietly reverse them (a 2026-09-30 plan once did, and code was built on it):";
  const retire = "Code on the retirement list (the prismatic part model: frep.rs, prismatic.rs, part.rs, the references in query.rs, icam-query's slab clearance, icam-stock's SweptStock and ImplicitStock, icam-gpu's prismatic kernel, icam-sim's column views, and the plans built on them) may be edited to lower its count or to delete it; judge only constructions the edit newly introduces.";
  const codeEnd = `Code in crates/icam-check, or inside a #[cfg(test)] module, that compares a kernel against its reference is allowed. ${retire} Return {"ok": true} unless the added text clearly introduces one of the forbidden constructions, or a new query, plan or model written for a held axis, a vertical tool, a slab or a class of parts (decision 11). If it does, return {"ok": false, "reason": "<the decision number, the exact added code that breaks it, and the field construction to use instead>. Do not route around this check: change the approach, or stop and ask Derek."}`;
  const docsEnd = `Allowed: recording a decision attributed to Derek; recording a reversed plan as reversed; a proposal clearly marked as not adopted or for Derek's review; analysis, measurements and literature. Block only text that states a forbidden construction, or a held-axis query, plan or part model, as the plan, the next step or a settled design, or that weakens or reverses a decision, without attributing that change to Derek. Return {"ok": true} unless that clearly happens. If it does, return {"ok": false, "reason": "<the decision number and the sentence that reverses it>. Mark it as a proposal for Derek's review, or ask Derek first."}`;
  return {
    "Edit(crates/**)": `${code} An edit must not contradict them:\n${body}\n\nThe proposed tool call:\n$ARGUMENTS\n\nJudge only what the edit ADDS (new_string, edits, or content); ignore what it removes. ${codeEnd}`,
    "Write(crates/**)": `${code} A new or rewritten file must not contradict them:\n${body}\n\nThe proposed tool call:\n$ARGUMENTS\n\nIf the file already existed, judge only constructions that look newly introduced. ${codeEnd}`,
    "Edit(docs/**)": `${docs}\n${body}\n\nThe proposed edit:\n$ARGUMENTS\n\nJudge only the added text (new_string). ${docsEnd}`,
    "Write(docs/**)": `${docs}\n${body}\n\nThe proposed file:\n$ARGUMENTS\n\nIf the file already existed, judge only text that looks newly introduced. ${docsEnd}`,
  };
}

function rulesText(list) {
  const src = readFileSync(RULES, "utf8");
  const a = src.indexOf("export const DECISIONS = [");
  const b = src.indexOf("\n];", a);
  if (a < 0 || b < 0) throw new Error("rules.mjs: DECISIONS not found");
  const body = list.map((d) => `  ${JSON.stringify(d)},`).join("\n");
  return `${src.slice(0, a)}export const DECISIONS = [\n${body}${src.slice(b)}`;
}

function settingsText(list) {
  const src = readFileSync(SETTINGS, "utf8");
  const json = JSON.parse(src);
  const want = prompts(list);
  let seen = 0;
  for (const group of json.hooks.PreToolUse ?? []) {
    for (const h of group.hooks ?? []) {
      if (h.type === "prompt" && want[h.if]) {
        h.prompt = want[h.if];
        seen++;
      }
    }
  }
  if (seen !== 4) throw new Error(`settings.json: expected 4 prompt hooks, found ${seen}`);
  return JSON.stringify(json, null, 2) + "\n";
}

const list = decisions();
const next = { [RULES]: rulesText(list), [SETTINGS]: settingsText(list) };
const check = process.argv.includes("--check");
let stale = 0;
for (const [file, text] of Object.entries(next)) {
  const same = readFileSync(file, "utf8").replace(/\r\n/g, "\n") === text.replace(/\r\n/g, "\n");
  if (same) continue;
  stale++;
  if (check) console.log(`out of step with CONTRIBUTING.md: ${path.relative(ROOT, file)}`);
  else {
    writeFileSync(file, text);
    console.log(`wrote ${path.relative(ROOT, file)} (${list.length} decisions)`);
  }
}
if (check && stale) process.exit(1);
if (!stale) console.log(`in step: ${list.length} decisions`);
