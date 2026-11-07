# Test parts

STEP files used for regression runs of the importer, the exact evaluator and the span
implicits. The files themselves are not committed: none of them carries a license that lets
this repository redistribute it (CONTRIBUTING.md, "Test data"). What is committed is
`manifest.json`: for every file its source, size and SHA-256, and the numbers it gave on
2026-10-25, so a later run can tell a changed file from a changed result.

| Folder | What | Files |
| --- | --- | --- |
| `steptools/` | STEP Tools' AP203 sample files: assemblies, faceted solids, CATIA version 4 short names, swept surfaces, every B-spline and conic form, B-spline degrees up to (5,11) | 55 |
| `the-part/` | THE PART by @emm0sh, the 5/7-axis mill-turn benchmark | 1 |

## Getting them

Each `steptools` entry has the URL it was downloaded from. THE PART has no public URL in the
manifest; ask Derek for the file and save it as `the-part/the-part-emm0sh.step`.

```sh
# from test-parts/
node -e "for (const e of require('./manifest.json').suites[0].files) console.log(e.url)" | while read u; do curl -sSfo "steptools/${u##*/}" "$u"; done
```

Check what is on disk against the manifest:

```sh
node -e "const c=require('crypto'),fs=require('fs');let bad=0;for(const s of require('./manifest.json').suites)for(const e of s.files){const p=s.dir+'/'+e.file;if(!fs.existsSync(p)){console.log('missing',p);bad++;continue}const h=c.createHash('sha256').update(fs.readFileSync(p)).digest('hex');if(h!==e.sha256){console.log('changed',p);bad++}}console.log(bad?bad+' problem(s)':'all match');process.exit(bad?1:0)"
```

## Runs that use them

```sh
cargo run --release -p icam-check --bin step_corpus -- test-parts/steptools        # importer and the retired path-walking distance (comparison)
cargo run --release -p icam-check -- exact test-parts/steptools                     # the exact evaluator's checks, no B-rep reference
cargo run --release -p icam-check -- spans test-parts/steptools [n]                 # span implicits, n spans per file
cargo run --release -p icam-check -- spans test-parts/the-part/the-part-emm0sh.step # THE PART, every span
cargo run --release -p icam-check -- surfcheck test-parts/steptools                 # fields against the exact surfaces, no B-rep distance
```

Baselines recorded in the manifest (2026-10-25): the importer takes 49 of 55 `steptools`
files whole (`chair` refuses one of eight solids; `jack_in_the_box`, `moon_buggy` and
`moon_buggy_asm` break the Lipschitz check; `gear` and `snet` hold no solid); 10 files carry
B-spline faces, 12,653 spans in all.
