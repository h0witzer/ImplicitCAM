//! Reads every STEP file in a folder through the importer and the exact signed distance
//! (`icam_core::solid3`), and checks each one: the file reads, every solid imports, and on
//! point pairs (half over the padded box, half near edges) the distance is 1-Lipschitz (to
//! twice the file's tolerance) and no point outside every solid's box reads inside. A time
//! budget per file bounds the pairs. One line per file, then the tally.
//!
//!   cargo run --release -p icam-check --bin step_corpus -- <dir> [pairs] [seconds]

use std::path::PathBuf;
use std::time::Instant;

use icam_core::curve3::V3;
use icam_core::solid3::Solid3;
use icam_core::step;
use rayon::prelude::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).expect("usage: step_corpus <dir> [pairs] [seconds]"));
    let pairs: usize = args.get(2).map_or(60, |s| s.parse().expect("pairs"));
    let budget: f64 = args.get(3).map_or(30.0, |s| s.parse().expect("seconds"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| matches!(x.to_string_lossy().to_lowercase().as_str(), "step" | "stp")))
        .collect();
    files.sort();
    let (mut pass, mut fail, mut empty) = (0, 0, 0);
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let t0 = Instant::now();
        let brep = match step::read(f) {
            Ok(b) => b,
            Err(e) => {
                fail += 1;
                println!("FAIL   {name:28} read: {e}");
                continue;
            }
        };
        if brep.solids.is_empty() {
            empty += 1;
            println!("EMPTY  {name:28} no solid in the file");
            continue;
        }
        let tol = brep.tolerance_mm.max(1e-6);
        let (mut solids, mut refused) = (Vec::new(), Vec::new());
        for s in &brep.solids {
            match Solid3::from_step(s, tol) {
                Ok((sol, _)) => solids.push(sol),
                Err(e) => refused.push(format!("#{} {e}", s.id)),
            }
        }
        let import_ms = t0.elapsed().as_secs_f64() * 1e3;
        let faces: usize = solids.iter().map(|s| s.faces.len()).sum();
        if !refused.is_empty() {
            fail += 1;
            println!("FAIL   {name:28} {} of {} solids refused, first: {}", refused.len(), brep.solids.len(), refused[0]);
            continue;
        }
        let (mut lo, mut hi) = (solids[0].lo, solids[0].hi);
        for s in &solids {
            lo = V3::new(lo.x.min(s.lo.x), lo.y.min(s.lo.y), lo.z.min(s.lo.z));
            hi = V3::new(hi.x.max(s.hi.x), hi.y.max(s.hi.y), hi.z.max(s.hi.z));
        }
        let diag = (hi - lo).len();
        let sdf = |p: V3| solids.iter().map(|s| s.sdf(p)).min_by(|a, b| a.d.total_cmp(&b.d)).unwrap();
        let outside_all = |p: V3| solids.iter().all(|s| p.x < s.lo.x || p.y < s.lo.y || p.z < s.lo.z || p.x > s.hi.x || p.y > s.hi.y || p.z > s.hi.z);
        let mut rng = Rng(0x5EED_57E9_C0B5);
        let pad = 0.05 * diag;
        let mut pts = Vec::with_capacity(pairs);
        for k in 0..pairs {
            let s = &solids[(rng.next() * solids.len() as f64) as usize % solids.len()];
            let p = if k % 2 == 0 || s.edges.is_empty() {
                V3::new(rng.range(lo.x - pad, hi.x + pad), rng.range(lo.y - pad, hi.y + pad), rng.range(lo.z - pad, hi.z + pad))
            } else {
                let e = &s.edges[(rng.next() * s.edges.len() as f64) as usize % s.edges.len()];
                let dir = V3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).unit();
                e.curve.at(rng.next()) + dir * (rng.next().powi(2) * 0.05 * diag)
            };
            let dir = V3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).unit();
            pts.push((p, dir, rng.next()));
        }
        // Batches across the threads until the pairs or the budget run out.
        let t1 = Instant::now();
        let mut res = Vec::new();
        for chunk in pts.chunks(rayon::current_num_threads().max(1)) {
            if t1.elapsed().as_secs_f64() > budget {
                break;
            }
            res.extend(chunk.par_iter().map(|&(p, dir, u)| {
                let a = sdf(p);
                let q = p + dir * (u * (2.0 * a.d.abs() + 1e-3 * diag));
                (p, a, q, sdf(q))
            }).collect::<Vec<_>>());
        }
        let secs = t1.elapsed().as_secs_f64();
        let (mut lip, mut worst, mut neg_out, mut unsure) = (0, 0.0f64, 0, 0);
        for (p, a, q, b) in &res {
            let excess = (a.d - b.d).abs() - (*p - *q).len();
            if excess > 2.0 * tol {
                lip += 1;
                worst = worst.max(excess);
            }
            neg_out += (outside_all(*p) && a.d < 0.0) as usize;
            unsure += (a.unsure || b.unsure) as usize;
        }
        let ok = lip == 0 && neg_out == 0;
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        println!(
            "{}  {name:28} {} solid(s), {faces:5} faces, import {import_ms:7.0} ms, {}/{pairs} pairs at {:7.1}/s; Lipschitz {lip} (worst {worst:.2e}), inside out of the box {neg_out}, unsure {unsure}",
            if ok { "PASS " } else { "FAIL " },
            solids.len(),
            res.len(),
            2.0 * res.len() as f64 / secs.max(1e-9)
        );
    }
    println!("{pass} pass, {fail} fail, {empty} with no solid, of {}", files.len());
}
