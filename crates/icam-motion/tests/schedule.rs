//! The entry feed on a turn: an entry circling on radius rho cuts its bore's wall with
//! the edge on radius rho + r, so its feed is the entry feed x rho / (rho + r); straight
//! entries keep the entry feed.

use icam_motion::analyze::{self, LoadParams};
use icam_post::{PathPt, Tag};

fn params(r: f64) -> LoadParams<'static> {
    LoadParams { r, z_cut: 0.0, v_prog: 34.0, phi_ref: 0.6, feed_cap: 2.0, accel: 2500.0, v_entry: 20.0, v_rapid: 200.0, cut: None, ap_at: None }
}

#[test]
fn entry_feeds_drop_on_turns_by_the_edge_radius() {
    let r = 6.35;
    for rho in [0.64, 2.0, 5.7] {
        // Three laps of a ramped circle, points every 0.5 mm as the entry writes them.
        let n = 3 * (2.0 * std::f64::consts::PI * rho / 0.5).ceil() as usize;
        let per = n / 3;
        let path: Vec<PathPt> = (0..=n).map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / per as f64;
            PathPt::new(rho * a.cos(), rho * a.sin(), 3.0 - 0.01 * k as f64, Tag::Entry)
        }).collect();
        let mut s = analyze::from_path(&path, 0.25);
        analyze::schedule(&mut s, &params(r));
        let want = 20.0 * rho / (rho + r);
        let mid = &s[s.len() / 4..3 * s.len() / 4];
        let worst = mid.iter().map(|p| (p.feed / want - 1.0).abs()).fold(0.0, f64::max);
        println!("rho {rho}: feed {:.3} mm/s wanted, worst {:.1} % off", want, 100.0 * worst);
        assert!(worst < 0.05, "rho {rho}: {worst}");
    }
    // A straight entry keeps the entry feed.
    let path: Vec<PathPt> = (0..=20).map(|k| PathPt::new(k as f64, 0.0, 3.0 - 0.1 * k as f64, Tag::Entry)).collect();
    let mut s = analyze::from_path(&path, 0.25);
    analyze::schedule(&mut s, &params(r));
    assert!(s[1..s.len() - 1].iter().all(|p| (p.feed - 20.0).abs() < 1e-9));
}
