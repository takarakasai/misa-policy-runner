//! Offline model validation — no robot, no sim.
//!
//! ```text
//! cargo run --release --example selftest -- path/to/policy.onnx
//! ```
//!
//! Loads the graph with tract (39-d Natural contract), runs the full
//! controller for 10 s of simulated ticks with a plausible synthetic
//! observation stream, and reports action ranges + inference latency
//! against the 50 Hz slot budget. This validates the exact bytes that later
//! run on the robot; only the sensors are fake.

use misa_policy_runner::go2::{GO2_TO_ISAAC, POLICY_HZ};
use misa_policy_runner::{NaturalController, ObsInput, OnnxPolicy, TrajectoryCfg};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: selftest <policy.onnx>");
    let policy = OnnxPolicy::load(&path, 39).expect("load");
    let mut ctl =
        NaturalController::new(policy, TrajectoryCfg::h30_standard()).expect("controller");
    eprintln!("selftest: loaded {path} (39 -> 36)");

    let default = ctl.default_pose_isaac();
    eprintln!("selftest: default pose (ik of the standing trajectory) = {default:?}");

    // synthetic sensors: standing still in the default pose
    let mut inp = ObsInput::default();
    for g in 0..12 {
        inp.joint_q_go2[g] = default[GO2_TO_ISAAC[g]];
    }

    let mut lat_us: Vec<u128> = Vec::new();
    let mut q_min = [f64::INFINITY; 12];
    let mut q_max = [f64::NEG_INFINITY; 12];
    let mut kp_min = f64::INFINITY;
    let mut kp_max = f64::NEG_INFINITY;
    let n = (10.0 * POLICY_HZ) as usize;
    for k in 0..n {
        let t0 = std::time::Instant::now();
        let tick = ctl
            .tick(&inp, [0.12, 0.0, 0.0])
            .unwrap_or_else(|e| panic!("tick {k}: {e}"));
        lat_us.push(t0.elapsed().as_micros());
        if !tick.anomalies.is_empty() {
            panic!("tick {k}: anomalies {:?}", tick.anomalies);
        }
        for i in 0..12 {
            q_min[i] = q_min[i].min(tick.q_des_isaac[i]);
            q_max[i] = q_max[i].max(tick.q_des_isaac[i]);
            kp_min = kp_min.min(tick.kp_isaac[i]);
            kp_max = kp_max.max(tick.kp_isaac[i]);
        }
        // feed the (filtered) command back as the measured pose — a crude
        // but plausible closed loop for a shape/latency test
        for g in 0..12 {
            inp.joint_q_go2[g] = tick.q_des_isaac[GO2_TO_ISAAC[g]];
        }
    }

    lat_us.sort_unstable();
    let mean = lat_us.iter().sum::<u128>() as f64 / lat_us.len() as f64;
    let p99 = lat_us[lat_us.len() * 99 / 100 - 1];
    let budget = (1e6 / POLICY_HZ) as u128;
    eprintln!(
        "selftest: {n} ticks; latency mean {mean:.0}us p99 {p99}us max {}us (slot {budget}us)",
        lat_us[lat_us.len() - 1]
    );
    if p99 > budget / 2 {
        eprintln!("selftest: WARNING - p99 over half the slot; expect jitter on the Go2 CPU");
    }
    eprintln!("selftest: q_des range per joint:");
    for i in 0..12 {
        eprintln!("  isaac[{i:2}]  {:+.3} .. {:+.3}", q_min[i], q_max[i]);
    }
    eprintln!("selftest: Kp range {kp_min:.1} .. {kp_max:.1}");
    eprintln!("selftest: OK");
}
