//! Tick-by-tick parity harness for the recurrent (GRU) contract — no robot,
//! no plant model: a fixed sensor stream in, the runtime's own observation,
//! action and recurrent state out.
//!
//! ```text
//! cargo run --release --example gru_parity -- \
//!     --model model_49.onnx --estimator estimator_42.onnx \
//!     --frames frames.txt --out rust.txt
//! ```
//!
//! `frames.txt`: one tick per line, whitespace separated, 37 f64 —
//! `cmd(3) quat_wxyz(4) gyro(3) accel(3) q_isaac(12) dq_isaac(12)`. Joint
//! values are given in **Isaac** order and reordered here, so the file is
//! engine-agnostic (the Go2 motor reorder has its own unit test in `obs`).
//! With `--host-velocity` the estimator is skipped and three more columns
//! (the body-frame velocity) are read from each line.
//!
//! 実機（Go2 の SBC = aarch64）で走らせるときは
//! `scripts/build_aarch64.sh --example gru_parity` — clang をクロス
//! アセンブラに、musl の self-contained crt を使うので **sudo もクロス gcc も
//! 要らず、出来上がりは静的リンク**（SBC 側に何も入れなくてよい）。
//!
//! `rust.txt`: one tick per line —
//! `obs(76) action_raw(36) hidden_out(128)`. The Python side
//! (`go2-runner/scripts/gru_parity_reference.py`) writes the same layout
//! from onnxruntime plus the reference implementation's rules, and the
//! comparison is a plain max-abs diff. The observation is compared too, not
//! just the action: a runtime that assembles the observation wrongly still
//! produces perfectly finite actions.

use misa_policy_runner::go2::ISAAC_TO_GO2;
use misa_policy_runner::gru::{HistoryVelocityEstimator, VelocitySource};
use misa_policy_runner::{ObsInput, PureGruController, RecurrentOnnxPolicy};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let model = arg("--model").expect("--model policy.onnx");
    let estimator = arg("--estimator");
    let host_velocity = std::env::args().any(|a| a == "--host-velocity");
    let frames_path = arg("--frames").expect("--frames frames.txt");
    let out_path = arg("--out").expect("--out rust.txt");

    let policy = RecurrentOnnxPolicy::load(&model, 76, 128, 36).expect("load recurrent policy");
    let velocity = if host_velocity {
        VelocitySource::Host
    } else {
        let p = estimator.expect("--estimator estimator.onnx (or --host-velocity)");
        VelocitySource::Estimator(HistoryVelocityEstimator::load(&p).expect("load estimator"))
    };
    let mut ctl = PureGruController::new(policy, velocity).expect("controller");
    ctl.reset();

    let expected = if host_velocity { 40 } else { 37 };
    let text = std::fs::read_to_string(&frames_path).expect("read frames");
    let mut out = String::new();
    let mut latency_us: Vec<u128> = Vec::new();
    for (k, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|t| t.parse().unwrap_or_else(|e| panic!("line {k}: {e}")))
            .collect();
        assert_eq!(
            v.len(),
            expected,
            "line {k}: want {expected} values, got {}",
            v.len()
        );
        let cmd = [v[0], v[1], v[2]];
        let mut inp = ObsInput {
            quat_wxyz: [v[3], v[4], v[5], v[6]],
            gyro_rad_s: [v[7], v[8], v[9]],
            accel_m_s2: [v[10], v[11], v[12]],
            joint_q_go2: [0.0; 12],
            joint_dq_go2: [0.0; 12],
        };
        for i in 0..12 {
            inp.joint_q_go2[ISAAC_TO_GO2[i]] = v[13 + i];
            inp.joint_dq_go2[ISAAC_TO_GO2[i]] = v[25 + i];
        }
        let vel_host = if host_velocity {
            [v[37], v[38], v[39]]
        } else {
            [0.0; 3]
        };

        let t0 = std::time::Instant::now();
        let tick = ctl
            .tick(&inp, cmd, vel_host)
            .unwrap_or_else(|e| panic!("tick {k}: {e}"));
        latency_us.push(t0.elapsed().as_micros());
        if !tick.anomalies.is_empty() {
            eprintln!(
                "gru_parity: tick {k}: observation anomalies {:?}",
                tick.anomalies
            );
        }
        for x in ctl.last_observation() {
            out.push_str(&format!("{:.17e} ", *x as f64));
        }
        for a in ctl.last_action_raw().iter() {
            out.push_str(&format!("{a:.17e} "));
        }
        for h in ctl.hidden_state() {
            out.push_str(&format!("{:.17e} ", *h as f64));
        }
        out.push('\n');
    }
    std::fs::write(&out_path, out).expect("write output");
    latency_us.sort_unstable();
    let n = latency_us.len();
    eprintln!(
        "gru_parity: {n} ticks -> {out_path}; inference {} µs median, {} µs p99 \
         (50 Hz slot = 20000 µs)",
        latency_us[n / 2],
        latency_us[(n * 99 / 100).min(n - 1)],
    );
}
