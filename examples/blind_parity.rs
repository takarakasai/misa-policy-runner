//! 盲目段差契約のティック単位の照合台 — ロボットもプラントも使わず、固定の
//! センサ列を入れて、実行時が組んだ観測と行動をそのまま書き出す。
//!
//! ```text
//! cargo run --release --example blind_parity -- \
//!     --model go2_blind_stairs20_policy.onnx --frames frames.txt --out rust.txt
//! ```
//!
//! `frames.txt`: 1 ティック 1 行、空白区切りで 40 個の f64 —
//! `cmd(3) quat_wxyz(4) gyro(3) accel(3) q_isaac(12) dq_isaac(12) vel_body(3)`。
//! 関節は **Isaac 順**で与えてここで Go2 の並びへ直すので、ファイルはエンジンに
//! 依らない（Go2 モータ順の入れ替えは `obs` 側に単体テストがある）。加速度は
//! この契約では使わないが、他の照合台と列を揃えるために読む。
//!
//! `rust.txt`: 1 ティック 1 行 — `obs(48·H) q_des_isaac(12)`。Python 側
//! （`go2-runner/scripts/blind_parity_reference.py`）が同じ並びを onnxruntime と
//! 参照実装から書き、素の最大絶対差で比べる。**観測も比べる**のが肝で、
//! 観測の組み立てを間違えた実行時でも行動は有限な値を返してしまう。

use misa_policy_runner::blind::{BlindController, N_OBS_BLIND};
use misa_policy_runner::go2::ISAAC_TO_GO2;
use misa_policy_runner::{ObsInput, OnnxPolicy};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let model = arg("--model").expect("--model policy.onnx");
    let frames_path = arg("--frames").expect("--frames frames.txt");
    let out_path = arg("--out").expect("--out rust.txt");

    // 履歴の段数はグラフの入力幅から決める（48·H）。
    let mut policy = None;
    for h in 1usize..=10 {
        if let Ok(p) = OnnxPolicy::load(&model, N_OBS_BLIND * h) {
            policy = Some(p);
            break;
        }
    }
    let policy = policy.expect("48 の倍数を入力に取るグラフが読めない");
    let mut ctl = BlindController::new(policy).expect("controller");
    ctl.reset();
    eprintln!("[blind_parity] 履歴 {} フレーム", ctl.history_len());

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
        assert_eq!(v.len(), 40, "line {k}: want 40 values, got {}", v.len());
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
        let vel_body = [v[37], v[38], v[39]];

        let t0 = std::time::Instant::now();
        let (obs, tick) = ctl
            .tick_with_obs(&inp, cmd, vel_body)
            .unwrap_or_else(|e| panic!("tick {k}: {e}"));
        latency_us.push(t0.elapsed().as_micros());
        if !tick.anomalies.is_empty() {
            eprintln!("[blind_parity] tick {k}: {:?}", tick.anomalies);
        }
        let mut row: Vec<String> = obs.iter().map(|x| format!("{x:.17e}")).collect();
        row.extend(tick.q_des_isaac.iter().map(|x| format!("{x:.17e}")));
        out.push_str(&row.join(" "));
        out.push('\n');
    }
    std::fs::write(&out_path, out).expect("write out");
    latency_us.sort_unstable();
    if !latency_us.is_empty() {
        let n = latency_us.len();
        eprintln!(
            "[blind_parity] {n} ティック、1 ティックの所要 中央値 {} µs / p95 {} µs / 最大 {} µs",
            latency_us[n / 2],
            latency_us[(n * 95) / 100],
            latency_us[n - 1]
        );
    }
}
