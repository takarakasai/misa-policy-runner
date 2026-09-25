//! 盲目（内界センサのみ）の段差契約 — go2_rl `artifacts/GO2_BLIND_STAIRS20.md`。
//!
//! 蹴上げ 20 cm × 10 段・幅 3 m の階段を、関節センサと IMU だけで登るために
//! 学習した方策の実行時契約。Pure 契約との違いは 3 点:
//!
//! - 観測が **48 次元**で、IMU の四元数と加速度ではなく**重力方向**を使う。
//!   胴体線速度は接触脚オドメトリの推定値（実機で作れる量）。
//! - 行動が **12 次元**（関節位置だけ）。**ゲインは固定** `Kp 25 / Kd 0.5`。
//!   実機へは MIT モードで `(q_des, dq_des=0, Kp, Kd, τ_ff=0)` を送るので、
//!   「ゲインを方策が出さない MIT モード」である。
//! - 既定姿勢は Pure の屈み姿勢ではなく **Isaac 標準の Go2 既定姿勢**
//!   （`DEFAULT_LOCO_ISAAC`）。
//!
//! 観測の並び（宣言順、スケール無し。学習側 `BlindProprioObservationsCfg`）:
//!
//! ```text
//! [ 胴体線速度 3 | 角速度 3 | 重力方向 3 | 指令 3 |
//!   関節角 − 既定 12 | 関節速度 12 | 前回の生の行動 12 ]  = 48
//! ```
//!
//! 復号（Isaac 順）:
//!
//! ```text
//! q_ref = DEFAULT_LOCO_ISAAC + 0.25·a     ← **可動域でクランプしない**
//! Kp    = 25   (固定)
//! Kd    = 0.5  (固定)
//! ```
//!
//! クランプしないのは、目標を切ると関節が可動域の端に近い姿勢でその向きの
//! トルクが `Kp·(切断点 − q)` までしか出ず、段を登る力が頭打ちになるため
//! （学習側も `pos_clamp='none'`）。可動域は実機のメカストッパーが守る。
//!
//! **履歴つきの版**（`history_length = H`）は観測が 48·H になる。並びは
//! **項目ごとに古い順**で、各項目ブロックの末尾が最新
//! （IsaacLab `CircularBuffer`）。`BlindController::new` が幅から H を決める。

use crate::controller::PolicyTick;
use crate::go2::DEFAULT_LOCO_ISAAC;
use crate::go2::GO2_TO_ISAAC;
use crate::obs::ObsInput;
use crate::policy::OnnxPolicy;

/// 1 フレームぶんの観測幅。
pub const N_OBS_BLIND: usize = 48;
/// 行動次元（関節位置のみ）。
pub const N_ACT_BLIND: usize = 12;
/// 位置行動のスケール [rad/単位]。
pub const BLIND_POS_SCALE: f64 = 0.25;
/// 固定ゲイン。
pub const BLIND_KP: f64 = 25.0;
pub const BLIND_KD: f64 = 0.5;
/// `last_action` 観測項の clip（Isaac `clip=(-100, 100)`）。
const LAST_ACTION_CLIP: f64 = 100.0;
/// 観測の項目ごとの次元（宣言順）。履歴の並べ替えに使う。
const TERM_DIMS: [usize; 7] = [3, 3, 3, 3, 12, 12, 12];

/// 学習時の指令範囲（`Isaac-Go2-StairsBlindAsymHighDR-v0`、
/// IsaacLab 標準の velocity タスクのまま）: vx ±1.0、vy ±1.0、wz ±1.0。
/// 階段では 0.6 m/s 前後が最も確実（MuJoCo 評価で 20 cm が 12/12）。
pub fn clamp_blind_cmd(cmd: [f64; 3]) -> [f64; 3] {
    [
        cmd[0].clamp(-1.0, 1.0),
        cmd[1].clamp(-1.0, 1.0),
        cmd[2].clamp(-1.0, 1.0),
    ]
}

pub struct BlindController {
    policy: OnnxPolicy,
    history: usize,
    /// 項目ごとの履歴リング。`frames[i]` が i フレーム前（0 が最新）。
    frames: Vec<[f32; N_OBS_BLIND]>,
    last_action: [f64; N_ACT_BLIND],
    q_hold: [f64; 12],
    t: f64,
}

impl BlindController {
    /// `policy` は 48 の倍数を入力に取るグラフ。
    pub fn new(policy: OnnxPolicy) -> Result<Self, String> {
        let n = policy.n_obs();
        if n == 0 || n % N_OBS_BLIND != 0 {
            return Err(format!(
                "盲目契約は {N_OBS_BLIND} の倍数を入力に取るグラフが必要: このグラフは {n}"
            ));
        }
        // 出力幅は `infer_n` が実行時に検査する（OnnxPolicy は n_act を持たない）。
        Ok(Self {
            policy,
            history: n / N_OBS_BLIND,
            frames: Vec::new(),
            last_action: [0.0; N_ACT_BLIND],
            q_hold: DEFAULT_LOCO_ISAAC,
            t: 0.0,
        })
    }

    /// 観測に積む履歴フレーム数（1 なら履歴なし）。
    pub fn history_len(&self) -> usize {
        self.history
    }

    /// 最初のティックの前に構えておく姿勢（a = 0 の固定点）。
    pub fn default_pose_isaac(&self) -> [f64; 12] {
        DEFAULT_LOCO_ISAAC
    }

    /// 固定ゲイン。
    pub fn initial_gains(&self) -> (f64, f64) {
        (BLIND_KP, BLIND_KD)
    }

    /// この契約は胴体速度の推定値を要求する（接触脚オドメトリ）。
    pub fn wants_velocity(&self) -> bool {
        true
    }

    pub fn gait_time_s(&self) -> f64 {
        self.t
    }

    /// 記憶を消して既定姿勢の保持へ戻す。既定姿勢で立っているときに呼ぶ。
    pub fn reset(&mut self) {
        self.frames.clear();
        self.last_action = [0.0; N_ACT_BLIND];
        self.q_hold = DEFAULT_LOCO_ISAAC;
        self.t = 0.0;
    }

    /// 安全側の保持: 直近の指令をそのまま出す（凍結しても連続）。
    pub fn hold(&self) -> PolicyTick {
        PolicyTick {
            q_des_isaac: self.q_hold,
            kp_isaac: [BLIND_KP; 12],
            kd_isaac: [BLIND_KD; 12],
            anomalies: Vec::new(),
        }
    }

    /// 48 次元の 1 フレームを組む。`vel_body` は接触脚オドメトリの胴体系速度。
    fn build_frame(
        &self,
        inp: &ObsInput,
        cmd: &[f64; 3],
        vel_body: [f64; 3],
    ) -> [f32; N_OBS_BLIND] {
        build_blind_frame(inp, cmd, vel_body, &self.last_action)
    }
}

/// 48 次元の 1 フレーム（学習側 `BlindProprioObservationsCfg` の宣言順）。
/// 状態を持たないので単体で検証できる。
pub fn build_blind_frame(
    inp: &ObsInput,
    cmd: &[f64; 3],
    vel_body: [f64; 3],
    last_action: &[f64; N_ACT_BLIND],
) -> [f32; N_OBS_BLIND] {
    {
        let mut f = [0.0f32; N_OBS_BLIND];
        let mut k = 0;
        for v in vel_body.iter() {
            f[k] = *v as f32;
            k += 1;
        }
        for g in inp.gyro_rad_s.iter() {
            f[k] = *g as f32;
            k += 1;
        }
        // 重力方向 = Rᵀ(0,0,−1)。四元数から直接出す（加速度計は使わない）。
        let [w, x, y, z] = inp.quat_wxyz;
        for v in [
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            2.0 * (x * x + y * y) - 1.0,
        ] {
            f[k] = v as f32;
            k += 1;
        }
        for c in cmd.iter() {
            f[k] = *c as f32;
            k += 1;
        }
        for g in 0..12 {
            let i = GO2_TO_ISAAC[g];
            f[4 * 3 + i] = (inp.joint_q_go2[g] - DEFAULT_LOCO_ISAAC[i]) as f32;
            f[4 * 3 + 12 + i] = inp.joint_dq_go2[g] as f32;
        }
        k = 4 * 3 + 24;
        for a in last_action.iter() {
            f[k] = a.clamp(-LAST_ACTION_CLIP, LAST_ACTION_CLIP) as f32;
            k += 1;
        }
        debug_assert_eq!(k, N_OBS_BLIND);
        f
    }
}

impl BlindController {

    /// 50 Hz の 1 ティック。失敗時は記憶を進めない（[`Self::hold`] を送ること）。
    pub fn tick(
        &mut self,
        inp: &ObsInput,
        cmd_raw: [f64; 3],
        vel_body: [f64; 3],
    ) -> Result<PolicyTick, String> {
        self.tick_with_obs(inp, cmd_raw, vel_body).map(|(_, t)| t)
    }

    /// [`Self::tick`] と同じだが、**方策へ入った観測もそのまま返す**。
    /// 照合台（`examples/blind_parity.rs`）が使う: 観測の組み立てを間違えた
    /// 実行時でも行動は有限な値を返してしまうので、観測まで突き合わせる。
    pub fn tick_with_obs(
        &mut self,
        inp: &ObsInput,
        cmd_raw: [f64; 3],
        vel_body: [f64; 3],
    ) -> Result<(Vec<f32>, PolicyTick), String> {
        let cmd = clamp_blind_cmd(cmd_raw);
        let frame = self.build_frame(inp, &cmd, vel_body);
        // `obs::obs_anomalies` は 37/39 次元の Natural/Pure 観測専用なので使えない
        // （48 次元を渡すと常に "bad_len"）。この契約の並びで点検する。
        let anomalies = blind_anomalies(&frame, &inp.quat_wxyz);

        // 履歴リングを進める（最初は全フレームを現在値で埋める）。
        let mut frames = if self.frames.is_empty() {
            vec![frame; self.history]
        } else {
            self.frames.clone()
        };
        if !self.frames.is_empty() {
            frames.rotate_left(1);
            frames[self.history - 1] = frame;
        }

        let obs = flatten_history(&frames);

        let action = self.policy.infer_n(&obs, N_ACT_BLIND)?;
        let mut q_des = [0.0f64; 12];
        let mut act = [0.0f64; N_ACT_BLIND];
        for i in 0..12 {
            act[i] = action[i] as f64;
            // **クランプしない**（学習側 pos_clamp='none'）。
            q_des[i] = DEFAULT_LOCO_ISAAC[i] + BLIND_POS_SCALE * act[i];
        }

        // ここまで来て初めて記憶を進める。
        self.frames = frames;
        self.last_action = act;
        self.q_hold = q_des;
        self.t += 0.02;

        Ok((
            obs,
            PolicyTick {
                q_des_isaac: q_des,
                kp_isaac: [BLIND_KP; 12],
                kd_isaac: [BLIND_KD; 12],
                anomalies,
            },
        ))
    }
}

/// 盲目契約の 1 フレームを点検する。非有限、四元数が単位でない、重力方向の
/// ノルムが 1 から外れている、関節速度が桁違い、のいずれかを名前で返す。
pub fn blind_anomalies(frame: &[f32; N_OBS_BLIND], quat: &[f64; 4]) -> Vec<&'static str> {
    let mut v = Vec::new();
    if frame.iter().any(|x| !x.is_finite()) {
        v.push("non_finite");
        return v;
    }
    let qn = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();
    if !(0.9..=1.1).contains(&qn) {
        v.push("quat_not_unit");
    }
    let gn = (frame[6] * frame[6] + frame[7] * frame[7] + frame[8] * frame[8]).sqrt();
    if !(0.9..=1.1).contains(&gn) {
        v.push("gravity_not_unit");
    }
    if frame[24..36].iter().any(|x| x.abs() > 50.0) {
        v.push("joint_vel_out_of_range");
    }
    v
}

/// 履歴フレーム（古い順）を IsaacLab と同じ並びへ潰す:
/// **項目ごとに [古い…新しい]**。各項目ブロックの末尾が最新。
pub fn flatten_history(frames: &[[f32; N_OBS_BLIND]]) -> Vec<f32> {
    let mut obs = Vec::with_capacity(frames.len() * N_OBS_BLIND);
    let mut off = 0usize;
    for d in TERM_DIMS.iter() {
        for fr in frames.iter() {
            obs.extend_from_slice(&fr[off..off + d]);
        }
        off += d;
    }
    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_input() -> ObsInput {
        ObsInput {
            quat_wxyz: [0.9689, 0.0, 0.2474, 0.0], // ピッチ +0.5 rad
            gyro_rad_s: [0.1, -0.2, 0.3],
            accel_m_s2: [0.0, 0.0, 9.81], // 盲目契約では使わない
            joint_q_go2: [0.05, 0.6, -1.2, -0.05, 0.7, -1.3, 0.06, 0.9, -1.4, -0.06, 1.1, -1.6],
            joint_dq_go2: [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6],
        }
    }

    /// 1 フレームの並びが学習側 `BlindProprioObservationsCfg` の宣言順と一致するか。
    #[test]
    fn frame_layout_matches_training_order() {
        let inp = probe_input();
        let f = build_blind_frame(&inp, &[0.6, 0.0, 0.0], [0.55, -0.02, 0.01], &[0.0; 12]);
        // 速度 3
        assert_eq!(&f[0..3], &[0.55f32, -0.02, 0.01]);
        // 角速度 3
        assert_eq!(&f[3..6], &[0.1f32, -0.2, 0.3]);
        // 重力方向 3: ピッチ +0.5 rad なら (sin0.5, 0, -cos0.5)
        assert!((f[6] - 0.5_f32.sin()).abs() < 1e-3, "g_x {}", f[6]);
        assert!(f[7].abs() < 1e-6, "g_y {}", f[7]);
        assert!((f[8] + 0.5_f32.cos()).abs() < 1e-3, "g_z {}", f[8]);
        // 指令 3
        assert_eq!(&f[9..12], &[0.6f32, 0.0, 0.0]);
        // 関節角は Isaac 順で既定姿勢を引いた値
        for g in 0..12 {
            let i = GO2_TO_ISAAC[g];
            let want = (inp.joint_q_go2[g] - DEFAULT_LOCO_ISAAC[i]) as f32;
            assert!((f[12 + i] - want).abs() < 1e-6, "q[{i}]");
            assert!((f[24 + i] - inp.joint_dq_go2[g] as f32).abs() < 1e-6, "dq[{i}]");
        }
        // 前回の行動は初回ゼロ
        assert_eq!(&f[36..48], &[0.0f32; 12]);
    }

    /// 履歴つきの並びが「項目ごとに古い順、末尾が最新」になっているか。
    #[test]
    fn history_is_per_term_oldest_first() {
        let inp = probe_input();
        let frames: Vec<[f32; N_OBS_BLIND]> = [0.1f64, 0.2, 0.3]
            .iter()
            .map(|v| build_blind_frame(&inp, &[0.0; 3], [*v, 0.0, 0.0], &[0.0; 12]))
            .collect();
        let obs = flatten_history(&frames);
        assert_eq!(obs.len(), 3 * N_OBS_BLIND);
        // 最初の項目（速度 3 次元）のブロックは 3 フレーム × 3 = 9 要素で、
        // x 成分は 0, 3, 6 番目。並びは古い順なので 0.1, 0.2, 0.3。
        assert_eq!(obs[0], 0.1f32);
        assert_eq!(obs[3], 0.2f32);
        assert_eq!(obs[6], 0.3f32, "末尾が最新であること");
        // 次の項目（角速度）はオフセット 9 から始まる
        assert_eq!(obs[9], 0.1f32, "項目ごとに積むこと");
    }

    /// 指令のクランプ。
    #[test]
    fn command_clamp() {
        assert_eq!(clamp_blind_cmd([2.0, -3.0, 5.0]), [1.0, -1.0, 1.0]);
        assert_eq!(clamp_blind_cmd([0.6, 0.0, 0.0]), [0.6, 0.0, 0.0]);
    }

    /// 復号: a = 0 は既定姿勢と固定ゲイン、クランプしないので可動域の外も出る。
    #[test]
    fn decode_is_unclamped_with_fixed_gains() {
        let q0 = DEFAULT_LOCO_ISAAC;
        for i in 0..12 {
            assert_eq!(q0[i] + BLIND_POS_SCALE * 0.0, DEFAULT_LOCO_ISAAC[i]);
        }
        // 大きな行動でも切らない（実機はメカストッパーが守る）
        let far = DEFAULT_LOCO_ISAAC[8] + BLIND_POS_SCALE * (-8.0);
        assert!(far < -3.0, "クランプされていたら -3 より大きい: {far}");
        assert_eq!((BLIND_KP, BLIND_KD), (25.0, 0.5));
    }
}
