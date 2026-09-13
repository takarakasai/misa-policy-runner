//! namiashi WalkFlatRef（v7）契約 — 参照 + 残差の低速歩行方策の実行系。
//!
//! go2_rl `Isaac-Namiashi-WalkFlatRef-v0`（namiashi_rl/actions_ref.py）の
//! デプロイ側。リファレンス実装は go2_rl `sim2sim_namiashi_ref_mujoco.py`
//! （実測: MuJoCo で cmd 0.2 → 97%、0.3 → 98%、転倒なし）。
//!
//! - obs 47 = [gyro_b(3) | 重力射影_b(3) | cmd(3) | q−default(12) | dq(12)
//!   | 前回生行動(12) | clock sin/cos(2πt/0.32)(2)]、生 SI・スケールなし。
//!   関節順は **Isaac 型順**（hips|thighs|calves、脚内 FL,FR,RL,RR）。
//! - デコード: q_des = trot_target(τ, cmd) + 0.08·tanh(a)。
//!   **位置目標のみ** — MG4005E の内蔵位置ループに 50 Hz で送るだけで、
//!   MIT モードは要らない（学習側の明示 PD kp25/kd0.5 はその近似）。
//! - trot_target = 実録 Trot（[`namiashi_ref`]、0.879 m/s・0.32 s）の
//!   足空間スケール: 平面ストライド ∝ |v_cmd|/0.879、リフトは移動中
//!   50% を下限、停止指令では立位に退化。
//!
//! FK/IK は解析（関節原点は namiashi.misa から）。Python 実装とは
//! ゴールデンベクタで照合（`tests/golden_namiashi.rs`）。

use crate::namiashi_ref::{FEET_NOMINAL, FEET_REF, N_PHASE};
use crate::policy::OnnxPolicy;

/// 実録 Trot の周期 [s] とその平面速度 [m/s]。
pub const PERIOD_S: f64 = 0.32;
pub const REF_SPEED: f64 = 0.879;
/// 残差の振幅 [rad]（tanh 有界）。
pub const RESIDUAL_RAD: f64 = 0.08;
/// 学習時の明示 PD（実機では内蔵位置ループがこの近似の実体）。
pub const KP: f64 = 25.0;
pub const KD: f64 = 0.5;
/// 立位（Isaac 型順）。obs の joint_pos オフセットでもある。
pub const DEFAULT_ISAAC: [f64; 12] = [
    0.0, 0.0, 0.0, 0.0, 0.695, 0.695, 0.695, 0.695, -1.390, -1.390, -1.390, -1.390,
];
/// 学習時の指令包絡（v7: 前進のみ + 停止。後退・高 wz は分布外）。
pub const CMD_VX_RANGE: (f64, f64) = (0.0, 0.35);
pub const CMD_VY_RANGE: (f64, f64) = (-0.10, 0.10);
pub const CMD_WZ_RANGE: (f64, f64) = (-0.40, 0.40);

pub const N_OBS: usize = 47;
pub const N_ACT: usize = 12;

/// 脚チェーンの寸法（namiashi.misa の関節原点。FK/IK 往復 3e-7 rad 検証済み）。
const HX: f64 = 0.147;
const HY: f64 = 0.0344;
const LAT: f64 = 0.0747;
const LINK: f64 = 0.1528;
const SX: [f64; 4] = [1.0, 1.0, -1.0, -1.0];
const SY: [f64; 4] = [1.0, -1.0, 1.0, -1.0];

pub fn clamp_cmd(mut c: [f64; 3]) -> [f64; 3] {
    c[0] = c[0].clamp(CMD_VX_RANGE.0, CMD_VX_RANGE.1);
    c[1] = c[1].clamp(CMD_VY_RANGE.0, CMD_VY_RANGE.1);
    c[2] = c[2].clamp(CMD_WZ_RANGE.0, CMD_WZ_RANGE.1);
    c
}

/// 解析 FK: 関節角（Isaac 型順）→ 足先（胴体座標系）。
pub fn fk(q: &[f64; 12]) -> [[f64; 3]; 4] {
    let mut feet = [[0.0f64; 3]; 4];
    for l in 0..4 {
        let (h, t, k) = (q[l], q[4 + l], q[8 + l]);
        let x = -LINK * (t.sin() + (t + k).sin());
        let zp = -LINK * (t.cos() + (t + k).cos());
        let lat = LAT * SY[l];
        feet[l] = [
            x + HX * SX[l],
            lat * h.cos() - zp * h.sin() + HY * SY[l],
            lat * h.sin() + zp * h.cos(),
        ];
    }
    feet
}

/// 解析 IK: 足先（胴体座標系）→ 関節角（Isaac 型順）。
pub fn ik(feet: &[[f64; 3]; 4]) -> [f64; 12] {
    let mut q = [0.0f64; 12];
    for l in 0..4 {
        let x = feet[l][0] - HX * SX[l];
        let y = feet[l][1] - HY * SY[l];
        let z = feet[l][2];
        let lat = LAT * SY[l];
        let zp = -(y * y + z * z - lat * lat).max(1e-9).sqrt();
        let h = z.atan2(y) - zp.atan2(lat);
        let h = h.sin().atan2(h.cos());
        let c = ((x * x + zp * zp - 2.0 * LINK * LINK) / (2.0 * LINK * LINK))
            .clamp(-0.99999, 0.99999);
        let k = -c.acos();
        q[l] = h;
        q[4 + l] = (-x).atan2(-zp) - k.sin().atan2(1.0 + k.cos());
        q[8 + l] = k;
    }
    q
}

/// 参照関節角: 実録 Trot の足空間スケール（τ は周期単位、mod 1）。
pub fn trot_target(tau: f64, cmd: [f64; 3]) -> [f64; 12] {
    let f = tau.rem_euclid(1.0) * (N_PHASE - 1) as f64;
    let i0 = f.floor() as usize;
    let i1 = (i0 + 1).min(N_PHASE - 1);
    let w = f - i0 as f64;
    let planar = (cmd[0] * cmd[0] + cmd[1] * cmd[1]).sqrt();
    let s = (planar / REF_SPEED).clamp(0.0, 1.0);
    let moving = planar + 0.3 * cmd[2].abs() > 0.03;
    let s_z = if moving { s.max(0.5) } else { 0.0 };
    let mut feet = [[0.0f64; 3]; 4];
    for l in 0..4 {
        for k in 0..3 {
            let v = FEET_REF[i0][l][k] * (1.0 - w) + FEET_REF[i1][l][k] * w;
            feet[l][k] = if k < 2 {
                FEET_NOMINAL[l][k] + s * (v - FEET_NOMINAL[l][k])
            } else {
                FEET_NOMINAL[l][2] + s_z * (v - FEET_NOMINAL[l][2])
            };
        }
    }
    ik(&feet)
}

/// センサ入力（関節は **Isaac 型順**。misa の脚順からはホストが並べ替える）。
#[derive(Clone, Copy, Debug)]
pub struct NamiashiObsInput {
    /// 角速度 [rad/s]、胴体座標系（ジャイロ）。
    pub gyro_rad_s: [f64; 3],
    /// 重力射影（単位ベクトル、胴体座標系）。姿勢角から
    /// `[−sinθ·…]` を組むか、[`gravity_from_rpy`] を使う。
    pub gravity_b: [f64; 3],
    pub joint_q_isaac: [f64; 12],
    pub joint_dq_isaac: [f64; 12],
}

/// rpy（ZYX）→ 重力射影 R(q)ᵀ(0,0,−1)。IMU が姿勢角しか出さない機体用。
pub fn gravity_from_rpy(rpy: [f64; 3]) -> [f64; 3] {
    let (sr, cr) = rpy[0].sin_cos();
    let (sp, cp) = rpy[1].sin_cos();
    // R = Rz(yaw)·Ry(pitch)·Rx(roll) の第 3 行に −1 を掛けたもの（yaw 不依存）
    [sp, -sr * cp, -cr * cp]
}

/// v7 コントローラ: クロック保持 + 観測組み立て + 推論 + デコード。
/// 50 Hz で [`Self::tick`] を呼び、返る q_des（Isaac 型順）を位置目標として
/// 送る。ゲインは固定（実機は内蔵位置ループ）。
pub struct NamiashiRefController {
    policy: OnnxPolicy,
    t: f64,
    last_action: [f64; 12],
    held: [f64; 12],
}

impl NamiashiRefController {
    pub fn new(policy: OnnxPolicy) -> Result<Self, String> {
        if policy.n_obs() != N_OBS {
            return Err(format!(
                "v7 契約は {N_OBS} 入力（このグラフは {}）",
                policy.n_obs()
            ));
        }
        Ok(Self {
            policy,
            t: 0.0,
            last_action: [0.0; 12],
            held: DEFAULT_ISAAC,
        })
    }

    /// 立位（ランプ先・停止時の安全保持）。
    pub fn default_pose_isaac(&self) -> [f64; 12] {
        DEFAULT_ISAAC
    }

    pub fn reset(&mut self) {
        self.t = 0.0;
        self.last_action = [0.0; 12];
        self.held = DEFAULT_ISAAC;
    }

    pub fn gait_time_s(&self) -> f64 {
        self.t
    }

    /// 推論失敗時の保持（直前の目標）。
    pub fn hold(&self) -> [f64; 12] {
        self.held
    }

    /// 1 tick（50 Hz）。`cmd` は内部で学習包絡にクランプ。
    /// 返り値は q_des（Isaac 型順、rad）。
    pub fn tick(&mut self, inp: &NamiashiObsInput, cmd_raw: [f64; 3]) -> Result<[f64; 12], String> {
        let cmd = clamp_cmd(cmd_raw);
        let tau = self.t / PERIOD_S;
        let ang = 2.0 * std::f64::consts::PI * tau;
        let mut obs = Vec::with_capacity(N_OBS);
        for v in inp.gyro_rad_s {
            obs.push(v as f32);
        }
        for v in inp.gravity_b {
            obs.push(v as f32);
        }
        for v in cmd {
            obs.push(v as f32);
        }
        for i in 0..12 {
            obs.push((inp.joint_q_isaac[i] - DEFAULT_ISAAC[i]) as f32);
        }
        for v in inp.joint_dq_isaac {
            obs.push(v as f32);
        }
        for v in self.last_action {
            obs.push(v as f32);
        }
        obs.push(ang.sin() as f32);
        obs.push(ang.cos() as f32);

        let a12 = self.policy.infer_n(&obs, N_ACT)?;
        let reference = trot_target(tau, cmd);
        let mut q = [0.0f64; 12];
        for i in 0..12 {
            self.last_action[i] = a12[i];
            q[i] = reference[i] + RESIDUAL_RAD * a12[i].tanh();
        }
        self.held = q;
        self.t += 1.0 / 50.0;
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FK(IK(feet)) の往復（参照テーブル全 48 位相）。
    #[test]
    fn fk_inverts_ik_over_the_reference() {
        for i in 0..N_PHASE {
            let q = ik(&FEET_REF[i]);
            let feet = fk(&q);
            for l in 0..4 {
                for k in 0..3 {
                    assert!(
                        (feet[l][k] - FEET_REF[i][l][k]).abs() < 1e-8,
                        "phase {i} leg {l} axis {k}"
                    );
                }
            }
        }
    }

    /// 停止指令 → 参照は立位ぴったり。
    #[test]
    fn standing_reference_is_stance() {
        let q = trot_target(0.37, [0.0; 3]);
        for i in 0..12 {
            assert!((q[i] - DEFAULT_ISAAC[i]).abs() < 1e-6, "q[{i}] = {}", q[i]);
        }
    }

    /// 重力射影: 水平で (0,0,−1)、前傾 90° で (1,0,0)。
    #[test]
    fn gravity_from_rpy_conventions() {
        let g = gravity_from_rpy([0.0, 0.0, 0.7]);
        assert!((g[0]).abs() < 1e-12 && (g[1]).abs() < 1e-12 && (g[2] + 1.0).abs() < 1e-12);
        let g = gravity_from_rpy([0.0, std::f64::consts::FRAC_PI_2, 0.0]);
        assert!((g[0] - 1.0).abs() < 1e-9, "{g:?}");
    }
}
