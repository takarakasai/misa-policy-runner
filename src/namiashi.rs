//! namiashi WalkFlat 契約（v12 以降、v13 "Adapt" / v14 "AdaptV2" で学習した
//! MLP が対象）— 参照 + 残差の全方位低速歩行方策の実行系。
//!
//! go2_rl `Isaac-Namiashi-WalkFlatAdapt*-v0`（namiashi_rl/actions_ref.py、
//! mdp_custom._trot_target_q）のデプロイ側。リファレンス実装は go2_rl
//! `sim2sim_namiashi_ref_mujoco.py`（MuJoCo で前進 0.2 → 95–107%、kp 15–40 ×
//! 遅延 0–15 ms で転倒なし）。
//!
//! - obs 47 = [gyro_b(3) | 重力射影_b(3) | cmd(3) | q−default(12) | dq(12)
//!   | 前回生行動(12) | clock sin/cos(2πt/0.32)(2)]、生 SI・スケールなし。
//!   関節順は **Isaac 型順**（hips|thighs|calves、脚内 FL,FR,RL,RR）。
//! - デコード: q_des = trot_target(τ, cmd) + res(cmd)·tanh(a)。
//!   res は旋回ゲート付き: 0.08 rad から |wz|/0.5 に比例して 0.30 rad まで
//!   （v11 以降）。**位置目標のみ** — MG4005E の内蔵位置ループに 50 Hz で
//!   送るだけで、MIT モードは要らない（学習側の明示 PD kp25/kd0.5 はその
//!   近似。v13 以降はゲイン ×0.6–1.6・遅延 0–20 ms の DR で学習）。
//! - trot_target は**完全解析**（v10 以降）: タイミング（duty、対角オフセット、
//!   リフト）だけ実録 Trot から採寸し、ストライドは足ごとの指令速度
//!   −(v + wz×r) から閉形式で引く。
//! - **歩容の形は契約世代で違う（[`RefGaitCfg`]）**: v12 は旋回指令で
//!   duty 0.60→0.50 / リフト 38→60 mm / ヨー利得 2→8 へ morph + 残差ゲート
//!   0.08→0.30。**v15 はヨー計測の ±π 巻き付き発覚後の再較正**（go2_rl
//!   doc/namiashi_policy_architecture.md §6）: morph 無し・一様 yg 1.25・
//!   一様残差 0.08。v15 のデプロイ標準は
//!   `2026-09-19_00-32-14_v15_seedcal/exported/policy_1899.onnx`
//!   （純旋回 Isaac 104% / MuJoCo 108–113%、決定論リセット 0）。
//!   **ONNX の入力幅は両世代とも 47 なので自動判別できない** — ホストが
//!   チェックポイントに合わせて選ぶ（間違えると歩く。悪く。エラーは出ない）。
//!
//! FK/IK は解析（関節原点は namiashi.misa から）。Python 実装とは
//! ゴールデンベクタで照合（`tests/golden_namiashi.rs`）。

use crate::namiashi_ref::FEET_NOMINAL;
use crate::policy::OnnxPolicy;

/// 実録 Trot の周期 [s]（クロックと参照のタイミングの基準）。
pub const PERIOD_S: f64 = 0.32;
/// 残差の基本振幅 [rad]（tanh 有界）と、旋回ゲートの最大振幅・基準 |wz|。
pub const RESIDUAL_RAD: f64 = 0.08;
pub const TURN_RESIDUAL_RAD: f64 = 0.30;
pub const TURN_WZ_REF: f64 = 0.5;
/// 学習時の明示 PD（実機では内蔵位置ループがこの近似の実体）。
pub const KP: f64 = 25.0;
pub const KD: f64 = 0.5;
/// 立位（Isaac 型順）。obs の joint_pos オフセットでもある。
pub const DEFAULT_ISAAC: [f64; 12] = [
    0.0, 0.0, 0.0, 0.0, 0.695, 0.695, 0.695, 0.695, -1.390, -1.390, -1.390, -1.390,
];
/// 学習時の指令包絡（v12: 全方位。wz は ±0.4 で学習したが、MuJoCo では
/// |wz| 0.4 の純旋回が転倒するので、運用の既定はホスト側でさらに絞る）。
pub const CMD_VX_RANGE: (f64, f64) = (-0.15, 0.35);
pub const CMD_VY_RANGE: (f64, f64) = (-0.15, 0.15);
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

/// 参照歩容のタイミング（実録 Trot から採寸、v10）。
const DIAG_OFFSETS: [f64; 4] = [0.89, 0.39, 0.39, 0.89];

/// 参照歩容と残差の契約世代パラメータ。学習時の設定と**一致必須**。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefGaitCfg {
    /// 歩行側のヨー・ストライド利得（turn_frac = 0 の端）。
    pub yg_walk: f64,
    /// 旋回側のヨー・ストライド利得（turn_frac = 1 の端）。
    pub yg_turn: f64,
    /// 旋回側の duty（歩行側は 0.60 固定）。
    pub turn_duty: f64,
    /// 旋回側の遊脚リフト [m]（歩行側は 0.038 固定）。
    pub turn_lift_m: f64,
    /// 残差振幅 [rad]（wz = 0 の端）。
    pub residual_rad: f64,
    /// |wz| = [`TURN_WZ_REF`] での残差振幅 [rad]。
    pub turn_residual_rad: f64,
    /// 前進（vx > 0）だけのストライド較正利得（v16。後退・vy には掛けない —
    /// open-loop 実測で前進 50% / 後退 123% の非対称のため）。
    pub sg_fwd: f64,
    /// 後退（vx < 0）だけのストライド較正利得（v24。open-loop で後退は 123%
    /// 過剰なので 0.78 で割り戻す）。v12〜v16 は 1.0。
    pub sg_back: f64,
}

impl RefGaitCfg {
    /// v12〜v14 のチェックポイント（旋回モーフ + 残差ゲート）。
    pub fn v12() -> Self {
        Self { yg_walk: 2.0, yg_turn: 8.0, turn_duty: 0.50, turn_lift_m: 0.060,
               residual_rad: RESIDUAL_RAD, turn_residual_rad: TURN_RESIDUAL_RAD, sg_fwd: 1.0, sg_back: 1.0 }
    }

    /// v15（種較正）: morph 無し・一様 yg 1.25・一様残差 0.08。
    pub fn v15() -> Self {
        Self { yg_walk: 1.25, yg_turn: 1.25, turn_duty: 0.60, turn_lift_m: 0.038,
               residual_rad: RESIDUAL_RAD, turn_residual_rad: RESIDUAL_RAD, sg_fwd: 1.0, sg_back: 1.0 }
    }

    /// v16（前進ストライド較正）: v15 + 前進のみ sg 1.4。**2026-09-25 に v24 へ
    /// 標準を譲った**（[`RefGaitCfg::v24`]）。対照用 =
    /// `2026-09-19_01-11-02_v16_sgfwd/exported/policy_4598.onnx`
    /// （MuJoCo 前進 99–107% / 後退 75% / 旋回 105–108% / 複合 vx 109%・
    /// ヨー 102%、Isaac 決定論リセット 1/48）。
    pub fn v16() -> Self {
        Self { sg_fwd: 1.4, ..Self::v15() }
    }

    /// v24（後退の 3 段設計 + 較正、ゼロ学習・正規化あり系譜）: v15 +
    /// sg_fwd 1.25 / sg_back 0.78。**デプロイ標準（2026-09-25〜）** =
    /// `2026-09-23_20-12-21_v24_long_s103/exported/policy_5500.onnx`。
    /// 体座標メトリクス + 方位サーボ（[`crate::heading`]、旋回中も補正）の
    /// Rust sim 25 ケースで v16 を全軸で上回る: 平均 |追従誤差| 5.1pt 対
    /// 11.2pt、最大傾き 4.5° 対 6.0°、後退 105–111% 対 71–72%、旋回・複合
    /// ヨー 100%、転倒 0（go2_rl doc/namiashi_policy_architecture.md §18）。
    pub fn v24() -> Self {
        Self { sg_fwd: 1.25, sg_back: 0.78, ..Self::v15() }
    }

    /// 残差振幅 [rad]: wz でゲート（両端が同値なら定数）。
    pub fn residual_rad(&self, cmd: [f64; 3]) -> f64 {
        let wz_frac = (cmd[2].abs() / TURN_WZ_REF).clamp(0.0, 1.0);
        self.residual_rad + wz_frac * (self.turn_residual_rad - self.residual_rad)
    }
}

pub fn clamp_cmd(mut c: [f64; 3]) -> [f64; 3] {
    c[0] = c[0].clamp(CMD_VX_RANGE.0, CMD_VX_RANGE.1);
    c[1] = c[1].clamp(CMD_VY_RANGE.0, CMD_VY_RANGE.1);
    c[2] = c[2].clamp(CMD_WZ_RANGE.0, CMD_WZ_RANGE.1);
    c
}

/// 残差振幅 [rad]: 旋回指令で 0.08 → 0.30 に広がる（actions_ref.py と同数値）。
pub fn residual_rad(cmd: [f64; 3]) -> f64 {
    let wz_frac = (cmd[2].abs() / TURN_WZ_REF).clamp(0.0, 1.0);
    RESIDUAL_RAD + wz_frac * (TURN_RESIDUAL_RAD - RESIDUAL_RAD)
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

/// 参照関節角（v12 解析形）: τ は周期単位（mod 1 はここで取る）。
///
/// mdp_custom._trot_target_q / sim2sim_namiashi_ref_mujoco.py の trot_target
/// と同数値（ゴールデンベクタ 1e-7）。停止指令では立位ぴったり。
pub fn trot_target(tau: f64, cmd: [f64; 3]) -> [f64; 12] {
    trot_target_cfg(tau, cmd, &RefGaitCfg::v12())
}

/// 参照関節角（契約世代パラメータ付き）。v15 は [`RefGaitCfg::v15`]。
pub fn trot_target_cfg(tau: f64, cmd: [f64; 3], g: &RefGaitCfg) -> [f64; 12] {
    let tau = tau.rem_euclid(1.0);
    let planar = (cmd[0] * cmd[0] + cmd[1] * cmd[1]).sqrt();
    let yaw_c = 0.3 * cmd[2].abs();
    let twist = planar + yaw_c;
    let moving = if twist > 0.03 { 1.0 } else { 0.0 };
    // 旋回歩容への morph（旋回の割合 turn_frac ∈ [0,1]。v15 は両端同値 = 無効）
    let tf = if twist > 1e-3 { yaw_c / twist } else { 0.0 };
    let duty = 0.60 - (0.60 - g.turn_duty) * tf;
    let lift_m = 0.038 + (g.turn_lift_m - 0.038) * tf;
    let yg = g.yg_walk + (g.yg_turn - g.yg_walk) * tf;
    let scale = duty * PERIOD_S * moving;

    let mut feet = [[0.0f64; 3]; 4];
    for l in 0..4 {
        let phase = (tau + DIAG_OFFSETS[l]).rem_euclid(1.0);
        let u = ((phase - duty) / (1.0 - duty)).clamp(0.0, 1.0);
        let blend = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
        let swing = phase >= duty;
        let profile = if swing { blend - 0.5 } else { 0.5 - phase / duty };
        let nom = FEET_NOMINAL[l];
        let sgx = if cmd[0] > 0.0 { g.sg_fwd } else if cmd[0] < 0.0 { g.sg_back } else { 1.0 };
        let ux = sgx * cmd[0] - yg * cmd[2] * nom[1];
        let uy = cmd[1] + yg * cmd[2] * nom[0];
        let s = (std::f64::consts::PI * u).sin();
        let lift = if swing { lift_m * s * s * moving } else { 0.0 };
        feet[l] = [nom[0] + ux * profile * scale, nom[1] + uy * profile * scale, nom[2] + lift];
    }
    ik(&feet)
}

/// センサ入力（関節は **Isaac 型順**。misa の脚順からはホストが並べ替える）。
#[derive(Clone, Copy, Debug)]
pub struct NamiashiObsInput {
    /// 角速度 [rad/s]、胴体座標系（ジャイロ）。
    pub gyro_rad_s: [f64; 3],
    /// 重力射影（単位ベクトル、胴体座標系）。姿勢角から
    /// [`gravity_from_rpy`] で組む。
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

/// misa の脚順（FL,FR,RL,RR × hip,thigh,calf）の軸 i → Isaac 型順の添字。
pub fn misa_to_isaac(i: usize) -> usize {
    (i % 3) * 4 + i / 3
}

/// コントローラ: クロック保持 + 観測組み立て + 推論 + デコード。
/// 50 Hz で [`Self::tick`] を呼び、返る q_des（Isaac 型順）を位置目標として
/// 送る。ゲインは固定（実機は内蔵位置ループ）。
pub struct NamiashiRefController {
    policy: OnnxPolicy,
    gait: RefGaitCfg,
    t: f64,
    last_action: [f64; 12],
    held: [f64; 12],
}

impl NamiashiRefController {
    /// `gait` はチェックポイントの世代に合わせる（[`RefGaitCfg::v15`] が
    /// デプロイ標準、v12〜v14 の ONNX には [`RefGaitCfg::v12`]）。
    pub fn new(policy: OnnxPolicy, gait: RefGaitCfg) -> Result<Self, String> {
        if policy.n_obs() != N_OBS {
            return Err(format!(
                "namiashi 契約は {N_OBS} 入力（このグラフは {}）— 履歴学生や教師の ONNX は載らない",
                policy.n_obs()
            ));
        }
        Ok(Self {
            policy,
            gait,
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
        let reference = trot_target_cfg(tau, cmd, &self.gait);
        let res = self.gait.residual_rad(cmd);
        let mut q = [0.0f64; 12];
        for i in 0..12 {
            self.last_action[i] = a12[i];
            q[i] = reference[i] + res * a12[i].tanh();
        }
        self.held = q;
        self.t += 1.0 / 50.0;
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namiashi_ref::{FEET_REF, N_PHASE};

    /// FK(IK(feet)) の往復（実録参照テーブル全 48 位相）。
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

    /// 立位の FK が埋め込みの公称足先と一致（参照生成の基準）。
    #[test]
    fn nominal_feet_are_fk_of_the_stance() {
        let feet = fk(&DEFAULT_ISAAC);
        for l in 0..4 {
            for k in 0..3 {
                assert!((feet[l][k] - FEET_NOMINAL[l][k]).abs() < 1e-9, "leg {l} axis {k}");
            }
        }
    }

    /// 停止指令 → 参照は立位ぴったり（位相によらず）。
    #[test]
    fn standing_reference_is_stance() {
        for tau in [0.0, 0.37, 0.71, 1.9] {
            let q = trot_target(tau, [0.0; 3]);
            for i in 0..12 {
                assert!((q[i] - DEFAULT_ISAAC[i]).abs() < 1e-6, "tau {tau} q[{i}] = {}", q[i]);
            }
        }
    }

    /// 旋回ゲート: wz 0 で 0.08、wz 0.5 以上で 0.30、その間は線形。
    #[test]
    fn residual_gate_follows_the_training_rule() {
        assert!((residual_rad([0.2, 0.0, 0.0]) - 0.08).abs() < 1e-12);
        assert!((residual_rad([0.0, 0.0, 0.25]) - 0.19).abs() < 1e-12);
        assert!((residual_rad([0.0, 0.0, -0.9]) - 0.30).abs() < 1e-12);
    }

    /// 重力射影: 水平で (0,0,−1)、前傾 90° で (1,0,0)。
    #[test]
    fn gravity_from_rpy_conventions() {
        let g = gravity_from_rpy([0.0, 0.0, 0.7]);
        assert!((g[0]).abs() < 1e-12 && (g[1]).abs() < 1e-12 && (g[2] + 1.0).abs() < 1e-12);
        let g = gravity_from_rpy([0.0, std::f64::consts::FRAC_PI_2, 0.0]);
        assert!((g[0] - 1.0).abs() < 1e-9, "{g:?}");
    }

    /// misa 脚順 → Isaac 型順の並べ替えは全単射。
    #[test]
    fn misa_to_isaac_is_a_permutation() {
        let mut seen = [false; 12];
        for i in 0..12 {
            seen[misa_to_isaac(i)] = true;
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(misa_to_isaac(0), 0); // FL hip
        assert_eq!(misa_to_isaac(1), 4); // FL thigh
        assert_eq!(misa_to_isaac(5), 9); // FR calf
    }
}
