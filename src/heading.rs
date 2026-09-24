//! 世界系ヨーの PI 方位保持 — 方策の**外側**に置く積分器（機体非依存）。
//!
//! 学習した歩行方策はリセット時にヨーを乱数化されて世界系の方位参照を
//! 持たないので、直進指令でも小さなヨー偏りがそのまま曲率になる。報酬で
//! 偏りを殺す手は Go2（GRU、8 本の一因子で最大 −34 %）でも namiashi
//! （v31 方位保持報酬、偏り −0.04〜−0.08 rad/s のまま）でも効かなかった。
//! 積分器を外に置けば厳密で、実機の IMU ヨーがそのまま参照になる。
//!
//! Go2 実績（go2-runner、KP 2 / KI 0.5、15 s）: 横流れ 1.0 m/s で
//! −9.47 → −0.21 m、1.5 で −14.38 → −0.50 m、速度・傾き不変。
//!
//! - 目標ヨーは「操作開始時の実ヨー + ∫wz_user dt」。旋回指令は参照が積む
//!   ので、意図した旋回は妨げない。
//! - 補正は |wz_user| < [`HeadingServo::STRAIGHT_WZ`] かつ移動中だけ。積分は
//!   旋回中と clip に当たった直後は凍結（anti-windup）。
//! - |e| > 45° または 停止→移動の操作開始で参照を現在ヨーに張り直す
//!   （±π の折り返しで勾配が逆を向く事故を防ぐ）。
//! - **静止中は補正しない**: IMU ヨーはジャイロ積分なので静止中も漂い、
//!   それを打ち消すと機体がその場でじわじわ回る。
//!
//! 使い方: 方策に渡す直前の指令に [`HeadingServo::apply`] を掛ける（レート
//! 制限があるならその**手前**）。

/// 角度を (−π, π] に折る。
pub fn wrap_pi(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}

/// クォータニオン (w, x, y, z) → ヨー（ZYX）。
pub fn yaw_from_quat_wxyz(q: &[f64; 4]) -> f64 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

pub struct HeadingServo {
    pub kp: f64,
    pub ki: f64,
    /// 補正の上限 [rad/s]。学習した wz 分布の一部に収めること。
    pub clip: f64,
    /// 「移動中」とみなす指令の大きさ（平面 [m/s]、ヨーは同じ数値を [rad/s]
    /// として使う）。低速機（namiashi 0.1–0.3 m/s）では既定 0.05 のままでよい。
    ///
    /// **その場旋回も「移動」に含める** — 操縦者が動きを指令している状態だから。
    /// 既定（`straight_wz` 0.05）では旋回中は補正しないので挙動は変わらず、
    /// `straight_wz` を上げたときだけ「その場旋回の追従も直す」に効く。
    pub moving_threshold: f64,
    /// これ未満の |wz| 指令のときだけ補正する [rad/s]。既定
    /// [`HeadingServo::STRAIGHT_WZ`] = 0.05（直進だけ）。**大きくすると旋回中も
    /// 補正する**: 参照は ∫wz_cmd を積んでいるので「指令どおりの角速度で回る」
    /// 向きに働き、旋回の過不足と複合指令でのヨー偏りを同時に直せる。
    /// 代償は姿勢の乱れ（Go2 では傾き 2.7 → 4.1°）。
    pub straight_wz: f64,
    reference: Option<f64>,
    integral: f64,
    last_corr: f64,
    was_moving: bool,
    pub last_err: f64,
    abs_corr_sum: f64,
    samples: u64,
}

impl HeadingServo {
    pub const STRAIGHT_WZ: f64 = 0.05;
    pub const RESET_ERR_RAD: f64 = std::f64::consts::FRAC_PI_4;

    pub fn new(kp: f64, ki: f64) -> Self {
        Self {
            kp,
            ki,
            clip: 0.30,
            moving_threshold: 0.05,
            straight_wz: Self::STRAIGHT_WZ,
            reference: None,
            integral: 0.0,
            last_corr: 0.0,
            was_moving: false,
            last_err: 0.0,
            abs_corr_sum: 0.0,
            samples: 0,
        }
    }

    /// 1 tick 進めて、サーボ込みの目標指令 `[vx, vy, wz]` を返す。
    pub fn apply(&mut self, yaw: f64, user: [f64; 3], dt: f64) -> [f64; 3] {
        let moving = user[0].abs() >= self.moving_threshold
            || user[1].abs() >= self.moving_threshold
            || user[2].abs() >= self.moving_threshold;
        if self.reference.is_none() || (moving && !self.was_moving) {
            self.reference = Some(yaw);
            self.integral = 0.0;
            self.last_corr = 0.0;
        }
        self.was_moving = moving;
        // **誤差を取ってから参照を進める。** 逆順だと旋回中は常に wz·dt
        // （0.3 rad/s なら 6 mrad）だけ遅れて見え、指令どおり回っていても
        // 積分が溜まり続ける（straight_wz を上げた途端に出る）。
        let mut reference = self.reference.unwrap();
        let mut err = wrap_pi(yaw - reference);
        if err.abs() > Self::RESET_ERR_RAD {
            reference = yaw;
            err = 0.0;
            self.integral = 0.0;
        }
        self.reference = Some(wrap_pi(reference + user[2] * dt));
        let straight = moving && user[2].abs() < self.straight_wz;
        let unclipped = self.last_corr.abs() < self.clip - 1.0e-9;
        if straight && unclipped {
            self.integral += err * dt;
        }
        let corr = if straight {
            (-self.kp * err - self.ki * self.integral).clamp(-self.clip, self.clip)
        } else {
            0.0
        };
        self.last_corr = corr;
        self.last_err = err;
        self.abs_corr_sum += corr.abs();
        self.samples += 1;
        [user[0], user[1], user[2] + corr]
    }

    pub fn mean_abs_corr(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.abs_corr_sum / self.samples as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quat_yaw(yaw: f64) -> [f64; 4] {
        [(yaw / 2.0).cos(), 0.0, 0.0, (yaw / 2.0).sin()]
    }

    #[test]
    fn yaw_round_trips_through_the_quaternion() {
        for yaw in [-3.0, -1.0, 0.0, 0.7, 2.9] {
            assert!((yaw_from_quat_wxyz(&quat_yaw(yaw)) - yaw).abs() < 1e-12);
        }
    }

    /// 定常ヨー偏り（namiashi v24 の前進 −0.05 rad/s 相当）を DC 外乱として
    /// 与えた 1 次モデル: 15 s で方位誤差が消え、補正が偏りに落ち着く。
    #[test]
    fn pi_cancels_a_constant_yaw_bias() {
        let bias = -0.05;
        let dt = 0.02;
        let mut sv = HeadingServo::new(2.0, 0.5);
        let mut yaw = 0.3;
        let mut applied = [0.0; 3];
        for _ in 0..750 {
            applied = sv.apply(yaw, [0.2, 0.0, 0.0], dt);
            yaw += (applied[2] + bias) * dt;
        }
        assert!(sv.last_err.abs() < 0.5f64.to_radians(), "err {:.4}", sv.last_err);
        assert!((applied[2] - (-bias)).abs() < 0.01, "corr {:.4}", applied[2]);
    }

    /// straight_wz を上げると旋回中も補正する（参照は ∫wz_cmd を積むので
    /// 「指令どおりに回る」向き）。指令と実測が一致していれば補正は 0。
    #[test]
    fn raising_straight_wz_corrects_during_turns() {
        let dt = 0.02;
        let mut sv = HeadingServo::new(2.0, 0.5);
        sv.straight_wz = 1.0;
        let mut yaw = 0.0;
        // 指令 0.3 に対し実測 0.24（80% 追従）→ 不足ぶんを補う正の補正が出る。
        let mut last = [0.0; 3];
        for _ in 0..250 {
            last = sv.apply(yaw, [0.2, 0.0, 0.3], dt);
            yaw += 0.24 * dt;
        }
        assert!(last[2] > 0.3, "旋回の不足を補えていない: {:.3}", last[2]);
        // その場旋回（平面指令なし）でも補正が出る。
        let mut sv3 = HeadingServo::new(2.0, 0.5);
        sv3.straight_wz = 1.0;
        let mut y3 = 0.0;
        let mut l3 = [0.0; 3];
        for _ in 0..250 {
            l3 = sv3.apply(y3, [0.0, 0.0, 0.4], dt);
            y3 += 0.46 * dt; // 116% の過追従
        }
        assert!(l3[2] < 0.4, "その場旋回の過剰を抑えられていない: {:.3}", l3[2]);
        // 指令どおり回っている個体では補正はほぼ 0。
        let mut sv2 = HeadingServo::new(2.0, 0.5);
        sv2.straight_wz = 1.0;
        let mut y2 = 0.0;
        let mut l2 = [0.0; 3];
        for _ in 0..250 {
            l2 = sv2.apply(y2, [0.2, 0.0, 0.3], dt);
            y2 += 0.3 * dt;
        }
        assert!((l2[2] - 0.3).abs() < 0.01, "指令どおりなのに補正が出た: {:.3}", l2[2]);
    }

    /// 既定（straight_wz 0.05）では旋回指令中は補正せず、参照だけが指令を積む。
    #[test]
    fn turning_is_never_fought() {
        let dt = 0.02;
        let mut sv = HeadingServo::new(2.0, 0.5);
        let mut yaw = 0.0;
        for _ in 0..250 {
            let t = sv.apply(yaw, [0.2, 0.0, 0.3], dt);
            assert_eq!(t[2], 0.3);
            yaw += 0.3 * dt;
        }
        let t = sv.apply(yaw, [0.2, 0.0, 0.0], dt);
        assert!(t[2].abs() < 1e-6, "corr after a tracked turn {:.4}", t[2]);
    }

    /// clip に当たっている間は積分が進まない（anti-windup）。
    #[test]
    fn integral_freezes_while_clipped() {
        let dt = 0.02;
        let mut sv = HeadingServo::new(4.0, 2.0);
        let _ = sv.apply(0.0, [0.2, 0.0, 0.0], dt);
        for _ in 0..200 {
            let t = sv.apply(0.5, [0.2, 0.0, 0.0], dt);
            assert!((t[2] - (-0.30)).abs() < 1e-9);
        }
        let frozen = sv.integral;
        for _ in 0..200 {
            let _ = sv.apply(0.5, [0.2, 0.0, 0.0], dt);
        }
        assert!((sv.integral - frozen).abs() < 1e-12);
    }

    /// 大きなずれ（>45°）と停止→移動の操作開始は参照を張り直す。
    #[test]
    fn reference_resets_on_large_error_and_on_move_start() {
        let dt = 0.02;
        let mut sv = HeadingServo::new(2.0, 0.5);
        let _ = sv.apply(0.0, [0.2, 0.0, 0.0], dt);
        let t = sv.apply(1.0, [0.2, 0.0, 0.0], dt);
        assert!(t[2].abs() < 1e-9, "large error must reset, got corr {:.3}", t[2]);
        let _ = sv.apply(1.0, [0.0, 0.0, 0.0], dt);
        let _ = sv.apply(1.3, [0.0, 0.0, 0.0], dt);
        let t = sv.apply(1.3, [0.2, 0.0, 0.0], dt);
        assert!(t[2].abs() < 1e-9, "move start must reset, got corr {:.3}", t[2]);
    }

    /// 静止中は yaw が漂っても補正しない（実機 IMU の yaw はジャイロ積分）。
    #[test]
    fn standing_never_corrects_even_if_yaw_drifts() {
        let dt = 0.02;
        let mut sv = HeadingServo::new(2.0, 0.5);
        let mut yaw = 0.0;
        for _ in 0..500 {
            yaw += 0.01 * dt;
            let t = sv.apply(yaw, [0.0, 0.0, 0.0], dt);
            assert_eq!(t[2], 0.0);
        }
        assert_eq!(sv.mean_abs_corr(), 0.0);
    }
}
