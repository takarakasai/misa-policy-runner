//! Rust port of `mit_rl/natural_walk.py` — the slow four-beat reference
//! trajectory and the analytic Go2 leg kinematics (IK / FK / Jacobian).
//!
//! The trajectory and IK are verified against the Python implementation by
//! golden vectors (`tests/golden_natural.rs`); FK and the Jacobian are
//! verified against IK round-trips and finite differences. Torch computes in
//! f32, so the goldens agree to ~1e-5 — this port computes in f64.
//!
//! Frames: everything is the trunk (base) frame. Foot positions are the foot
//! ball centers; leg order FL, FR, RL, RR.

/// Gait frequency (Hz) of the Natural contract.
pub const FREQUENCY: f64 = 0.65;
/// Stance fraction of the cycle.
pub const DUTY: f64 = 0.80;
/// Standing lead-in before the clock starts (s).
pub const START_DELAY: f64 = 1.5;

/// Per-leg phase offsets of the four-beat walk (FL, FR, RL, RR).
const OFFSETS: [f64; 4] = [0.75, 0.25, 0.0, 0.5];
/// x sign per leg (front +, rear −), y sign per leg (left +, right −).
const SX: [f64; 4] = [1.0, 1.0, -1.0, -1.0];
const SY: [f64; 4] = [1.0, -1.0, 1.0, -1.0];

/// Hip pivot offsets from the trunk origin (m).
const HIP_X: f64 = 0.1934;
const HIP_Y: f64 = 0.0465;
/// Lateral hip-to-leg-plane offset (m).
const LATERAL: f64 = 0.0955;
/// Thigh and calf link length (m).
const LINK: f64 = 0.213;

/// Nominal planar foot positions.
const NOM_X: f64 = 0.1934;
const NOM_Y: f64 = 0.18;

/// One tick of the reference trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reference {
    /// Foot ball centers in the trunk frame, legs FL, FR, RL, RR.
    pub feet: [[f64; 3]; 4],
    /// True while the leg is in its swing window (and the gait is active).
    pub swing: [bool; 4],
    /// Body height reference (m) — the support wrench's height target.
    pub height: f64,
}

/// Trajectory configuration: the parameters that MUST match the training
/// config of the checkpoint being deployed (doc/mit_natural.md).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryCfg {
    /// Planar stride calibration gain (h30 STANDARD: 1.55).
    pub stride_gain: f64,
    /// Yaw stride gain (h30 STANDARD: 2.0). `None` = same as stride_gain.
    pub yaw_stride_gain: Option<f64>,
    /// Constant body height (m); `None` keeps the legacy 0.40 → 0.36 ramp.
    /// (h30 STANDARD: Some(0.30)).
    pub body_height: Option<f64>,
    /// Support-shift scale (training default 1.0).
    pub support_shift_gain: f64,
}

/// Lowest body height the h30 checkpoint was measured at (go2_rl
/// `doc/push_robustness.md` §7.4). Below this the speed envelope breaks down:
/// at 0.20 m, forward 0.24 m/s tracks 75% and a combined command 48%.
pub const MIN_BODY_HEIGHT: f64 = 0.21;
/// Highest measured body height (the `low05` variant's 0.31 m).
pub const MAX_BODY_HEIGHT: f64 = 0.31;

impl TrajectoryCfg {
    /// The STANDARD deploy checkpoint's configuration
    /// (`Isaac-MIT-NaturalH30-Go2-v0`: gains 1.55 / 2.0, height 0.30 m).
    pub fn h30_standard() -> Self {
        Self {
            stride_gain: 1.55,
            yaw_stride_gain: Some(2.0),
            body_height: Some(0.30),
            support_shift_gain: 1.0,
        }
    }

    /// Yaw stride gain calibrated for `height`.
    ///
    /// The h30 gain 2.0 is calibrated at 0.30 m. Crouching makes the SAME
    /// policy over-track yaw (133% of a 0.4 rad/s command at 0.21 m) because
    /// the per-foot yaw stride is applied at a shorter leg. The error is in
    /// the host-side reference, not the network, so it is fixed here rather
    /// than by retraining. Measured 103–108% across 0.21–0.30 m
    /// (go2_rl `doc/push_robustness.md` §7.2).
    pub fn yaw_gain_for_height(height: f64) -> f64 {
        2.0 + 4.444 * (height - 0.30)
    }

    /// This config at a different body height, with the yaw gain recalibrated.
    ///
    /// `height` is clamped to the measured range [`MIN_BODY_HEIGHT`,
    /// `MAX_BODY_HEIGHT`]; `stride_gain` is unchanged (measured flat over the
    /// range). The policy itself is height-agnostic — the height enters only
    /// through this reference trajectory — so the SAME ONNX covers the range.
    pub fn with_height(self, height: f64) -> Self {
        let h = height.clamp(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT);
        Self {
            body_height: Some(h),
            yaw_stride_gain: Some(Self::yaw_gain_for_height(h)),
            ..self
        }
    }

    /// The low-stance deploy configuration: 0.22 m, yaw gain 1.644.
    ///
    /// Same ONNX and same `stride_gain` as [`Self::h30_standard`]. Standing
    /// push survival rises 11% → 60% and the tolerated force from 0.37 to
    /// 0.54–0.80 × body weight; tracking inside the trained envelope
    /// (|vx| ≤ 0.16 m/s) is equal or better. The cost is 8 cm of belly
    /// clearance, so the choice is a mission decision, not a default.
    pub fn h30_low_stance() -> Self {
        Self::h30_standard().with_height(0.22)
    }
}

/// Reference trajectory at time `t` (s, from gait start incl. the lead-in)
/// for body twist command `cmd = [vx, vy, wz]`.
///
/// Each foot's stance-phase planar velocity is `-(v + wz × r)` at its
/// nominal planar position, so yaw is a differential per-foot stride.
pub fn trajectory(t: f64, cmd: [f64; 3], cfg: &TrajectoryCfg) -> Reference {
    let gt = (t - START_DELAY).max(0.0);
    let moving = ((cmd[0] * cmd[0] + cmd[1] * cmd[1]).sqrt() + 0.3 * cmd[2].abs()) > 0.03;
    let ramp = if moving { (gt / 1.5).min(1.0) } else { 0.0 };
    let yaw_gain = cfg.yaw_stride_gain.unwrap_or(cfg.stride_gain);

    let height = match cfg.body_height {
        Some(h) => h,
        None => 0.40 - 0.04 * (t / 1.0).clamp(0.0, 1.0),
    };

    // support-shift weights: softmax over legs of 5·cos(2π(phase − 0.9))
    let mut phase = [0.0f64; 4];
    let mut w = [0.0f64; 4];
    let mut wmax = f64::NEG_INFINITY;
    for l in 0..4 {
        phase[l] = (gt * FREQUENCY + OFFSETS[l]).rem_euclid(1.0);
        w[l] = 5.0 * (2.0 * std::f64::consts::PI * (phase[l] - 0.9)).cos();
        wmax = wmax.max(w[l]);
    }
    let mut wsum = 0.0;
    for l in 0..4 {
        w[l] = (w[l] - wmax).exp();
        wsum += w[l];
    }
    let mut shift_x = 0.0;
    let mut shift_y = 0.0;
    for l in 0..4 {
        shift_x += w[l] / wsum * SX[l];
        shift_y += w[l] / wsum * SY[l];
    }
    let shift_x = -0.025 * cfg.support_shift_gain * shift_x * ramp;
    let shift_y = -0.035 * cfg.support_shift_gain * shift_y * ramp;

    let mut feet = [[0.0f64; 3]; 4];
    let mut swing = [false; 4];
    for l in 0..4 {
        let u = ((phase[l] - DUTY) / (1.0 - DUTY)).clamp(0.0, 1.0);
        let blend = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
        swing[l] = phase[l] >= DUTY && ramp > 0.001;
        let ux = cfg.stride_gain * cmd[0] - yaw_gain * cmd[2] * (NOM_Y * SY[l]);
        let uy = cfg.stride_gain * cmd[1] + yaw_gain * cmd[2] * (NOM_X * SX[l]);
        let profile = if phase[l] >= DUTY {
            blend - 0.5
        } else {
            0.5 - phase[l] / DUTY
        };
        let scale = (DUTY / FREQUENCY) * ramp;
        let lift = 0.065
            * (std::f64::consts::PI * u).sin().powi(2)
            * if swing[l] { ramp } else { 0.0 };
        feet[l] = [
            NOM_X * SX[l] + ux * profile * scale - shift_x,
            NOM_Y * SY[l] + uy * profile * scale - shift_y,
            0.023 + lift - height,
        ];
    }
    Reference { feet, swing, height }
}

/// Analytic IK: foot ball centers (trunk frame) → joint angles in Isaac
/// order `[hips FL,FR,RL,RR | thighs | calves]`.
pub fn ik(feet: &[[f64; 3]; 4]) -> [f64; 12] {
    let mut q = [0.0f64; 12];
    for l in 0..4 {
        let x = feet[l][0] - HIP_X * SX[l];
        let y = feet[l][1] - HIP_Y * SY[l];
        let z = feet[l][2];
        let lateral = LATERAL * SY[l];
        let zp = -(y * y + z * z - lateral * lateral).max(1e-8).sqrt();
        let hip = z.atan2(y) - zp.atan2(lateral);
        let hip = hip.sin().atan2(hip.cos()); // wrap to (−π, π]
        let c = ((x * x + zp * zp - 2.0 * LINK * LINK) / (2.0 * LINK * LINK))
            .clamp(-0.99999, 0.99999);
        let knee = -c.acos();
        let thigh = (-x).atan2(-zp) - knee.sin().atan2(1.0 + knee.cos());
        q[l] = hip;
        q[4 + l] = thigh;
        q[8 + l] = knee;
    }
    q
}

/// Forward kinematics: joint angles (Isaac order) → foot ball centers in the
/// trunk frame. Exact inverse of [`ik`] inside the leg workspace.
pub fn fk(q_isaac: &[f64; 12]) -> [[f64; 3]; 4] {
    let mut feet = [[0.0f64; 3]; 4];
    for l in 0..4 {
        let (hip, thigh, knee) = (q_isaac[l], q_isaac[4 + l], q_isaac[8 + l]);
        let x = -LINK * (thigh.sin() + (thigh + knee).sin());
        let zp = -LINK * (thigh.cos() + (thigh + knee).cos());
        let lateral = LATERAL * SY[l];
        // hip roll about +x applied to the in-plane (lateral, zp) pair
        let y = lateral * hip.cos() - zp * hip.sin();
        let z = lateral * hip.sin() + zp * hip.cos();
        feet[l] = [x + HIP_X * SX[l], y + HIP_Y * SY[l], z];
    }
    feet
}

/// Foot-position Jacobian of one leg: ∂(foot position, trunk frame)/∂(hip,
/// thigh, knee), a 3×3 matrix in row-major order. Columns are the three
/// joints of leg `l` (0 = FL, 1 = FR, 2 = RL, 3 = RR).
pub fn leg_jacobian(q_isaac: &[f64; 12], l: usize) -> [[f64; 3]; 3] {
    let (hip, thigh, knee) = (q_isaac[l], q_isaac[4 + l], q_isaac[8 + l]);
    let lateral = LATERAL * SY[l];
    let (sh, ch) = hip.sin_cos();
    let st1 = thigh.sin();
    let ct1 = thigh.cos();
    let st12 = (thigh + knee).sin();
    let ct12 = (thigh + knee).cos();
    let zp = -LINK * (ct1 + ct12);
    // partials of the in-plane coordinates
    let dx_dthigh = -LINK * (ct1 + ct12);
    let dx_dknee = -LINK * ct12;
    let dzp_dthigh = LINK * (st1 + st12);
    let dzp_dknee = LINK * st12;
    // y = lateral·cos(hip) − zp·sin(hip); z = lateral·sin(hip) + zp·cos(hip)
    [
        [0.0, dx_dthigh, dx_dknee],
        [
            -lateral * sh - zp * ch,
            -dzp_dthigh * sh,
            -dzp_dknee * sh,
        ],
        [
            lateral * ch - zp * sh,
            dzp_dthigh * ch,
            dzp_dknee * ch,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FK(IK(feet)) reproduces the reference feet everywhere on the gait.
    #[test]
    fn fk_inverts_ik_over_the_whole_gait() {
        let cfg = TrajectoryCfg::h30_standard();
        for i in 0..200 {
            let t = i as f64 * 0.05;
            let r = trajectory(t, [0.12, 0.05, 0.2], &cfg);
            let q = ik(&r.feet);
            let feet = fk(&q);
            for l in 0..4 {
                for k in 0..3 {
                    assert!(
                        (feet[l][k] - r.feet[l][k]).abs() < 1e-9,
                        "t={t} leg {l} axis {k}: fk {} vs ref {}",
                        feet[l][k],
                        r.feet[l][k]
                    );
                }
            }
        }
    }

    /// The analytic Jacobian matches central finite differences of FK.
    #[test]
    fn jacobian_matches_finite_differences() {
        let cfg = TrajectoryCfg::h30_standard();
        let r = trajectory(3.3, [0.1, -0.05, 0.25], &cfg);
        let q = ik(&r.feet);
        let h = 1e-6;
        for l in 0..4 {
            let jac = leg_jacobian(&q, l);
            for (col, jidx) in [l, 4 + l, 8 + l].into_iter().enumerate() {
                let mut qp = q;
                qp[jidx] += h;
                let mut qm = q;
                qm[jidx] -= h;
                let fp = fk(&qp)[l];
                let fm = fk(&qm)[l];
                for row in 0..3 {
                    let fd = (fp[row] - fm[row]) / (2.0 * h);
                    assert!(
                        (jac[row][col] - fd).abs() < 1e-6,
                        "leg {l} row {row} col {col}: {} vs fd {}",
                        jac[row][col],
                        fd
                    );
                }
            }
        }
    }

    /// Standing (zero command): the reference is the static stance at the
    /// commanded height, no swing, forever.
    #[test]
    fn standing_reference_is_static() {
        let cfg = TrajectoryCfg::h30_standard();
        let a = trajectory(0.0, [0.0; 3], &cfg);
        let b = trajectory(12.34, [0.0; 3], &cfg);
        assert_eq!(a, b);
        assert_eq!(a.swing, [false; 4]);
        assert!((a.height - 0.30).abs() < 1e-12);
        for l in 0..4 {
            assert!((a.feet[l][2] - (0.023 - 0.30)).abs() < 1e-12);
        }
    }

    /// One foot at a time: while walking, at most one leg is in swing.
    #[test]
    fn at_most_one_leg_swings() {
        let cfg = TrajectoryCfg::h30_standard();
        for i in 0..400 {
            let t = 1.5 + i as f64 * 0.02;
            let r = trajectory(t, [0.12, 0.0, 0.0], &cfg);
            assert!(r.swing.iter().filter(|s| **s).count() <= 1, "t={t}");
        }
    }
}

#[cfg(test)]
mod height_tests {
    use super::*;

    /// The schedule reproduces the two calibration points it was fitted to,
    /// and the standard config is unchanged.
    #[test]
    fn yaw_gain_schedule_matches_calibration() {
        assert!((TrajectoryCfg::yaw_gain_for_height(0.30) - 2.0).abs() < 1e-12);
        assert!((TrajectoryCfg::yaw_gain_for_height(0.21) - 1.6).abs() < 2e-3);
        let std = TrajectoryCfg::h30_standard();
        assert_eq!(std.with_height(0.30), std);
    }

    /// The low-stance config is the standard one at 0.22 m with the
    /// recalibrated yaw gain, everything else identical.
    #[test]
    fn low_stance_is_standard_at_022() {
        let low = TrajectoryCfg::h30_low_stance();
        let std = TrajectoryCfg::h30_standard();
        assert_eq!(low.body_height, Some(0.22));
        assert!((low.yaw_stride_gain.unwrap() - 1.644).abs() < 1e-3);
        assert_eq!(low.stride_gain, std.stride_gain);
        assert_eq!(low.support_shift_gain, std.support_shift_gain);
    }

    /// Heights outside the measured range are clamped, not extrapolated.
    #[test]
    fn height_is_clamped_to_the_measured_range() {
        assert_eq!(
            TrajectoryCfg::h30_standard().with_height(0.05).body_height,
            Some(MIN_BODY_HEIGHT)
        );
        assert_eq!(
            TrajectoryCfg::h30_standard().with_height(0.50).body_height,
            Some(MAX_BODY_HEIGHT)
        );
    }

    /// Standing at the low stance puts every foot at the commanded depth.
    #[test]
    fn low_stance_standing_reference_is_at_the_commanded_height() {
        let r = trajectory(0.0, [0.0; 3], &TrajectoryCfg::h30_low_stance());
        assert!((r.height - 0.22).abs() < 1e-12);
        for l in 0..4 {
            assert!((r.feet[l][2] - (0.023 - 0.22)).abs() < 1e-12);
        }
    }
}
