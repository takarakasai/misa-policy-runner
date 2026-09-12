//! Observation assembly for the MIT-mode policies (37-d base + optional
//! clock), and the plausibility screen.
//!
//! Layout (raw SI, no scaling — `doc/mit_deploy_contract.md`):
//!
//! | slice     | content                                        |
//! |-----------|------------------------------------------------|
//! | `[0..3)`  | velocity command `[vx, vy, wz]`, body frame    |
//! | `[3..7)`  | orientation quaternion `(w,x,y,z)`, **w ≥ 0**  |
//! | `[7..10)` | angular velocity, body frame (gyro), rad/s     |
//! | `[10..13)`| linear acceleration, body, gravity included, clipped ±30 |
//! | `[13..25)`| joint position − default pose, **Isaac order** |
//! | `[25..37)`| joint velocity, Isaac order                    |
//! | `[37..39)`| `sin, cos` of the gait clock (39-d policies)   |

use crate::go2::GO2_TO_ISAAC;

/// Base observation dimension (Loco contract).
pub const N_OBS_BASE: usize = 37;
/// Clock-conditioned observation dimension (Natural contract).
pub const N_OBS_CLOCK: usize = 39;
/// Action dimension: `[a_pos(12) | a_kp(12) | a_kd(12)]`, Isaac order.
pub const N_ACT: usize = 36;

/// Sensor snapshot in **Go2 SDK conventions** (motor order FR,FL,RR,RL ×
/// hip/thigh/calf; IMU quaternion `w,x,y,z`). Hosts adapt their source:
/// hardware from `LowState`, sim from MuJoCo state.
#[derive(Clone, Copy, Debug)]
pub struct ObsInput {
    /// Base orientation quaternion `(w, x, y, z)`, body → world. Any sign;
    /// the builder canonicalizes to w ≥ 0.
    pub quat_wxyz: [f64; 4],
    /// Base angular velocity, body frame, rad/s.
    pub gyro_rad_s: [f64; 3],
    /// Base linear acceleration, body frame, **gravity included** (reads
    /// `Rᵀ(0,0,+9.81)` at rest, like a real accelerometer), m/s².
    pub accel_m_s2: [f64; 3],
    /// Measured joint positions, Go2 motor order, rad.
    pub joint_q_go2: [f64; 12],
    /// Measured joint velocities, Go2 motor order, rad/s.
    pub joint_dq_go2: [f64; 12],
}

impl Default for ObsInput {
    fn default() -> Self {
        Self {
            quat_wxyz: [1.0, 0.0, 0.0, 0.0],
            gyro_rad_s: [0.0; 3],
            accel_m_s2: [0.0, 0.0, 9.81],
            joint_q_go2: [0.0; 12],
            joint_dq_go2: [0.0; 12],
        }
    }
}

/// The obs term's accelerometer clip (Isaac `clip=(-30, 30)`).
const ACCEL_CLIP: f64 = 30.0;

/// Assemble the 37-d base observation. `default_isaac` is the policy's
/// nominal pose in Isaac order — for the Natural policy that is
/// `ik(trajectory(0, 0))`, NOT the stock Go2 default.
pub fn build_base_obs(
    inp: &ObsInput,
    cmd: &[f64; 3],
    default_isaac: &[f64; 12],
) -> Vec<f32> {
    let mut obs = Vec::with_capacity(N_OBS_CLOCK);
    obs.push(cmd[0] as f32);
    obs.push(cmd[1] as f32);
    obs.push(cmd[2] as f32);
    // quaternion, sign-canonicalized so w >= 0 (quat_unique)
    let s = if inp.quat_wxyz[0] < 0.0 { -1.0 } else { 1.0 };
    for q in inp.quat_wxyz.iter() {
        obs.push((s * q) as f32);
    }
    for g in inp.gyro_rad_s.iter() {
        obs.push(*g as f32);
    }
    for a in inp.accel_m_s2.iter() {
        obs.push(a.clamp(-ACCEL_CLIP, ACCEL_CLIP) as f32);
    }
    let mut jp = [0.0f32; 12];
    let mut jv = [0.0f32; 12];
    for g in 0..12 {
        let i = GO2_TO_ISAAC[g];
        jp[i] = (inp.joint_q_go2[g] - default_isaac[i]) as f32;
        jv[i] = inp.joint_dq_go2[g] as f32;
    }
    obs.extend_from_slice(&jp);
    obs.extend_from_slice(&jv);
    debug_assert_eq!(obs.len(), N_OBS_BASE);
    obs
}

/// Append the gait clock: `sin, cos` of `2π · frequency · max(t − delay, 0)`.
/// The Natural contract uses frequency 0.65 Hz and delay 1.5 s.
pub fn push_clock(obs: &mut Vec<f32>, t_s: f64, frequency_hz: f64, start_delay_s: f64) {
    let phase = 2.0 * std::f64::consts::PI * frequency_hz * (t_s - start_delay_s).max(0.0);
    obs.push(phase.sin() as f32);
    obs.push(phase.cos() as f32);
}

/// Plausibility screen. Returns the (static) names of every violated check —
/// sign/unit/order mistakes and NaNs show up here long before they are
/// debuggable from robot behaviour. Thresholds are generous: anything
/// flagged is *implausible*, not merely unusual.
pub fn obs_anomalies(obs: &[f32]) -> Vec<&'static str> {
    let mut v = Vec::new();
    if obs.len() != N_OBS_BASE && obs.len() != N_OBS_CLOCK {
        v.push("bad_len");
        return v;
    }
    if obs.iter().any(|x| !x.is_finite()) {
        v.push("non_finite");
    }
    let q_norm =
        (obs[3] * obs[3] + obs[4] * obs[4] + obs[5] * obs[5] + obs[6] * obs[6]).sqrt();
    if !(0.9..=1.1).contains(&q_norm) {
        v.push("quat_not_unit");
    }
    if obs[3] < 0.0 {
        v.push("quat_w_negative");
    }
    if obs[7..10].iter().any(|x| x.abs() > 15.0) {
        v.push("gyro_out_of_range");
    }
    // gravity-included accel: |a| far from g means a unit or frame mistake
    let a_norm = (obs[10] * obs[10] + obs[11] * obs[11] + obs[12] * obs[12]).sqrt();
    if !(2.0..=30.5).contains(&a_norm) {
        v.push("accel_implausible");
    }
    if obs[13..25].iter().any(|x| x.abs() > 1.8) {
        v.push("joint_pos_offset_large");
    }
    if obs[25..37].iter().any(|x| x.abs() > 35.0) {
        v.push("joint_vel_large");
    }
    if obs.len() == N_OBS_CLOCK {
        let c_norm = (obs[37] * obs[37] + obs[38] * obs[38]).sqrt();
        if (c_norm - 1.0).abs() > 1e-3 {
            v.push("clock_not_unit");
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::go2::{DEFAULT_LOCO_ISAAC, ISAAC_TO_GO2};

    fn default_input() -> ObsInput {
        let mut inp = ObsInput::default();
        for g in 0..12 {
            inp.joint_q_go2[g] = DEFAULT_LOCO_ISAAC[GO2_TO_ISAAC[g]];
        }
        inp
    }

    /// At rest in the default pose the observation is: cmd, identity quat,
    /// zero rates, gravity on z, zero joint offsets/velocities.
    #[test]
    fn resting_observation_is_the_fixed_point() {
        let obs = build_base_obs(&default_input(), &[0.1, -0.2, 0.3], &DEFAULT_LOCO_ISAAC);
        assert_eq!(obs.len(), N_OBS_BASE);
        assert_eq!(&obs[0..3], &[0.1, -0.2, 0.3]);
        assert_eq!(&obs[3..7], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&obs[7..10], &[0.0, 0.0, 0.0]);
        assert_eq!(&obs[10..13], &[0.0, 0.0, 9.81]);
        for k in 13..37 {
            assert_eq!(obs[k], 0.0, "obs[{k}]");
        }
        assert!(obs_anomalies(&obs).is_empty());
    }

    /// A negative-w quaternion is sign-flipped, not passed through.
    #[test]
    fn quaternion_is_canonicalized_to_w_nonneg() {
        let mut inp = default_input();
        inp.quat_wxyz = [-0.5, 0.5, 0.5, -0.5];
        let obs = build_base_obs(&inp, &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        assert_eq!(&obs[3..7], &[0.5, -0.5, -0.5, 0.5]);
    }

    /// Each Go2 motor lands in its Isaac slot (the reorder that silently
    /// pairs legs wrongly when done by a second loop).
    #[test]
    fn joint_reorder_lands_each_motor_in_its_isaac_slot() {
        let mut inp = default_input();
        for g in 0..12 {
            inp.joint_q_go2[g] += 0.01 * g as f64;
            inp.joint_dq_go2[g] = g as f64;
        }
        let obs = build_base_obs(&inp, &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        for g in 0..12 {
            let i = GO2_TO_ISAAC[g];
            assert!((obs[13 + i] as f64 - 0.01 * g as f64).abs() < 1e-6, "jp {g}");
            assert!((obs[25 + i] as f64 - g as f64).abs() < 1e-6, "jv {g}");
        }
        // and the tables really are mutual inverses
        for i in 0..12 {
            assert_eq!(GO2_TO_ISAAC[ISAAC_TO_GO2[i]], i);
        }
    }

    /// The accelerometer clip is ±30, per the obs term's declaration.
    #[test]
    fn accelerometer_is_clipped() {
        let mut inp = default_input();
        inp.accel_m_s2 = [100.0, -100.0, 9.81];
        let obs = build_base_obs(&inp, &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        assert_eq!(&obs[10..13], &[30.0, -30.0, 9.81]);
    }

    /// The clock starts held at phase 0 through the standing lead-in.
    #[test]
    fn clock_is_held_through_the_lead_in() {
        let mut a = build_base_obs(&default_input(), &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        push_clock(&mut a, 0.7, 0.65, 1.5);
        assert_eq!(a.len(), N_OBS_CLOCK);
        assert_eq!(&a[37..39], &[0.0, 1.0]);
        let mut b = build_base_obs(&default_input(), &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        // one full cycle past the delay: back to (0, 1)
        push_clock(&mut b, 1.5 + 1.0 / 0.65, 0.65, 1.5);
        assert!((b[37] - 0.0).abs() < 1e-5 && (b[38] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn anomaly_screen_flags_the_right_things() {
        let clean = build_base_obs(&default_input(), &[0.0; 3], &DEFAULT_LOCO_ISAAC);
        assert!(obs_anomalies(&clean).is_empty());

        let mut o = clean.clone();
        o[15] = f32::NAN;
        assert!(obs_anomalies(&o).contains(&"non_finite"));

        let mut o = clean.clone();
        o[3] = -0.9;
        assert!(obs_anomalies(&o).contains(&"quat_w_negative"));

        let mut o = clean.clone();
        o[8] = 100.0;
        assert!(obs_anomalies(&o).contains(&"gyro_out_of_range"));

        let mut o = clean.clone();
        o[12] = 0.0;
        assert!(obs_anomalies(&o).contains(&"accel_implausible"));

        assert_eq!(obs_anomalies(&[0.0; 5]), vec!["bad_len"]);
    }
}
