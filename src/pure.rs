//! The PURE-network contract (go2_rl `doc/mit_pure.md`): the runtime is
//! nothing but obs → ONNX → a trivial affine decode. No reference
//! trajectory, no IK, no gait clock, no command filter, no support-wrench
//! feedforward — the network is the whole controller.
//!
//! Two graph widths exist:
//!
//! - 73 = the 37-d base observation + the previous RAW network output
//!   (`last_action`, clipped ±100; zeros at reset).
//! - 76 = 73 + an estimated **body-frame** linear velocity `(vx, vy, vz)`
//!   appended last. On hardware this comes from leg odometry — it is a
//!   sensor-style estimate fed to the network, not a controller.
//!
//! The verified 1 m/s checkpoint (2026-09-13,
//! `go2_mit_pure_vel_speed/…_pure76_speed100_lr1e4`) is the 76 form:
//! sim2sim 0.4 → 97%, 0.7 → 104%, 1.0 → 104% on the realistic-damping
//! plant. Decode (Isaac order, `a = [a_pos|a_kp|a_kd]`):
//!
//! ```text
//! q_ref = default_crouch + 0.25·a_pos      (soft joint limits)
//! Kp    = clamp(45 + 12.5·a_kp, 10, 60)
//! Kd    = clamp( 2 +  0.5·a_kd, 0.5, 3.5)
//! ```
//!
//! `default_crouch` is `ik(trajectory(0, 0))` at the H30 standard height
//! (0.30 m) — the same crouch the Natural contract stands in, so the same
//! ramp-in procedure applies.

use crate::controller::PolicyTick;
use crate::go2::{soft_limits_isaac, GainMap};
use crate::natural::{ik, trajectory, TrajectoryCfg};
use crate::obs::{build_base_obs, obs_anomalies, ObsInput, N_ACT};
use crate::policy::OnnxPolicy;

/// 73-input graph: base 37 + last_action 36.
pub const N_OBS_PURE: usize = 73;
/// 76-input graph: 73 + body-frame velocity estimate (vx, vy, vz).
pub const N_OBS_PURE_VEL: usize = 76;

/// Position residual scale (rad per unit action) around the crouch default.
pub const PURE_POS_SCALE: f64 = 0.25;
/// Pure contract gains: Kp = 45 + 12.5a ∈ [10, 60].
pub const PURE_KP: GainMap = GainMap { g0: 45.0, scale: 12.5, min: 10.0, max: 60.0 };
/// Kd = 2 + 0.5a ∈ [0.5, 3.5].
pub const PURE_KD: GainMap = GainMap { g0: 2.0, scale: 0.5, min: 0.5, max: 3.5 };
/// The last_action observation term's clip (Isaac `clip=(-100, 100)`).
const LAST_ACTION_CLIP: f64 = 100.0;

/// Trained command envelope of the pure76 speed100 checkpoint:
/// vx ∈ [−0.16, 1.0], vy ∈ ±0.10, wz ∈ ±0.40 (its params/env.yaml).
pub fn clamp_pure_cmd(c: [f64; 3]) -> [f64; 3] {
    [
        c[0].clamp(-0.16, 1.0),
        c[1].clamp(-0.10, 0.10),
        c[2].clamp(-0.40, 0.40),
    ]
}

/// Decode one pure-contract action around `default_isaac` (no filter — the
/// decode IS the outgoing command).
pub fn decode_pure(
    action: &[f64; N_ACT],
    default_isaac: &[f64; 12],
) -> ([f64; 12], [f64; 12], [f64; 12]) {
    let mut q = [0.0f64; 12];
    let mut kp = [0.0f64; 12];
    let mut kd = [0.0f64; 12];
    for i in 0..12 {
        let (lo, hi) = soft_limits_isaac(i);
        q[i] = (default_isaac[i] + PURE_POS_SCALE * action[i]).clamp(lo, hi);
        kp[i] = PURE_KP.apply(action[12 + i]);
        kd[i] = PURE_KD.apply(action[24 + i]);
    }
    (q, kp, kd)
}

pub struct PureController {
    policy: OnnxPolicy,
    default_isaac: [f64; 12],
    /// The previous RAW network output — the network's one-step memory.
    last_action: [f64; N_ACT],
    /// Last decoded command, kept for [`Self::hold`].
    q_hold: [f64; 12],
    kp_hold: [f64; 12],
    kd_hold: [f64; 12],
    /// Wall-of-the-gait time for display only (advances 1/50 s per tick).
    t: f64,
}

impl PureController {
    /// `policy` must be a 73- or 76-input pure graph.
    pub fn new(policy: OnnxPolicy) -> Result<Self, String> {
        if policy.n_obs() != N_OBS_PURE && policy.n_obs() != N_OBS_PURE_VEL {
            return Err(format!(
                "the pure contract needs a {N_OBS_PURE}- or {N_OBS_PURE_VEL}-input graph, \
                 this one takes {}",
                policy.n_obs()
            ));
        }
        // The crouch default: identical to the Natural H30 stand (stride
        // gains are irrelevant at zero command / t = 0).
        let r0 = trajectory(0.0, [0.0; 3], &TrajectoryCfg::h30_standard());
        let default_isaac = ik(&r0.feet);
        Ok(Self {
            policy,
            default_isaac,
            last_action: [0.0; N_ACT],
            q_hold: default_isaac,
            kp_hold: [PURE_KP.g0; 12],
            kd_hold: [PURE_KD.g0; 12],
            t: 0.0,
        })
    }

    /// Whether the graph expects the appended body-velocity estimate (76).
    pub fn wants_velocity(&self) -> bool {
        self.policy.n_obs() == N_OBS_PURE_VEL
    }

    /// The pose to ramp into before the first tick (the a = 0 fixed point).
    pub fn default_pose_isaac(&self) -> [f64; 12] {
        self.default_isaac
    }

    /// Nominal (a = 0) gains: Kp 45, Kd 2.
    pub fn initial_gains(&self) -> (f64, f64) {
        (PURE_KP.g0, PURE_KD.g0)
    }

    /// Zero the one-step memory and return to the standing hold. Call when
    /// the robot is standing in the default pose.
    pub fn reset(&mut self) {
        self.last_action = [0.0; N_ACT];
        self.q_hold = self.default_isaac;
        self.kp_hold = [PURE_KP.g0; 12];
        self.kd_hold = [PURE_KD.g0; 12];
        self.t = 0.0;
    }

    pub fn gait_time_s(&self) -> f64 {
        self.t
    }

    /// Safe hold: the last decoded command (continuous under a freeze).
    pub fn hold(&self) -> PolicyTick {
        PolicyTick {
            q_des_isaac: self.q_hold,
            kp_isaac: self.kp_hold,
            kd_isaac: self.kd_hold,
            anomalies: Vec::new(),
        }
    }

    /// One 50 Hz policy tick. `vel_body` is the body-frame linear velocity
    /// estimate (m/s) — used only by 76-input graphs, pass the leg-odometry
    /// value (or zeros for a 73 graph; it is ignored there). On failure the
    /// memory is NOT advanced; send [`Self::hold`].
    pub fn tick(
        &mut self,
        inp: &ObsInput,
        cmd_raw: [f64; 3],
        vel_body: [f64; 3],
    ) -> Result<PolicyTick, String> {
        let cmd = clamp_pure_cmd(cmd_raw);
        let mut obs = build_base_obs(inp, &cmd, &self.default_isaac);
        let anomalies = obs_anomalies(&obs); // screen the 37-d base
        for a in self.last_action.iter() {
            obs.push(a.clamp(-LAST_ACTION_CLIP, LAST_ACTION_CLIP) as f32);
        }
        if self.wants_velocity() {
            for v in vel_body.iter() {
                obs.push(*v as f32);
            }
        }
        let action = self.policy.infer(&obs)?;
        let (q, kp, kd) = decode_pure(&action, &self.default_isaac);
        self.last_action = action;
        self.q_hold = q;
        self.kp_hold = kp;
        self.kd_hold = kd;
        self.t += 1.0 / crate::go2::POLICY_HZ;
        Ok(PolicyTick {
            q_des_isaac: q,
            kp_isaac: kp,
            kd_isaac: kd,
            anomalies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero action decodes to exactly the crouch default at nominal gains.
    #[test]
    fn pure_zero_action_is_the_fixed_point() {
        let r0 = trajectory(0.0, [0.0; 3], &TrajectoryCfg::h30_standard());
        let default = ik(&r0.feet);
        let (q, kp, kd) = decode_pure(&[0.0; N_ACT], &default);
        assert_eq!(q, default);
        assert!(kp.iter().all(|v| *v == PURE_KP.g0));
        assert!(kd.iter().all(|v| *v == PURE_KD.g0));
    }

    /// Saturated actions pin gains to the clamps and targets to soft limits.
    #[test]
    fn pure_saturated_action_hits_the_clamps() {
        let r0 = trajectory(0.0, [0.0; 3], &TrajectoryCfg::h30_standard());
        let default = ik(&r0.feet);
        let (q, kp, kd) = decode_pure(&[1000.0; N_ACT], &default);
        for i in 0..12 {
            let (_, hi) = soft_limits_isaac(i);
            assert_eq!(q[i], hi, "q[{i}]");
            assert_eq!(kp[i], PURE_KP.max);
            assert_eq!(kd[i], PURE_KD.max);
        }
    }

    /// The command clamp is the trained envelope of the pure76 checkpoint.
    #[test]
    fn pure_cmd_envelope() {
        assert_eq!(clamp_pure_cmd([2.0, 2.0, 2.0]), [1.0, 0.10, 0.40]);
        assert_eq!(clamp_pure_cmd([-2.0, -2.0, -2.0]), [-0.16, -0.10, -0.40]);
    }
}
