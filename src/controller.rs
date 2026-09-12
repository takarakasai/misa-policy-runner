//! The Natural policy controller — everything between sensors and the MIT
//! joint command, at the two rates the contract prescribes:
//!
//! - [`NaturalController::tick`] at 50 Hz: observation → inference →
//!   reference + residual decode → gain map → 30 ms first-order filter.
//! - [`NaturalController::support_torque`] every low-level tick (500 Hz on
//!   the Go2): the support-wrench feedforward, using the swing mask and
//!   height reference **held from the last policy tick** (Isaac's
//!   process/apply split).
//!
//! The controller owns the gait clock (`t` advances 1/50 s per tick — never
//! wall time, so replays are exact) and the filter state. The host owns the
//! loop, the plant I/O, and the safe-stop decision.

use crate::go2::{
    clamp_cmd, soft_limits_isaac, NATURAL_FILTER_TAU_S, NATURAL_KD, NATURAL_KP,
    NATURAL_RESIDUAL_RAD, POLICY_HZ, TOTAL_MASS_KG,
};
use crate::natural::{ik, trajectory, Reference, TrajectoryCfg, FREQUENCY, START_DELAY};
use crate::obs::{build_base_obs, obs_anomalies, push_clock, ObsInput, N_OBS_CLOCK};
use crate::policy::OnnxPolicy;
use crate::support::{support_torque, BaseState};

/// One policy tick's output: the MIT joint targets in **Isaac order**.
/// (Hosts reorder to their motor order; `go2::ISAAC_TO_GO2` for the Go2.)
#[derive(Debug, Clone)]
pub struct PolicyTick {
    pub q_des_isaac: [f64; 12],
    pub kp_isaac: [f64; 12],
    pub kd_isaac: [f64; 12],
    /// Anomaly names from the observation screen — empty in normal
    /// operation. The host should log every occurrence and stop on a burst.
    pub anomalies: Vec<&'static str>,
}

pub struct NaturalController {
    policy: OnnxPolicy,
    cfg: TrajectoryCfg,
    /// The policy's nominal pose: `ik(trajectory(0, 0))` — also the initial
    /// pose and the joint_pos observation offset.
    default_isaac: [f64; 12],
    /// Gait time (s); advances 1/50 s per tick.
    t: f64,
    /// Filtered targets (the filter state IS the outgoing command).
    q_filt: [f64; 12],
    kp_filt: [f64; 12],
    kd_filt: [f64; 12],
    /// Held for the support feedforward until the next tick.
    swing: [bool; 4],
    height_ref: f64,
}

impl NaturalController {
    /// `policy` must be a 39-d clock-conditioned graph.
    pub fn new(policy: OnnxPolicy, cfg: TrajectoryCfg) -> Result<Self, String> {
        if policy.n_obs() != N_OBS_CLOCK {
            return Err(format!(
                "the Natural contract needs a {N_OBS_CLOCK}-d policy, this graph takes {}",
                policy.n_obs()
            ));
        }
        let r0 = trajectory(0.0, [0.0; 3], &cfg);
        let default_isaac = ik(&r0.feet);
        Ok(Self {
            policy,
            cfg,
            default_isaac,
            t: 0.0,
            q_filt: default_isaac,
            kp_filt: [NATURAL_KP.g0; 12],
            kd_filt: [NATURAL_KD.g0; 12],
            swing: [false; 4],
            height_ref: r0.height,
        })
    }

    /// The pose the robot must be standing in before the first tick — the
    /// policy's a = 0 fixed point. Ramp to this pose at `initial_gains()`
    /// before starting the loop.
    pub fn default_pose_isaac(&self) -> [f64; 12] {
        self.default_isaac
    }

    /// Nominal (a = 0) gains: Kp 45, Kd 2.
    pub fn initial_gains(&self) -> (f64, f64) {
        (NATURAL_KP.g0, NATURAL_KD.g0)
    }

    /// Reset the clock and filters to the standing start. Call when the
    /// robot is standing in the default pose.
    pub fn reset(&mut self) {
        self.t = 0.0;
        self.q_filt = self.default_isaac;
        self.kp_filt = [NATURAL_KP.g0; 12];
        self.kd_filt = [NATURAL_KD.g0; 12];
        self.swing = [false; 4];
        self.height_ref = trajectory(0.0, [0.0; 3], &self.cfg).height;
    }

    /// Gait time since [`Self::reset`] (s).
    pub fn gait_time_s(&self) -> f64 {
        self.t
    }

    /// The current safe hold: keep the last filtered targets. Use this when
    /// [`Self::tick`] errors — freezing the filtered command is continuous,
    /// jumping to any other pose is not.
    pub fn hold(&self) -> PolicyTick {
        PolicyTick {
            q_des_isaac: self.q_filt,
            kp_isaac: self.kp_filt,
            kd_isaac: self.kd_filt,
            anomalies: Vec::new(),
        }
    }

    /// One 50 Hz policy tick. `cmd` is clamped to the trained envelope
    /// inside. On inference failure the clock and state are NOT advanced;
    /// send [`Self::hold`] and decide whether to stop.
    pub fn tick(&mut self, inp: &ObsInput, cmd_raw: [f64; 3]) -> Result<PolicyTick, String> {
        let cmd = clamp_cmd(cmd_raw);
        let mut obs = build_base_obs(inp, &cmd, &self.default_isaac);
        push_clock(&mut obs, self.t, FREQUENCY, START_DELAY);
        let anomalies = obs_anomalies(&obs);
        let action = self.policy.infer(&obs)?;

        // reference + residual, gains, then the shared first-order filter
        let r: Reference = trajectory(self.t, cmd, &self.cfg);
        let reference = ik(&r.feet);
        let dt = 1.0 / POLICY_HZ;
        let alpha = 1.0 - (-dt / NATURAL_FILTER_TAU_S).exp();
        for i in 0..12 {
            let (lo, hi) = soft_limits_isaac(i);
            let q_raw =
                (reference[i] + NATURAL_RESIDUAL_RAD * action[i].tanh()).clamp(lo, hi);
            let kp_raw = NATURAL_KP.apply(action[12 + i]);
            let kd_raw = NATURAL_KD.apply(action[24 + i]);
            self.q_filt[i] += alpha * (q_raw - self.q_filt[i]);
            self.kp_filt[i] += alpha * (kp_raw - self.kp_filt[i]);
            self.kd_filt[i] += alpha * (kd_raw - self.kd_filt[i]);
        }
        self.swing = r.swing;
        self.height_ref = r.height;
        self.t += dt;

        Ok(PolicyTick {
            q_des_isaac: self.q_filt,
            kp_isaac: self.kp_filt,
            kd_isaac: self.kd_filt,
            anomalies,
        })
    }

    /// The per-low-level-tick support-wrench feedforward (Isaac order),
    /// using the swing mask / height reference held from the last policy
    /// tick. `q_isaac` is the CURRENT measured joint state.
    pub fn support_torque(&self, base: &BaseState, cmd_raw: [f64; 3], q_isaac: &[f64; 12]) -> [f64; 12] {
        support_torque(
            base,
            &clamp_cmd(cmd_raw),
            q_isaac,
            &self.swing,
            self.height_ref,
            TOTAL_MASS_KG,
        )
    }

    /// The swing mask held from the last tick — stance = `!swing` is what
    /// the host should feed its contact-assuming estimator.
    pub fn swing(&self) -> [bool; 4] {
        self.swing
    }
}

/// Decode one Loco-contract (37-d) action, for completeness: linear
/// residual around the stock default pose, wider gain range, no reference
/// trajectory and no feedforward.
pub fn decode_loco(action: &[f64; 36]) -> ([f64; 12], [f64; 12], [f64; 12]) {
    use crate::go2::{DEFAULT_LOCO_ISAAC, LOCO_ACTION_SCALE, LOCO_KD, LOCO_KP};
    let mut q = [0.0f64; 12];
    let mut kp = [0.0f64; 12];
    let mut kd = [0.0f64; 12];
    for i in 0..12 {
        let (lo, hi) = soft_limits_isaac(i);
        q[i] = (DEFAULT_LOCO_ISAAC[i] + LOCO_ACTION_SCALE * action[i]).clamp(lo, hi);
        kp[i] = LOCO_KP.apply(action[12 + i]);
        kd[i] = LOCO_KD.apply(action[24 + i]);
    }
    (q, kp, kd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::go2::{LOCO_KD, LOCO_KP};

    /// Zero action decodes to exactly the default pose at nominal gains.
    #[test]
    fn loco_zero_action_is_the_fixed_point() {
        let (q, kp, kd) = decode_loco(&[0.0; 36]);
        assert_eq!(q, crate::go2::DEFAULT_LOCO_ISAAC);
        assert!(kp.iter().all(|v| *v == LOCO_KP.g0));
        assert!(kd.iter().all(|v| *v == LOCO_KD.g0));
    }

    /// Saturated actions pin the gains to the contract clamps and the
    /// targets to the soft joint limits.
    #[test]
    fn loco_saturated_action_hits_the_clamps() {
        let (q, kp, kd) = decode_loco(&[1000.0; 36]);
        for i in 0..12 {
            let (_, hi) = soft_limits_isaac(i);
            assert_eq!(q[i], hi, "q[{i}]");
            assert_eq!(kp[i], LOCO_KP.max);
            assert_eq!(kd[i], LOCO_KD.max);
        }
    }
}
