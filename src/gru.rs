//! The **recurrent** pure-network contract: a GRU actor with an explicit
//! 128-d hidden state, plus a frozen six-frame velocity estimator.
//!
//! Trained in go2_rl `train_history_policy.py` and exported by
//! `export_gru_policy.py`; the runtime rules below are the ones audited in
//! go2_rl `doc/gru_runtime_contract_audit_20260920.md`. The reference
//! implementation is `sim2sim_mit_go2_mujoco.py --pure76-gru
//! --velocity-estimator …`.
//!
//! What differs from [`crate::pure`] (which is the MLP form of the same
//! 76-input observation) — every one of these is load-bearing:
//!
//! 1. **The graph takes two inputs and returns two outputs.** Input width
//!    alone does NOT identify the contract: Pure76 and the GRU are both 76.
//!    Dispatch on [`crate::policy::graph_input_arity`].
//! 2. **The runtime carries state.** `hidden_out` of tick *k* is
//!    `hidden_in` of tick *k+1*; it is zeroed on reset, together with the
//!    `last_action` feedback and the velocity history.
//! 3. **The velocity slot `[73..76)` comes from a frozen estimator**, not
//!    from leg odometry: six oldest-first 73-d frames (base 37 +
//!    `last_action` 36) flattened to 438 → body-frame `(vx, vy, vz)`. At
//!    startup the ring is filled with six copies of the current frame.
//!    Feeding leg odometry instead is a *different input distribution* —
//!    it walks in MuJoCo, but it is not this contract (the audit measured
//!    0.287 → 0.306 m/s at a 0.3 m/s command), so it needs the explicit
//!    [`VelocitySource::Host`] opt-in.
//! 4. **The command envelope is the checkpoint's own**: vx ∈ [−0.16, 2.0],
//!    **vy ≡ 0** (it was never trained with lateral commands), wz ∈ ±0.8 —
//!    not Pure's ±0.30 lateral / 1.0 forward.
//!
//! Everything else is shared with Pure: the 37-d base observation, the
//! `±100` clip on the action feedback, the H30 crouch default pose, and the
//! decode `q = default + 0.25·a`, `Kp = 45 + 12.5·a`, `Kd = 2 + 0.5·a`.
//! There is no reference trajectory, no gait clock and no support-wrench
//! feedforward.

use crate::controller::PolicyTick;
use crate::natural::{ik, trajectory, TrajectoryCfg};
use crate::obs::{build_base_obs, obs_anomalies, ObsInput, N_ACT};
use crate::policy::{OnnxPolicy, RecurrentOnnxPolicy};
use crate::pure::{decode_pure, PURE_KD, PURE_KP};

/// Observation width of the GRU actor (identical to Pure76 — not a
/// discriminator, see the module docs).
pub const N_OBS_GRU: usize = 76;
/// GRU hidden width (`export_gru_policy.py`: `nn.GRU(76, 128, 1)`).
pub const N_HIDDEN_GRU: usize = 128;
/// Frames in the velocity estimator's ring, oldest first.
pub const VEL_FRAMES: usize = 6;
/// Features per frame: the 73-d prefix of the observation (base + feedback).
pub const VEL_FEATURES: usize = 73;
/// Flattened estimator input width.
pub const N_VEL_HISTORY: usize = VEL_FRAMES * VEL_FEATURES;
/// The `last_action` observation term's clip (Isaac `clip=(-100, 100)`).
const LAST_ACTION_CLIP: f64 = 100.0;

/// Trained command envelope of the adopted checkpoint
/// (`outputs/history_gru_v1/curriculum_force10_stage2_v1/model_49`, its
/// `env.yaml`: `lin_vel_x (-0.16, 2.0)`, `lin_vel_y (0, 0)`,
/// `ang_vel_z (-0.8, 0.8)`).
///
/// As with [`crate::pure::clamp_pure_cmd`] the envelope belongs to the
/// CHECKPOINT and the ONNX carries no metadata about it. **vy is pinned to
/// zero** because the checkpoint never saw a lateral command; a non-zero vy
/// is not "worse tracking", it is an observation value the network has
/// never been trained on.
pub fn clamp_gru_cmd(c: [f64; 3]) -> [f64; 3] {
    [c[0].clamp(-0.16, 2.0), 0.0, c[2].clamp(-0.80, 0.80)]
}

/// Roll the six-frame ring: `None` (startup / just reset) fills all six
/// slots with the current frame, exactly like the Python reference's
/// `np.repeat(frame, 6)`; otherwise drop the oldest and append.
fn rolled(
    previous: Option<&[[f32; VEL_FEATURES]; VEL_FRAMES]>,
    frame: &[f32],
) -> [[f32; VEL_FEATURES]; VEL_FRAMES] {
    let mut f = [[0.0f32; VEL_FEATURES]; VEL_FRAMES];
    match previous {
        None => {
            for slot in f.iter_mut() {
                slot.copy_from_slice(frame);
            }
        }
        Some(prev) => {
            f[..VEL_FRAMES - 1].copy_from_slice(&prev[1..]);
            f[VEL_FRAMES - 1].copy_from_slice(frame);
        }
    }
    f
}

/// The frozen six-frame body-velocity estimator (438 → 3). Normalization is
/// folded into the exported graph, so the input is raw SI observation
/// frames.
pub struct HistoryVelocityEstimator {
    model: OnnxPolicy,
    frames: Option<[[f32; VEL_FEATURES]; VEL_FRAMES]>,
}

impl HistoryVelocityEstimator {
    pub fn load(path: &str) -> Result<Self, String> {
        let model = OnnxPolicy::load(path, N_VEL_HISTORY)?;
        Ok(Self {
            model,
            frames: None,
        })
    }

    /// Drop the history; the next frame refills all six slots.
    pub fn reset(&mut self) {
        self.frames = None;
    }

    /// Push `frame` (73-d) and estimate. The ring is advanced **only on
    /// success**, so a failed tick leaves the runtime exactly as it was.
    pub fn estimate(&mut self, frame: &[f32]) -> Result<[f64; 3], String> {
        if frame.len() != VEL_FEATURES {
            return Err(format!("velocity frame {} != {VEL_FEATURES}", frame.len()));
        }
        let next = rolled(self.frames.as_ref(), frame);
        let mut flat = Vec::with_capacity(N_VEL_HISTORY);
        for f in next.iter() {
            flat.extend_from_slice(f);
        }
        let v = self.model.infer_n(&flat, 3)?;
        self.frames = Some(next);
        Ok([v[0], v[1], v[2]])
    }
}

/// Where the observation's `[73..76)` body-velocity slot comes from.
pub enum VelocitySource {
    /// The contract: the frozen six-frame estimator ONNX.
    Estimator(HistoryVelocityEstimator),
    /// Diagnostics only: whatever the host passes to [`PureGruController::tick`]
    /// (go2-runner's leg odometry). A different input distribution — see
    /// the module docs.
    Host,
}

pub struct PureGruController {
    policy: RecurrentOnnxPolicy,
    velocity: VelocitySource,
    default_isaac: [f64; 12],
    /// The previous RAW network output (observation term, clipped ±100).
    last_action: [f64; N_ACT],
    /// The recurrent state carried across ticks; zeros at reset.
    hidden: Vec<f32>,
    q_hold: [f64; 12],
    kp_hold: [f64; 12],
    kd_hold: [f64; 12],
    /// Last velocity actually fed to the network (status display / logging).
    vel_used: [f64; 3],
    /// The last observation actually sent to the graph. Kept so hosts (and
    /// the parity harness) can diff the assembled 76-d vector against the
    /// Python reference term by term — a wrongly assembled observation still
    /// yields a perfectly finite action.
    last_obs: Vec<f32>,
    /// Wall-of-the-gait time for display only (advances 1/50 s per tick).
    t: f64,
}

impl PureGruController {
    /// `policy` must be a 76-input / 128-hidden recurrent graph.
    pub fn new(policy: RecurrentOnnxPolicy, velocity: VelocitySource) -> Result<Self, String> {
        if policy.n_obs() != N_OBS_GRU {
            return Err(format!(
                "the GRU contract needs a {N_OBS_GRU}-input graph, this one takes {}",
                policy.n_obs()
            ));
        }
        if policy.n_hidden() != N_HIDDEN_GRU {
            return Err(format!(
                "the GRU contract needs a {N_HIDDEN_GRU}-d hidden state, this one has {}",
                policy.n_hidden()
            ));
        }
        // Same H30 crouch as the Pure contract: the checkpoint was trained
        // standing at 0.30 m.
        let r0 = trajectory(0.0, [0.0; 3], &TrajectoryCfg::h30_standard());
        let default_isaac = ik(&r0.feet);
        Ok(Self {
            policy,
            velocity,
            default_isaac,
            last_action: [0.0; N_ACT],
            hidden: vec![0.0; N_HIDDEN_GRU],
            q_hold: default_isaac,
            kp_hold: [PURE_KP.g0; 12],
            kd_hold: [PURE_KD.g0; 12],
            vel_used: [0.0; 3],
            last_obs: Vec::new(),
            t: 0.0,
        })
    }

    /// True when the host must supply the body-velocity estimate itself
    /// ([`VelocitySource::Host`]); false when the frozen estimator owns it.
    pub fn wants_host_velocity(&self) -> bool {
        matches!(self.velocity, VelocitySource::Host)
    }

    /// The pose to ramp into before the first tick (the a = 0 fixed point).
    pub fn default_pose_isaac(&self) -> [f64; 12] {
        self.default_isaac
    }

    /// Nominal (a = 0) gains: Kp 45, Kd 2.
    pub fn initial_gains(&self) -> (f64, f64) {
        (PURE_KP.g0, PURE_KD.g0)
    }

    /// The full reset rule of the contract: hidden state 0, action feedback
    /// 0, velocity history dropped (the next tick refills it with six copies
    /// of the current frame). Call when the robot is standing in the default
    /// pose.
    pub fn reset(&mut self) {
        self.last_action = [0.0; N_ACT];
        self.hidden = vec![0.0; N_HIDDEN_GRU];
        if let VelocitySource::Estimator(e) = &mut self.velocity {
            e.reset();
        }
        self.q_hold = self.default_isaac;
        self.kp_hold = [PURE_KP.g0; 12];
        self.kd_hold = [PURE_KD.g0; 12];
        self.vel_used = [0.0; 3];
        self.last_obs.clear();
        self.t = 0.0;
    }

    pub fn gait_time_s(&self) -> f64 {
        self.t
    }

    /// The body-frame velocity fed to the network on the last successful
    /// tick.
    pub fn velocity_used(&self) -> [f64; 3] {
        self.vel_used
    }

    /// The 76-d observation of the last successful tick (empty before the
    /// first one).
    pub fn last_observation(&self) -> &[f32] {
        &self.last_obs
    }

    /// The RAW network output of the last successful tick — what the next
    /// observation feeds back, before any decode.
    pub fn last_action_raw(&self) -> &[f64; N_ACT] {
        &self.last_action
    }

    /// The recurrent state carried into the next tick.
    pub fn hidden_state(&self) -> &[f32] {
        &self.hidden
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

    /// One 50 Hz policy tick. `vel_body_host` is used **only** with
    /// [`VelocitySource::Host`]; with the estimator it is ignored. On
    /// failure NOTHING is advanced — not the hidden state, not the action
    /// feedback, not the velocity history — so the caller can send
    /// [`Self::hold`] and retry on the next tick from a consistent state.
    pub fn tick(
        &mut self,
        inp: &ObsInput,
        cmd_raw: [f64; 3],
        vel_body_host: [f64; 3],
    ) -> Result<PolicyTick, String> {
        let cmd = clamp_gru_cmd(cmd_raw);
        let mut obs = build_base_obs(inp, &cmd, &self.default_isaac);
        let anomalies = obs_anomalies(&obs); // screen the 37-d base
        for a in self.last_action.iter() {
            obs.push(a.clamp(-LAST_ACTION_CLIP, LAST_ACTION_CLIP) as f32);
        }
        // The estimator reads the 73-d prefix (base + feedback) — the
        // velocity slot is what it produces, so it is appended after.
        let vel = match &mut self.velocity {
            VelocitySource::Estimator(e) => e.estimate(&obs[..VEL_FEATURES])?,
            VelocitySource::Host => vel_body_host,
        };
        for v in vel.iter() {
            obs.push(*v as f32);
        }
        let (action, hidden) = self.policy.infer(&obs, &self.hidden)?;
        let mut raw = [0.0f64; N_ACT];
        raw.copy_from_slice(&action);
        let (q, kp, kd) = decode_pure(&raw, &self.default_isaac);
        self.last_action = raw;
        self.hidden = hidden;
        self.last_obs = obs;
        self.vel_used = vel;
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

    /// The checkpoint's envelope: forward to 2.0, backward to −0.16, yaw
    /// ±0.8, and lateral pinned to zero whatever is asked for.
    #[test]
    fn gru_cmd_envelope() {
        assert_eq!(clamp_gru_cmd([9.0, 9.0, 9.0]), [2.0, 0.0, 0.80]);
        assert_eq!(clamp_gru_cmd([-9.0, -9.0, -9.0]), [-0.16, 0.0, -0.80]);
        assert_eq!(clamp_gru_cmd([0.5, 0.2, 0.1]), [0.5, 0.0, 0.1]);
    }

    /// Startup fills all six slots with the current frame; later frames
    /// shift the ring left, oldest first.
    #[test]
    fn velocity_history_ring_matches_the_python_reference() {
        let mut frame = [0.0f32; VEL_FEATURES];
        frame[0] = 1.0;
        let first = rolled(None, &frame);
        assert!(
            first.iter().all(|f| f[0] == 1.0),
            "startup repeats the frame"
        );

        let mut next = [0.0f32; VEL_FEATURES];
        next[0] = 2.0;
        let second = rolled(Some(&first), &next);
        assert_eq!(second[0][0], 1.0);
        assert_eq!(second[VEL_FRAMES - 2][0], 1.0);
        assert_eq!(second[VEL_FRAMES - 1][0], 2.0, "newest is last");

        // Six pushes flush the startup fill completely.
        let mut ring = second;
        for k in 3..=VEL_FRAMES + 1 {
            let mut f = [0.0f32; VEL_FEATURES];
            f[0] = k as f32;
            ring = rolled(Some(&ring), &f);
        }
        let expect: Vec<f32> = (2..=VEL_FRAMES + 1).map(|k| k as f32).collect();
        let got: Vec<f32> = ring.iter().map(|f| f[0]).collect();
        assert_eq!(got, expect);
    }
}
