//! Go2 deployment constants shared by every module: joint-order tables,
//! trained gain maps, hardware limits, and the command envelope.
//!
//! Values are the MIT-mode deploy contract (go2_rl `doc/mit_deploy_contract.md`
//! and `doc/mit_natural.md`); the joint tables match quadruped-gait's
//! `policy-runtime` crate.

/// Isaac orders the 12 joints by TYPE (all hips, all thighs, all calves,
/// FL/FR/RL/RR within each); the Go2 SDK orders by LEG (FR, FL, RR, RL ×
/// hip/thigh/calf).
///
/// Go2 SDK motor index for each Isaac joint index (reorder an ACTION out).
pub const ISAAC_TO_GO2: [usize; 12] = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8];
/// Isaac joint index for each Go2 SDK motor index (build the OBSERVATION).
pub const GO2_TO_ISAAC: [usize; 12] = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10];

/// Leg order used by the trajectory / IK / support-wrench math: FL, FR, RL, RR
/// (the Isaac within-type order). Go2 SDK leg base index for each.
pub const LEG_TO_GO2_BASE: [usize; 4] = [3, 0, 9, 6];

/// The stock Go2 default pose in Isaac order — the Loco (37-d) policy's
/// nominal pose and joint_pos offset. **The Natural policy does NOT use
/// this**: its default pose is `ik(trajectory(0, 0))`.
pub const DEFAULT_LOCO_ISAAC: [f64; 12] = [
    0.1, -0.1, 0.1, -0.1, // hips FL,FR,RL,RR
    0.8, 0.8, 1.0, 1.0, // thighs
    -1.5, -1.5, -1.5, -1.5, // calves
];

/// Go2 hardware joint limits (rad), indexed hip/thigh/calf (`go2_index % 3`).
/// From `go2.misa`; identical to the MuJoCo model's ranges.
pub const JOINT_LIMITS: [(f64, f64); 3] = [
    (-1.0472, 1.0472),   // hip
    (-1.5708, 3.4907),   // thigh
    (-2.7227, -0.83776), // calf
];

/// Isaac's soft-joint-limit convention: mid ± factor · half-range. The
/// runner re-applies this clamp itself (the network is unbounded).
pub const SOFT_LIMIT_FACTOR: f64 = 0.9;

/// Soft joint limit for an Isaac-order joint index.
pub fn soft_limits_isaac(isaac_idx: usize) -> (f64, f64) {
    let (lo, hi) = JOINT_LIMITS[ISAAC_TO_GO2[isaac_idx] % 3];
    let mid = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo) * SOFT_LIMIT_FACTOR;
    (mid - half, mid + half)
}

/// Real per-joint torque envelope (N·m) the policy trained against, Isaac
/// order: hips/thighs 23.7, calves 45.43. `saturation == effort_limit`, so
/// the DCMotor curve degenerates to the shape training saw.
pub fn effort_limit_isaac(isaac_idx: usize) -> f64 {
    if isaac_idx >= 8 {
        45.43
    } else {
        23.7
    }
}
/// DCMotor velocity limit (rad/s), all joints.
pub const VELOCITY_LIMIT: f64 = 30.0;

/// Exact port of isaaclab's `DCMotor._clip_effort` for one joint, with
/// `saturation_effort == effort_limit` (the trained configuration).
pub fn dc_motor_clip(effort: f64, joint_vel: f64, effort_limit: f64) -> f64 {
    let vel_at_effort_lim = VELOCITY_LIMIT * 2.0; // v_lim · (1 + e/e_sat)
    let v = joint_vel.clamp(-vel_at_effort_lim, vel_at_effort_lim);
    let top = (effort_limit * (1.0 - v / VELOCITY_LIMIT)).min(effort_limit);
    let bottom = (effort_limit * (-1.0 - v / VELOCITY_LIMIT)).max(-effort_limit);
    effort.clamp(bottom, top)
}

/// Variable-gain map `g = g0 + scale · a`, clamped to `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainMap {
    pub g0: f64,
    pub scale: f64,
    pub min: f64,
    pub max: f64,
}

impl GainMap {
    pub fn apply(&self, a: f64) -> f64 {
        (self.g0 + self.scale * a).clamp(self.min, self.max)
    }
}

/// Natural contract gains: Kp = 45 + 5a ∈ [35, 60], Kd = 2 + 0.2a ∈ [1.5, 3].
pub const NATURAL_KP: GainMap = GainMap { g0: 45.0, scale: 5.0, min: 35.0, max: 60.0 };
pub const NATURAL_KD: GainMap = GainMap { g0: 2.0, scale: 0.2, min: 1.5, max: 3.0 };
/// Natural residual: `q_ref = ik(trajectory) + 0.035 · tanh(a_pos)`.
pub const NATURAL_RESIDUAL_RAD: f64 = 0.035;
/// Natural target/gain first-order filter time constant (s), at 50 Hz.
pub const NATURAL_FILTER_TAU_S: f64 = 0.03;

/// Loco (37-d base) contract gains: Kp = 25 + 12.5a ∈ [10, 60],
/// Kd = 0.5 + 0.5a ∈ [0.1, 2.5]; `q_des = default + 0.25 · a_pos` (linear).
pub const LOCO_KP: GainMap = GainMap { g0: 25.0, scale: 12.5, min: 10.0, max: 60.0 };
pub const LOCO_KD: GainMap = GainMap { g0: 0.5, scale: 0.5, min: 0.1, max: 2.5 };
pub const LOCO_ACTION_SCALE: f64 = 0.25;

/// Policy inference rate (Isaac decimation 4 × sim dt 0.005 s).
pub const POLICY_HZ: f64 = 50.0;

/// Natural h30 training command envelope (STANDARD checkpoint,
/// `Isaac-MIT-NaturalH30-Go2-v0`): outside these ranges is
/// out-of-distribution — the runner clamps.
pub const CMD_VX_RANGE: (f64, f64) = (-0.16, 0.16);
pub const CMD_VY_RANGE: (f64, f64) = (-0.10, 0.10);
pub const CMD_WZ_RANGE: (f64, f64) = (-0.40, 0.40);

/// Clamp a `[vx, vy, wz]` command to the trained envelope.
pub fn clamp_cmd(mut c: [f64; 3]) -> [f64; 3] {
    c[0] = c[0].clamp(CMD_VX_RANGE.0, CMD_VX_RANGE.1);
    c[1] = c[1].clamp(CMD_VY_RANGE.0, CMD_VY_RANGE.1);
    c[2] = c[2].clamp(CMD_WZ_RANGE.0, CMD_WZ_RANGE.1);
    c
}

/// Total robot mass (kg) the support-wrench feedforward was trained with —
/// the MuJoCo deploy model's `body_mass.sum()` (go2-gait-runner scene).
pub const TOTAL_MASS_KG: f64 = 15.206408;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isaac_go2_tables_are_mutual_inverses() {
        for g in 0..12 {
            assert_eq!(ISAAC_TO_GO2[GO2_TO_ISAAC[g]], g, "go2 {g}");
        }
        for i in 0..12 {
            assert_eq!(GO2_TO_ISAAC[ISAAC_TO_GO2[i]], i, "isaac {i}");
        }
    }

    /// The leg bases are the hip motor of each leg, FL,FR,RL,RR — the same
    /// legs the Isaac hip block (indices 0..4) maps to.
    #[test]
    fn leg_bases_match_the_isaac_hip_block() {
        for (leg, base) in LEG_TO_GO2_BASE.iter().enumerate() {
            assert_eq!(ISAAC_TO_GO2[leg], *base, "leg {leg}");
        }
    }

    /// Gain maps clamp both ways and are exact at a = 0.
    #[test]
    fn gain_maps_match_the_contract() {
        assert_eq!(NATURAL_KP.apply(0.0), 45.0);
        assert_eq!(NATURAL_KP.apply(100.0), 60.0);
        assert_eq!(NATURAL_KP.apply(-100.0), 35.0);
        assert_eq!(NATURAL_KD.apply(0.0), 2.0);
        assert_eq!(LOCO_KP.apply(0.0), 25.0);
        assert_eq!(LOCO_KD.apply(-100.0), 0.1);
    }

    /// Soft limits: mid ± 0.9·half of the hardware range, per joint type.
    #[test]
    fn soft_limits_shrink_the_hardware_range() {
        for i in 0..12 {
            let (hlo, hhi) = JOINT_LIMITS[ISAAC_TO_GO2[i] % 3];
            let (slo, shi) = soft_limits_isaac(i);
            assert!(slo > hlo && shi < hhi, "isaac {i}");
            assert!((0.5 * (slo + shi) - 0.5 * (hlo + hhi)).abs() < 1e-12);
        }
    }

    /// DCMotor clip: static torque passes through, the envelope shrinks with
    /// speed in the direction of motion, and reverses sign past ±v_limit.
    #[test]
    fn dc_motor_clip_matches_the_python_port() {
        // matches sim2sim_mit_go2_mujoco.dc_motor_clip semantics
        assert_eq!(dc_motor_clip(10.0, 0.0, 23.7), 10.0);
        assert_eq!(dc_motor_clip(100.0, 0.0, 23.7), 23.7);
        // at v = +15 (half the limit) the positive envelope halves: 23.7·0.5
        assert!((dc_motor_clip(100.0, 15.0, 23.7) - 11.85).abs() < 1e-12);
        // at v = +30 the positive envelope is 0
        assert_eq!(dc_motor_clip(100.0, 30.0, 23.7), 0.0);
        // negative side is unchanged at positive speed until -2·e_lim
        assert_eq!(dc_motor_clip(-100.0, 15.0, 23.7), -23.7);
    }
}
