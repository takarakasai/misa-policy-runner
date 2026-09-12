//! Support-wrench feedforward — the part of the Natural contract that runs
//! every low-level tick, not every policy tick.
//!
//! Port of `MITNaturalJointAction.apply_actions` (verified NumPy port:
//! `natural_support_torque` in go2_rl `sim2sim_mit_go2_mujoco.py`): a desired
//! body wrench from velocity/height/orientation PD → least-squares
//! distribution over STANCE feet (0.001-regularized) → unilateral fz in
//! [0, 120] N per foot and a 0.6 friction-pyramid clamp → feedforward
//! `τ = −Jᵀ f` through each foot's Jacobian.
//!
//! Frames: the wrench and the per-foot forces live in the body frame under a
//! small-tilt assumption (exactly as trained). The Python reference rotates
//! the foot forces to the world frame and applies them through the world
//! Jacobian; because the joint columns of the world Jacobian are `R · J_b`,
//! the rotations cancel and `τ = −J_bᵀ f_b` — which is what this computes.

use nalgebra::{Matrix6, Vector6};

use crate::natural::{fk, leg_jacobian};

/// Constants of the trained feedforward (mit_rl/actions_mit.py).
const PLANAR_KP: f64 = 5.0;
const HEIGHT_KP: f64 = 100.0;
const HEIGHT_KD: f64 = 16.0;
const ATT_KP: f64 = 180.0;
const ATT_KD: f64 = 12.0;
const YAW_KP: f64 = 5.0;
const TOTAL_FZ_MAX_N: f64 = 250.0;
const FOOT_FZ_MAX_N: f64 = 120.0;
const FRICTION: f64 = 0.6;
const REGULARIZER: f64 = 0.001;

/// Base-state inputs the feedforward needs, per low-level tick.
#[derive(Debug, Clone, Copy)]
pub struct BaseState {
    /// Orientation quaternion `(w, x, y, z)`, body → world.
    pub quat_wxyz: [f64; 4],
    /// Base linear velocity in the **world** frame (estimated), m/s.
    pub vel_world: [f64; 3],
    /// Base angular velocity, body frame (gyro), rad/s.
    pub gyro_rad_s: [f64; 3],
    /// Estimated base height above the floor, m.
    pub height_m: f64,
}

/// `R(q)ᵀ v` — world vector into the body frame.
pub fn quat_rotate_inverse(q: &[f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let qv = [x, y, z];
    let a = 2.0 * w * w - 1.0;
    let cross = [
        qv[1] * v[2] - qv[2] * v[1],
        qv[2] * v[0] - qv[0] * v[2],
        qv[0] * v[1] - qv[1] * v[0],
    ];
    let dot = qv[0] * v[0] + qv[1] * v[1] + qv[2] * v[2];
    [
        v[0] * a - 2.0 * w * cross[0] + 2.0 * qv[0] * dot,
        v[1] * a - 2.0 * w * cross[1] + 2.0 * qv[1] * dot,
        v[2] * a - 2.0 * w * cross[2] + 2.0 * qv[2] * dot,
    ]
}

/// Projected gravity `R(q)ᵀ (0,0,−1)` — the attitude error signal.
pub fn projected_gravity(q: &[f64; 4]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        2.0 * (w * y - x * z),
        -2.0 * (y * z + w * x),
        2.0 * (x * x + y * y) - 1.0,
    ]
}

/// Compute the feedforward joint torques (Isaac order).
///
/// `q_isaac` are the measured joint angles; `swing` and `height_ref` are the
/// values held from the last policy tick — exactly Isaac's process/apply
/// split. `total_mass_kg` is [`crate::go2::TOTAL_MASS_KG`] for the Go2.
pub fn support_torque(
    state: &BaseState,
    cmd: &[f64; 3],
    q_isaac: &[f64; 12],
    swing: &[bool; 4],
    height_ref_m: f64,
    total_mass_kg: f64,
) -> [f64; 12] {
    let vel_b = quat_rotate_inverse(&state.quat_wxyz, state.vel_world);
    let g_b = projected_gravity(&state.quat_wxyz);
    let m = total_mass_kg;

    // desired body wrench (body frame, small-tilt)
    let force = [
        m * PLANAR_KP * (cmd[0] - vel_b[0]),
        m * PLANAR_KP * (cmd[1] - vel_b[1]),
        (m * (9.81 + HEIGHT_KP * (height_ref_m - state.height_m)
            - HEIGHT_KD * state.vel_world[2]))
            .clamp(0.0, TOTAL_FZ_MAX_N),
    ];
    let moment = [
        ATT_KP * g_b[1] - ATT_KD * state.gyro_rad_s[0], // small-angle roll = −g_y
        -ATT_KP * g_b[0] - ATT_KD * state.gyro_rad_s[1], // small-angle pitch = g_x
        YAW_KP * (cmd[2] - state.gyro_rad_s[2]),
    ];

    // wrench map W: per-foot force → body wrench ([I; [r]×] blocks), swing
    // legs zeroed. Solve f = Wᵀ (W Wᵀ + εI)⁻¹ wrench.
    let feet = fk(q_isaac);
    let mut wwt = Matrix6::<f64>::zeros();
    // A is 6×12 but only the 6×6 normal matrix and Aᵀy are ever formed.
    let mut cols = [[0.0f64; 6]; 12]; // A's columns
    for l in 0..4 {
        if swing[l] {
            continue;
        }
        let r = feet[l];
        // columns for foot l: force component k contributes I at row k and
        // [r]× at rows 3..6
        let rx = [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ];
        for k in 0..3 {
            cols[3 * l + k][k] = 1.0;
            for row in 0..3 {
                cols[3 * l + k][3 + row] = rx[row][k];
            }
        }
    }
    for c in cols.iter() {
        for i in 0..6 {
            for j in 0..6 {
                wwt[(i, j)] += c[i] * c[j];
            }
        }
    }
    for i in 0..6 {
        wwt[(i, i)] += REGULARIZER;
    }
    let wrench = Vector6::new(force[0], force[1], force[2], moment[0], moment[1], moment[2]);
    let lambda = wwt
        .lu()
        .solve(&wrench)
        .unwrap_or_else(Vector6::zeros);

    // per-foot forces, then the unilateral + friction-pyramid clamps
    let mut f = [[0.0f64; 3]; 4];
    for l in 0..4 {
        for k in 0..3 {
            let c = &cols[3 * l + k];
            for i in 0..6 {
                f[l][k] += c[i] * lambda[i];
            }
        }
        f[l][2] = f[l][2].clamp(0.0, FOOT_FZ_MAX_N);
        let lim = FRICTION * f[l][2];
        f[l][0] = f[l][0].clamp(-lim, lim);
        f[l][1] = f[l][1].clamp(-lim, lim);
    }

    // τ = −J_bᵀ f_b per leg, into Isaac joint slots (hip l, thigh 4+l, calf 8+l)
    let mut tau = [0.0f64; 12];
    for l in 0..4 {
        let j = leg_jacobian(q_isaac, l);
        for (col, slot) in [l, 4 + l, 8 + l].into_iter().enumerate() {
            tau[slot] = -(j[0][col] * f[l][0] + j[1][col] * f[l][1] + j[2][col] * f[l][2]);
        }
    }
    tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::natural::{ik, trajectory, TrajectoryCfg};

    fn standing_state(height: f64) -> (BaseState, [f64; 12]) {
        let cfg = TrajectoryCfg::h30_standard();
        let r = trajectory(0.0, [0.0; 3], &cfg);
        let q = ik(&r.feet);
        (
            BaseState {
                quat_wxyz: [1.0, 0.0, 0.0, 0.0],
                vel_world: [0.0; 3],
                gyro_rad_s: [0.0; 3],
                height_m: height,
            },
            q,
        )
    }

    /// Level standing at the reference height: the requested wrench is pure
    /// weight, so each foot carries ~mg/4 and the knee feedforward is
    /// extensor-signed (negative knee angle, positive-x moment arm).
    #[test]
    fn standing_equilibrium_supports_the_weight() {
        let (state, q) = standing_state(0.30);
        let tau = support_torque(&state, &[0.0; 3], &q, &[false; 4], 0.30, 15.2064);
        // symmetric stance: same magnitude on all four calves
        for l in 1..4 {
            assert!((tau[8 + l] - tau[8]).abs() < 1e-9, "calf {l}");
        }
        // the feedforward does real work (order of mg·arm, not zero)
        assert!(tau[8].abs() > 1.0, "calf tau {}", tau[8]);
        // hips carry the roll moment of the vertical load (the feet stand
        // laterally outside the hip axes): equal magnitude on all four,
        // mirrored between the left (sy = +1) and right (sy = −1) side.
        for l in 1..4 {
            assert!((tau[l].abs() - tau[0].abs()).abs() < 1e-9, "hip {l}");
        }
        assert!(tau[0].abs() > 1.0, "hip tau {}", tau[0]);
        assert!((tau[0] - tau[2]).abs() < 1e-9, "left hips equal");
        assert!((tau[1] - tau[3]).abs() < 1e-9, "right hips equal");
        assert!((tau[0] + tau[1]).abs() < 1e-9, "left/right mirrored");
    }

    /// Swing legs get exactly zero feedforward.
    #[test]
    fn swing_legs_carry_nothing() {
        let (state, q) = standing_state(0.30);
        let tau = support_torque(&state, &[0.0; 3], &q, &[true, false, false, false], 0.30, 15.2064);
        assert_eq!(tau[0], 0.0);
        assert_eq!(tau[4], 0.0);
        assert_eq!(tau[8], 0.0);
        assert!(tau[9].abs() > 1.0, "stance legs still work");
    }

    /// Dropping below the height reference raises the total vertical demand
    /// (bounded by the 250 N clamp).
    #[test]
    fn height_error_raises_the_vertical_demand() {
        let (low, q) = standing_state(0.25);
        let (nom, _) = standing_state(0.30);
        let t_low = support_torque(&low, &[0.0; 3], &q, &[false; 4], 0.30, 15.2064);
        let t_nom = support_torque(&nom, &[0.0; 3], &q, &[false; 4], 0.30, 15.2064);
        // extensor knees push harder when the body sags
        assert!(t_low[8].abs() > t_nom[8].abs());
    }

    /// All-swing (airborne): no stance feet, the solve degenerates through
    /// the regularizer to zero forces, zero torque.
    #[test]
    fn airborne_produces_zero_feedforward() {
        let (state, q) = standing_state(0.30);
        let tau = support_torque(&state, &[0.0; 3], &q, &[true; 4], 0.30, 15.2064);
        for (i, t) in tau.iter().enumerate() {
            assert_eq!(*t, 0.0, "tau[{i}]");
        }
    }

    /// A forward-velocity error adds a forward force request, which shifts
    /// the thigh torques away from the pure-weight solution.
    #[test]
    fn velocity_error_changes_the_thigh_torques() {
        let (nom, q) = standing_state(0.30);
        let (mut err, _) = standing_state(0.30);
        err.vel_world = [-0.2, 0.0, 0.0]; // moving backward while cmd forward
        let t_nom = support_torque(&nom, &[0.0; 3], &q, &[false; 4], 0.30, 15.2064);
        let t_err = support_torque(&err, &[0.12, 0.0, 0.0], &q, &[false; 4], 0.30, 15.2064);
        for l in 0..4 {
            assert!(
                (t_err[4 + l] - t_nom[4 + l]).abs() > 0.05,
                "thigh {l}: {} vs {}",
                t_err[4 + l],
                t_nom[4 + l]
            );
        }
    }
}
