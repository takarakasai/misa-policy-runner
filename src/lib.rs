//! misa-policy-runner — deploy-side runtime for the MIT-mode variable-gain
//! RL locomotion policies trained in go2_rl (`mit_rl/`).
//!
//! **The exported ONNX is not the controller** — the Natural contract
//! (go2_rl `doc/mit_natural.md`) is: a four-beat reference trajectory with
//! analytic IK, a bounded residual `0.035·tanh(a_pos)` on top of it,
//! variable PD gains `Kp = 45 ± 5 ∈ [35, 60]` / `Kd = 2 ± 0.2 ∈ [1.5, 3]`,
//! a 30 ms first-order filter on targets and gains, and a per-low-level-tick
//! support-wrench feedforward `τ_ff = −Jᵀ f` distributed over the stance
//! feet. This crate implements all of it, verified against the Python
//! reference implementations by golden vectors and property tests.
//!
//! What this crate deliberately does NOT contain: plant I/O (DDS, serial,
//! MuJoCo), piloting, and state estimation. Hosts adapt their sensor source
//! to [`obs::ObsInput`] / [`support::BaseState`] and map the Isaac-order
//! outputs to their motor order — `go2-runner` does exactly that for the
//! Unitree Go2 on top of misa-runner.
//!
//! The STANDARD deploy checkpoint (2026-09-12) is
//! `Isaac-MIT-NaturalH30-Go2-v0` (walking height 0.30 m, stride gains
//! 1.55 / 2.0): [`natural::TrajectoryCfg::h30_standard`]. **The trajectory
//! gains and body height MUST match the checkpoint's training config** — a
//! mismatch walks, badly, and nothing raises an error.
//!
//! Rates: inference at [`go2::POLICY_HZ`] (50 Hz) with zero-order hold; the
//! host's low-level loop (500 Hz LowCmd on the Go2) recomputes the MIT
//! torque from the held targets every tick and re-evaluates
//! [`controller::NaturalController::support_torque`].

pub mod blind;
pub mod controller;
pub mod go2;
pub mod gru;
pub mod heading;
pub mod namiashi;
pub mod namiashi_ref;
pub mod natural;
pub mod obs;
pub mod policy;
pub mod pure;
pub mod support;

pub use blind::{clamp_blind_cmd, BlindController};
pub use controller::{decode_loco, NaturalController, PolicyTick};
pub use gru::{clamp_gru_cmd, HistoryVelocityEstimator, PureGruController, VelocitySource};
pub use pure::{clamp_pure_cmd, PureController};
pub use natural::TrajectoryCfg;
pub use obs::ObsInput;
pub use policy::{graph_input_arity, OnnxPolicy, RecurrentOnnxPolicy};
pub use support::BaseState;
