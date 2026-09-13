//! ONNX inference via tract — pure Rust, no C dependencies, so the same
//! binary cross-compiles to the Go2's aarch64 (the reason `policy-runtime`
//! and go2-gait-runner chose tract too).

use tract_onnx::prelude::*;

use crate::obs::N_ACT;

/// A loaded, optimized, runnable MIT-mode policy: `[1, n_obs] → [1, 36]`.
pub struct OnnxPolicy {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    n_obs: usize,
}

impl OnnxPolicy {
    /// Load and optimize an exported policy. `n_obs` must match the graph
    /// (37 for the Loco contract, 39 for the clock-conditioned Natural).
    pub fn load(path: &str, n_obs: usize) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("load onnx {path}: {e}"))?
            .with_input_fact(0, f32::fact([1, n_obs]).into())
            .map_err(|e| format!("input fact (is this really a {n_obs}-d policy?): {e}"))?
            .into_optimized()
            .map_err(|e| format!("optimize: {e}"))?
            .into_runnable()
            .map_err(|e| format!("runnable: {e}"))?;
        Ok(Self { model, n_obs })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// One inference step: observation → 36-d action `[a_pos|a_kp|a_kd]`
    /// (Isaac order). Non-finite outputs are an error — the caller must
    /// fall back to its safe hold, not send the values.
    pub fn infer(&self, obs: &[f32]) -> Result<[f64; N_ACT], String> {
        if obs.len() != self.n_obs {
            return Err(format!("obs length {} != {}", obs.len(), self.n_obs));
        }
        let input: Tensor =
            tract_ndarray::Array2::<f32>::from_shape_vec((1, self.n_obs), obs.to_vec())
                .map_err(|e| format!("obs shape: {e}"))?
                .into();
        let out = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| format!("inference: {e}"))?;
        let view = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("output view: {e}"))?;
        if view.len() != N_ACT {
            return Err(format!("expected {N_ACT} outputs, got {}", view.len()));
        }
        let mut action = [0.0f64; N_ACT];
        for (i, a) in action.iter_mut().enumerate() {
            let v = view[[0, i]];
            if !v.is_finite() {
                return Err(format!("non-finite action[{i}]"));
            }
            *a = v as f64;
        }
        Ok(action)
    }

    /// 出力次元が 36 以外の契約用（namiashi WalkFlatRef は 12）。
    /// 非有限は [`Self::infer`] と同じくエラー。
    pub fn infer_n(&self, obs: &[f32], n_out: usize) -> Result<Vec<f64>, String> {
        if obs.len() != self.n_obs {
            return Err(format!("obs length {} != {}", obs.len(), self.n_obs));
        }
        let input: Tensor =
            tract_ndarray::Array2::<f32>::from_shape_vec((1, self.n_obs), obs.to_vec())
                .map_err(|e| format!("obs shape: {e}"))?
                .into();
        let out = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| format!("inference: {e}"))?;
        let view = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("output view: {e}"))?;
        if view.len() != n_out {
            return Err(format!("expected {n_out} outputs, got {}", view.len()));
        }
        let mut action = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let v = view[[0, i]];
            if !v.is_finite() {
                return Err(format!("non-finite action[{i}]"));
            }
            action.push(v as f64);
        }
        Ok(action)
    }
}
