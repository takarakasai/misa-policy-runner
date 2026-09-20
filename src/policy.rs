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

/// 2 入力 2 出力の**再帰**方策グラフ（`observation[1,n_obs]`,
/// `hidden_in[1,1,n_hidden]` → `action[1,n_act]`, `hidden_out[1,1,n_hidden]`）。
///
/// go2_rl `export_gru_policy.py` が出す GRU アクタの形。隠れ状態はグラフの
/// 外（呼び出し側）が持つ — ランタイムが状態を持つので、リセット規則
/// （エピソード開始・コントローラ reset で 0）が契約の一部になる。
pub struct RecurrentOnnxPolicy {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    n_obs: usize,
    n_hidden: usize,
    n_act: usize,
}

impl RecurrentOnnxPolicy {
    /// 読み込んで最適化する。形は固定 batch 1 で束縛する（エクスポート時の
    /// dynamic_axes は batch 方向だけなので、ここで 1 に潰せる）。
    pub fn load(path: &str, n_obs: usize, n_hidden: usize, n_act: usize) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("load onnx {path}: {e}"))?
            .with_input_fact(0, f32::fact([1, n_obs]).into())
            .map_err(|e| format!("input fact (is this really a {n_obs}-d recurrent policy?): {e}"))?
            .with_input_fact(1, f32::fact([1, 1, n_hidden]).into())
            .map_err(|e| format!("hidden fact (expected [1, 1, {n_hidden}]): {e}"))?
            .into_optimized()
            .map_err(|e| format!("optimize: {e}"))?
            .into_runnable()
            .map_err(|e| format!("runnable: {e}"))?;
        Ok(Self { model, n_obs, n_hidden, n_act })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn n_hidden(&self) -> usize {
        self.n_hidden
    }

    /// 1 周期。`hidden_in` は前周期の `hidden_out`（初期は 0）。返すのは
    /// 行動と**新しい隠れ状態**で、非有限はどちらもエラー — 隠れ状態が
    /// 壊れると以後ずっと壊れたままなので、保持へ落とすのは呼び出し側。
    pub fn infer(&self, obs: &[f32], hidden_in: &[f32]) -> Result<(Vec<f64>, Vec<f32>), String> {
        if obs.len() != self.n_obs {
            return Err(format!("obs length {} != {}", obs.len(), self.n_obs));
        }
        if hidden_in.len() != self.n_hidden {
            return Err(format!("hidden length {} != {}", hidden_in.len(), self.n_hidden));
        }
        let obs_t: Tensor =
            tract_ndarray::Array2::<f32>::from_shape_vec((1, self.n_obs), obs.to_vec())
                .map_err(|e| format!("obs shape: {e}"))?
                .into();
        let hidden_t: Tensor = tract_ndarray::Array3::<f32>::from_shape_vec(
            (1, 1, self.n_hidden),
            hidden_in.to_vec(),
        )
        .map_err(|e| format!("hidden shape: {e}"))?
        .into();
        let out = self
            .model
            .run(tvec!(obs_t.into(), hidden_t.into()))
            .map_err(|e| format!("inference: {e}"))?;
        if out.len() != 2 {
            return Err(format!("expected 2 outputs (action, hidden_out), got {}", out.len()));
        }
        let act_view = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("action view: {e}"))?;
        if act_view.len() != self.n_act {
            return Err(format!("expected {} outputs, got {}", self.n_act, act_view.len()));
        }
        let mut action = Vec::with_capacity(self.n_act);
        for v in act_view.iter() {
            if !v.is_finite() {
                return Err("non-finite action".into());
            }
            action.push(*v as f64);
        }
        let hid_view = out[1]
            .to_array_view::<f32>()
            .map_err(|e| format!("hidden view: {e}"))?;
        if hid_view.len() != self.n_hidden {
            return Err(format!(
                "expected {} hidden values, got {}",
                self.n_hidden,
                hid_view.len()
            ));
        }
        let mut hidden_out = Vec::with_capacity(self.n_hidden);
        for v in hid_view.iter() {
            if !v.is_finite() {
                return Err("non-finite hidden state".into());
            }
            hidden_out.push(*v);
        }
        Ok((action, hidden_out))
    }
}

/// グラフの**入力の本数**だけを見る（最適化も形の束縛もしない）。
/// 契約ディスパッチに使う: 1 本 = MLP（Natural / Pure）、2 本 = 再帰（GRU）。
/// 入力幅だけで MLP と GRU を見分けてはいけない — どちらも 76 がありうる。
pub fn graph_input_arity(path: &str) -> Result<usize, String> {
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| format!("load onnx {path}: {e}"))?;
    let inputs = model
        .input_outlets()
        .map_err(|e| format!("input outlets: {e}"))?;
    Ok(inputs.len())
}
