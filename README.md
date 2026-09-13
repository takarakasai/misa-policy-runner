# misa-policy-runner

go2_rl（Isaac Lab, `mit_rl/`）で学習した **MIT モード可変ゲイン歩行方策**の
デプロイ側実行系。**ONNX 単体はコントローラではない**——Natural 契約
（go2_rl `doc/mit_natural.md` / `doc/mit_deploy_contract.md`）の全部を
この crate が持つ:

- `natural_walk.py` の四拍歩容軌道 + 解析 IK / FK / ヤコビアン（Rust 移植。
  Python 実装とゴールデンベクタで照合済み — `tests/golden_natural.rs`）
- 39 次元観測の組み立て（Isaac 関節順、クォータニオン w≥0 正準化、
  クロック sin/cos、妥当性スクリーン）
- tract-onnx 推論（純 Rust。C 依存が無いので Go2 の aarch64 に素直に乗る）
- 行動デコード: `q_ref = IK(軌道) + 0.035·tanh(a_pos)`、
  `Kp = 45±5 ∈ [35,60]`、`Kd = 2±0.2 ∈ [1.5,3]`、30 ms 一次フィルタ
- 毎 low-level 周期の支持脚レンチ・フィードフォワード
  `τ_ff = −Jᵀf`（速度/高さ/姿勢 PD → スタンス足への正則化最小二乗配分 →
  片側拘束 + 摩擦ピラミッドのクランプ）

**持たないもの**: 機体接続（DDS・シリアル・MuJoCo）、操縦、状態推定。
ホストがセンサを [`ObsInput`] / [`BaseState`] に詰め、Isaac 順の出力を
自分のモータ順に並べ替える。Go2 では **go2-runner** がその役
（misa-runner の Plant/Backend と束ねる）。

## 標準チェックポイント（2026-09-12 時点）

`Isaac-MIT-NaturalH30-Go2-v0`（歩行高さ 0.30 m）、run
`go2_rl/logs/rsl_rl/go2_mit_natural_h30/2026-09-12_14-27-39_h30_v1/exported/policy.onnx`。
`TrajectoryCfg::h30_standard()` = stride 1.55 / yaw 2.0 / height 0.30 が
**学習時の設定と一致していなければならない**（違っても歩く。悪く。
エラーは出ない）。

## 使い方（50 Hz + 500 Hz の二層）

```rust
let policy = OnnxPolicy::load("policy.onnx", 39)?;
let mut ctl = NaturalController::new(policy, TrajectoryCfg::h30_standard())?;
// 1) default_pose_isaac() へ initial_gains() でランプしてから reset()
loop {                       // 500 Hz low-level
    if tick_50hz {
        match ctl.tick(&obs_input, cmd) {
            Ok(t) => held = t,             // q_des/kp/kd (Isaac 順)
            Err(e) => { held = ctl.hold(); /* 連発なら停止 */ }
        }
    }
    let tau_ff = ctl.support_torque(&base_state, cmd, &q_isaac);
    // held.q_des/kp/kd + tau_ff を MIT 指令として送る（Go2 順へ並べ替え）
}
```

## 検証

```
cargo test                                   # ゴールデンベクタ + 性質テスト
cargo run --release --example selftest -- <policy.onnx>   # 形状・レイテンシ
```

実測（本機）: 推論 mean 17 µs / p99 22 µs（50 Hz スロット 20 ms）。

まだ無い検証: Rust コントローラを MuJoCo（misa-plant-mujoco）で閉ループに
して Python の `sim2sim_mit_go2_mujoco.py --natural-walk` と突き合わせる
比較。go2-runner 側の課題として残っている。

## Pure contract (network-only)

`pure::PureController` implements go2_rl `doc/mit_pure.md`: the runtime is
obs → ONNX → affine decode, nothing else. No reference trajectory, no IK, no
gait clock, no filter, no support-wrench feedforward.

- 73 inputs = the 37-d base observation + the previous raw action (±100 clip,
  zeros at reset).
- 76 inputs = 73 + an estimated body-frame linear velocity, appended last.
  `wants_velocity()` reports which form the loaded graph needs.
- Decode: `q = crouch_default + 0.25·a_pos` (soft limits),
  `Kp = clamp(45 + 12.5·a_kp, 10, 60)`, `Kd = clamp(2 + 0.5·a_kd, 0.5, 3.5)`.
  `crouch_default = ik(trajectory(0,0))` at 0.30 m — the same stand the
  Natural contract ramps into.
- Command envelope (`clamp_pure_cmd`): vx ∈ [−0.16, 1.0], vy ±0.30,
  wz ±0.80 — the trained range of the recommended pure76 checkpoint
  (`rival_wide_dr`). The envelope belongs to the checkpoint, not the
  contract; older pure76 checkpoints used vy ±0.10 / wz ±0.40.

Hosts pick the contract by graph width; `go2-runner`'s `policy` subcommand
does that automatically (39 → Natural, 73/76 → Pure).
