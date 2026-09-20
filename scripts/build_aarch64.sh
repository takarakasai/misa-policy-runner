#!/usr/bin/env bash
# Go2 の SBC（aarch64）向けに、**sudo なしで**クロスビルドする。
#
#   ./scripts/build_aarch64.sh [--example gru_parity]
#
# なぜ普通に通らないか: tract-linalg は aarch64 の行列カーネルをアセンブラで
# 持っていて、cc-rs 経由で `aarch64-linux-gnu-gcc` を探す。これが無いと
# `cargo check --target aarch64-…` すら落ちる（クロス gcc は apt = sudo が要る）。
#
# 逃げ道: **clang はマルチターゲット**なので、`--target=` を渡せばそのまま
# aarch64 のアセンブラとして使える。アセンブリだけなら libc のヘッダは要らない。
# リンクは musl ターゲットに rustup が同梱する self-contained な crt + libc.a と
# rust-lld で済ませる（glibc の sysroot を用意しなくてよく、出来上がりは
# **静的リンク**なので SBC 側に何も入れずに走る）。
#
# 必要なもの: clang と llvm-ar（Ubuntu 24.04 なら既定で入っている）、
#   rustup target add aarch64-unknown-linux-musl
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=aarch64-unknown-linux-musl
AR=$(command -v llvm-ar || command -v llvm-ar-18 || true)
[ -n "$AR" ] || { echo "llvm-ar が見つかりません" >&2; exit 1; }
command -v clang >/dev/null || { echo "clang が見つかりません" >&2; exit 1; }
rustup target list --installed | grep -qx "$TARGET" || rustup target add "$TARGET"

TOOLCHAIN=$(rustup default | cut -d' ' -f1)
export CC_aarch64_unknown_linux_musl=clang
export CFLAGS_aarch64_unknown_linux_musl="--target=$TARGET"
export AR_aarch64_unknown_linux_musl="$AR"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export PATH="$HOME/.rustup/toolchains/$TOOLCHAIN/lib/rustlib/x86_64-unknown-linux-gnu/bin:$PATH"

cargo build --release --target "$TARGET" "$@"
echo
echo "出来上がり: target/$TARGET/release/"
file target/$TARGET/release/examples/* 2>/dev/null | grep -i aarch64 || true
