#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cd rust_pipeline

cargo run -- ../data/worldbank_panel_final.csv ../data/worldbank_panel_engineered_rust.csv
