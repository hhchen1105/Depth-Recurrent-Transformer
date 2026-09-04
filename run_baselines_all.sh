#!/bin/bash
#SBATCH --job-name=baselines_all
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/baselines_all_%j.log
#SBATCH --error=slurm_logs/baselines_all_err_%j.log

# Fresh full-budget baseline rerun against the CURRENT experiment scripts, for
# the paper's Table (tab:baselines). Regenerates:
#   graph/baseline_results.json          (fixed-depth GNN / Graph Transformer /
#                                         Weight-Tied Transformer)
#   nested-expr/baseline_results.json    (fixed-depth Transformer / Weight-Tied Transformer)
#   family-reason/baseline_results.json  (same)
# then prints the LaTeX rows.
#
# Submit from the repo root with:  mkdir -p slurm_logs && sbatch run_baselines_all.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

echo "[START] baselines_all | $(date) | host=$(hostname)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"

echo "### graph baselines"
(cd graph && "$PY" baselines.py) 2>&1 | tee graph/baseline_run.log

echo "### logic baselines"
"$PY" seq_baselines.py logic  --full 2>&1 | tee nested-expr/baseline_run.log

echo "### family baselines"
"$PY" seq_baselines.py family --full 2>&1 | tee family-reason/baseline_run.log

echo "### aggregate"
"$PY" make_baseline_table.py

echo "======================================================================"
echo "[END] baselines_all | $(date)"
