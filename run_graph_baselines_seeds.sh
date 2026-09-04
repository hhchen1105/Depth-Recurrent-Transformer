#!/bin/bash
#SBATCH --job-name=gbl_seeds
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/gbl_seeds_%A_%a.log
#SBATCH --error=slurm_logs/gbl_seeds_err_%A_%a.log

# 3-seed run of the graph baselines (fixed-depth GNN / Graph Transformer /
# Weight-Tied Transformer) for tab:baselines mean+/-std.
# Writes graph/baseline_results.json (seed 42) and graph/baseline_results_s{43,44}.json.
#
# Submit from code/ with:  mkdir -p slurm_logs && sbatch run_graph_baselines_seeds.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

SEEDS=(42 43 44)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "[START] gbl_seeds seed=$SEED | $(date) | host=$(hostname)"
(cd graph && "$PY" baselines.py --seed "$SEED")
echo "[END] gbl_seeds seed=$SEED | $(date)"
