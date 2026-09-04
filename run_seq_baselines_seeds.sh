#!/bin/bash
#SBATCH --job-name=sbl_seeds
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/sbl_seeds_%A_%a.log
#SBATCH --error=slurm_logs/sbl_seeds_err_%A_%a.log

# 3-seed run of the sequence-task baselines (Fixed-Depth Transformer /
# Weight-Tied Transformer) for logic and family, so tab:baselines can report
# mean+/-std on those rows too (matching the graph rows).
#
# Full per-task budget (--full): logic 64k/30, family 60k/40 -- identical to
# the existing baseline_results.json, so seed 42 reproduces it.
# Writes <task>/baseline_results.json (seed 42) and
#        <task>/baseline_results_s{43,44}.json.
#
# Submit from the repo root with:  mkdir -p slurm_logs && sbatch run_seq_baselines_seeds.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

TASKS=(logic logic logic family family family)
SEEDS=(42 43 44 42 43 44)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASKS[$IDX]}"
SEED="${SEEDS[$IDX]}"

echo "[START] sbl_seeds task=$TASK seed=$SEED | $(date) | host=$(hostname)"
"$PY" seq_baselines.py "$TASK" --full --seed "$SEED"
echo "[END] sbl_seeds task=$TASK seed=$SEED | $(date)"
