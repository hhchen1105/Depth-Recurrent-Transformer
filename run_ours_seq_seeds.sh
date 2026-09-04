#!/bin/bash
#SBATCH --job-name=ours_seq
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/ours_seq_%A_%a.log
#SBATCH --error=slurm_logs/ours_seq_err_%A_%a.log

# 3-seed runs of the "ours" model on the two sequence tasks (logic, family) so
# tab:baselines can report a 3-seed mean+/-std on the Depth-Recurrent row for
# those tasks too (graph already comes from the 3-seed supervision sweep).
#
# Same hyper-parameters / budget as the primary runs -- only the seed changes.
# Each element writes:
#   nested-expr/logic_grid_s<seed>.npy   (+ logic_model_s<seed>.pt for seed!=42)
#   family-reason/family_results_s<seed>.npy
# seed 42 also refreshes the canonical *_results.pdf; the paper figures come from
# plot_results.py and are unaffected. make_baseline_table.py picks the grids up
# automatically.
#
# Submit from the repo root with:  mkdir -p slurm_logs && sbatch run_ours_seq_seeds.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

DIRS=(nested-expr nested-expr nested-expr family-reason family-reason family-reason)
SCRIPTS=(logic_experiment.py logic_experiment.py logic_experiment.py \
         family_experiment.py family_experiment.py family_experiment.py)
SEEDS=(42 43 44 42 43 44)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DIR="${DIRS[$IDX]}"
SCRIPT="${SCRIPTS[$IDX]}"
SEED="${SEEDS[$IDX]}"

echo "[START] ours_seq dir=$DIR script=$SCRIPT seed=$SEED | $(date) | host=$(hostname)"
(cd "$DIR" && "$PY" "$SCRIPT" --seed "$SEED")
echo "[END] ours_seq dir=$DIR seed=$SEED | $(date)"
