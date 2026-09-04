#!/bin/bash
#SBATCH --job-name=ut_act
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/ut_act_%A_%a.log
#SBATCH --error=slurm_logs/ut_act_err_%A_%a.log

# Genuine Universal Transformer with Adaptive Computation Time (ACT): the real
# Dehghani et al. (2019) per-position halting + ponder cost, NOT the
# weight-sharing-only simplification in baselines.py / seq_baselines.py (which
# is now the "Weight-Tied Transformer" row). 3 seeds (42/43/44) per task.
#
#   element 0 : graph  -> graph/ut_act.py            (GraphIO + adjacency mask)
#   element 1 : logic  -> ut_act_seq.py logic  --full (SeqIO + RoPE, 64k/30)
#   element 2 : family -> ut_act_seq.py family --full (SeqIO + RoPE, 60k/40)
#
# Each element writes <task_dir>/act_results.json, which make_baseline_table.py
# reads to emit the "Universal Transformer (ACT)" row.
#
# Submit with:  mkdir -p slurm_logs && sbatch run_ut_act.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "[START] ut_act element $IDX | $(date) | host=$(hostname)"
nvidia-smi -L || true
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible on this node'; print('cuda OK', torch.cuda.get_device_name(0))"

case "$IDX" in
  0) (cd graph && "$PY" ut_act.py --seeds 42 43 44) ;;
  1) "$PY" ut_act_seq.py logic  --full --seeds 42 43 44 ;;
  2) "$PY" ut_act_seq.py family --full --seeds 42 43 44 ;;
esac

echo "[END] ut_act element $IDX | $(date)"
