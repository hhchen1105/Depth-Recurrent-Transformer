#!/bin/bash
#SBATCH --job-name=embabl_logic
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/embabl_logic_%j.log
#SBATCH --error=slurm_logs/embabl_logic_err_%j.log

# Depth-embedding OOD confound diagnostic -- nested boolean expression task.
# Retrains the headline model once (seed 42), saves the checkpoint
# (nested-expr/logic_model.pt), then runs:
#   Part A: baseline / zero / clamp / reinit(x3 seeds) depth-embedding
#           replacement over the full (nesting-depth x thinking-steps) grid
#   Part B: per-thinking-step update-gate z statistics
# Outputs: nested-expr/embedding_ablation_results.json
#          nested-expr/gate_stats.png  nested-expr/embedding_ablation_summary.png
#
# Submit from the repo root with:  sbatch run_emb_ablation_logic.sh
# Monitor with:            squeue -u $USER
#                          tail -f slurm_logs/embabl_logic_<jobid>.log

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[START] embabl logic | $(date) | host=$(hostname) | commit=${GIT_COMMIT}"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"

(cd nested-expr && "$PY" logic_experiment.py --emb-ablation 2>&1 | tee embedding_ablation_run.log)

echo "======================================================================"
echo "[END] embabl logic complete | $(date)"
