#!/bin/bash
#SBATCH --job-name=embabl_family
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/embabl_family_%j.log
#SBATCH --error=slurm_logs/embabl_family_err_%j.log

# Depth-embedding OOD confound diagnostic -- family-relation reasoning task.
# Eval-only: loads the existing family-reason/family_model.pt checkpoint
# (NOT retrained, NOT overwritten) and runs:
#   Part A: baseline / zero / clamp / reinit(x3 seeds) depth-embedding
#           replacement over the full (chain-depth x thinking-steps) grid
#   Part B: per-thinking-step update-gate z statistics
# Outputs: family-reason/embedding_ablation_results.json
#          family-reason/gate_stats.png  family-reason/embedding_ablation_summary.png
#
# Submit from the repo root with:  sbatch run_emb_ablation_family.sh
# Monitor with:            squeue -u $USER
#                          tail -f slurm_logs/embabl_family_<jobid>.log

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[START] embabl family | $(date) | host=$(hostname) | commit=${GIT_COMMIT}"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"

(cd family-reason && "$PY" embedding_ablation.py 2>&1 | tee embedding_ablation_run.log)

echo "======================================================================"
echo "[END] embabl family complete | $(date)"
